"""Red-line intersection follow: latch leader turn, execute timed arc, resume lane."""

from collections import deque
from typing import Any, Deque, Dict, Tuple

import numpy as np

from tasks.project.packages.leader_grid import GridDetection
from tasks.visual_lane_servoing.packages import visual_servoing_activity as _lane_hsv

TURN_STRAIGHT = "straight"
TURN_LEFT = "left"
TURN_RIGHT = "right"


class RedLineProximity(object):
    """Red paint in near (wheel) vs far (horizon) bands — trigger only on near."""

    __slots__ = ("near_px", "near_frac", "far_px", "at_line")

    def __init__(self, near_px, near_frac, far_px, at_line):
        self.near_px = near_px
        self.near_frac = near_frac
        self.far_px = far_px
        self.at_line = at_line


class LeaderTurnTracker(object):
    """Guess leader turn from rear-grid motion during red-line approach only."""

    def __init__(self, window=20):
        self._window = max(3, int(window))
        self._cx = deque(maxlen=self._window)  # type: Deque[float]
        self._headings = deque(maxlen=self._window)  # type: Deque[float]
        self._approach_active = False

    def reset(self):
        self._cx.clear()
        self._headings.clear()
        self._approach_active = False

    def update(self, det):
        if not det.found:
            return
        if det.center_x is not None:
            self._cx.append(float(det.center_x))
        elif det.centers is not None and len(det.centers) > 0:
            self._cx.append(float(det.centers[0, 0, 0]))
        if det.heading is not None:
            self._headings.append(float(det.heading))

    def begin_approach_if_needed(self, red_prox, cfg):
        """Clear stale cruise drift when red first shows in the far band."""
        min_far = int(cfg.get("intersection_red_approach_min_far_px", 600))
        approaching = red_prox.far_px >= min_far
        if approaching and not self._approach_active:
            self._cx.clear()
            self._headings.clear()
        self._approach_active = approaching

    @property
    def approach_active(self):
        return self._approach_active

    def _cx_drift(self, cfg):
        lookback = max(3, int(cfg.get("intersection_turn_infer_lookback", 8)))
        samples = list(self._cx)
        if len(samples) < 3:
            return 0.0
        recent = samples[-min(lookback, len(samples)):]
        return recent[-1] - recent[0]

    def _cx_vote(self, drift, cfg):
        thresh = float(cfg.get("intersection_cx_drift_px", 30.0))
        if drift >= thresh:
            return TURN_RIGHT
        if drift <= -thresh:
            return TURN_LEFT
        return TURN_STRAIGHT

    def _heading_vote(self, cfg):
        if len(self._headings) < 3:
            return TURN_STRAIGHT
        thresh = float(cfg.get("intersection_heading_thresh", 0.12))
        h = float(np.mean(list(self._headings)[-5:]))
        if h >= thresh:
            return TURN_RIGHT
        if h <= -thresh:
            return TURN_LEFT
        return TURN_STRAIGHT

    def infer(self, cfg):
        """Turn only when recent cx drift and heading agree; else straight."""
        cx_vote = self._cx_vote(self._cx_drift(cfg), cfg)
        heading_vote = self._heading_vote(cfg)
        if cx_vote == heading_vote:
            return cx_vote
        return TURN_STRAIGHT

    def debug_votes(self, cfg):
        drift = self._cx_drift(cfg)
        return drift, self._cx_vote(drift, cfg), self._heading_vote(cfg)


def measure_red_at_line(frame_bgr, cfg):
    """Detect red only in the bottom band (on the paint), not distant intersection preview."""
    if frame_bgr is None:
        return RedLineProximity(0, 0.0, 0, False)
    try:
        _, _, mask_r = _lane_hsv.detect_lane_markings(frame_bgr)
    except Exception:
        return RedLineProximity(0, 0.0, 0, False)

    h, _w = mask_r.shape[:2]
    near_top = float(cfg.get("intersection_red_near_top_frac", 0.72))
    far_top = float(cfg.get("intersection_red_far_top_frac", 0.35))
    near_top = min(max(near_top, 0.55), 0.92)
    far_top = min(max(far_top, 0.20), near_top - 0.05)

    y_near = int(h * near_top)
    y_far = int(h * far_top)
    near = mask_r[y_near:, :]
    far = mask_r[y_far:y_near, :]

    near_px = int(np.count_nonzero(near))
    far_px = int(np.count_nonzero(far))
    near_frac = near_px / max(1, near.size)

    min_near_px = int(cfg.get("intersection_red_near_min_pixels", 3500))
    min_near_frac = float(cfg.get("intersection_red_near_min_frac", 0.055))
    min_ratio = float(cfg.get("intersection_red_near_far_ratio", 2.0))

    ratio = near_px / max(1, far_px)
    at_line = (
        near_px >= min_near_px
        and near_frac >= min_near_frac
        and ratio >= min_ratio
    )
    return RedLineProximity(near_px, float(near_frac), far_px, at_line)


def intersection_turn_duration(direction, cfg):
    key = {
        TURN_STRAIGHT: "intersection_turn_straight_s",
        TURN_LEFT: "intersection_turn_left_s",
        TURN_RIGHT: "intersection_turn_right_s",
    }.get(direction, "intersection_turn_straight_s")
    return max(0.2, float(cfg.get(key, 1.5)))


def intersection_wheel_commands(direction, cfg):
    """Fixed differential PWM for intersection arc, scaled to intersection_turn_speed."""
    turn_speed = min(1.0, max(0.05, float(cfg.get("intersection_turn_speed", 0.15))))
    inner_ratio = float(cfg.get("intersection_turn_inner_ratio", 0.27))
    outer_ratio = float(cfg.get("intersection_turn_outer_ratio", 1.0))
    inner = min(1.0, turn_speed * inner_ratio)
    outer = min(1.0, turn_speed * outer_ratio)
    if direction == TURN_LEFT:
        return inner, outer
    if direction == TURN_RIGHT:
        return outer, inner
    return turn_speed, turn_speed


def lane_frame_ok_for_recovery(lane_agent, cfg):
    """True when yellow/white are back and red paint is mostly behind us."""
    debug = getattr(lane_agent, "last_debug_info", None) or {}
    if not debug.get("lane_detected"):
        return False
    yellow_px = int(np.count_nonzero(debug.get("yellow_mask", 0)))
    white_px = int(np.count_nonzero(debug.get("white_mask", 0)))
    red_px = int(np.count_nonzero(debug.get("red_mask", 0)))
    min_lane = int(cfg.get("intersection_recovery_min_lane_px", 400))
    max_red = int(cfg.get("intersection_recovery_max_red_px", 800))
    min_lane_frac = float(cfg.get("intersection_recovery_min_lane_frac", 0.0))
    if min_lane_frac > 0.0:
        h, w = debug.get("yellow_mask", np.zeros((1, 1))).shape[:2]
        if yellow_px + white_px < int(h * w * min_lane_frac):
            return False
    elif yellow_px + white_px < min_lane:
        return False
    if red_px > max_red:
        return False
    return True
