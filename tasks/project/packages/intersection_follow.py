"""Red-line intersection follow: latch leader turn, execute timed arc, resume lane."""

from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

from tasks.project.packages.leader_grid import GridDetection
from tasks.visual_lane_servoing.packages import visual_servoing_activity as _lane_hsv

TURN_STRAIGHT = "straight"
TURN_LEFT = "left"
TURN_RIGHT = "right"


class RedLineProximity(object):
    """Red paint in near (wheel) vs far (horizon) bands — trigger only on near."""

    __slots__ = (
        "near_px", "near_frac", "far_px", "at_line",
        "center_near_frac", "centroid_x_frac",
    )

    def __init__(
        self, near_px, near_frac, far_px, at_line,
        center_near_frac=0.0, centroid_x_frac=0.5,
    ):
        self.near_px = near_px
        self.near_frac = near_frac
        self.far_px = far_px
        self.at_line = at_line
        self.center_near_frac = center_near_frac
        self.centroid_x_frac = centroid_x_frac


def grid_bbox_aspect(det: GridDetection) -> Optional[float]:
    """Width / height of the grid bbox. Face-on 7x3 grid is wide; yawing grid gets squarer."""
    bbox = det.bbox
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    w = float(x2 - x1)
    h = float(y2 - y1)
    if w < 1.0 or h < 1.0:
        return None
    return w / h


class LeaderTurnTracker(object):
    """Guess leader turn from rear-grid shape during red-line approach.

    Primary cue: bbox aspect ratio drops (grid becomes squarer / narrower) when the
    leader yaws — center x often stays near the image middle while following.

    Direction: perspective heading (left vs right dot-column height), then weaker
    heading / cx drift as fallbacks once yaw is confirmed.
    """

    def __init__(self, window=20):
        self._window = max(3, int(window))
        self._cx = deque(maxlen=self._window)  # type: Deque[float]
        self._headings = deque(maxlen=self._window)  # type: Deque[float]
        self._aspects = deque(maxlen=self._window)  # type: Deque[float]
        self._approach_active = False

    def reset(self):
        self._cx.clear()
        self._headings.clear()
        self._aspects.clear()
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
        aspect = grid_bbox_aspect(det)
        if aspect is not None:
            self._aspects.append(aspect)

    def begin_approach_if_needed(self, red_prox, cfg):
        """Clear stale cruise samples when red first shows in the far band."""
        min_far = int(cfg.get("intersection_red_approach_min_far_px", 600))
        approaching = red_prox.far_px >= min_far
        if approaching and not self._approach_active:
            self._cx.clear()
            self._headings.clear()
            self._aspects.clear()
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

    def _cx_vote(self, drift, cfg, scale=1.0):
        thresh = float(cfg.get("intersection_cx_drift_px", 30.0)) * float(scale)
        if drift >= thresh:
            return TURN_RIGHT
        if drift <= -thresh:
            return TURN_LEFT
        return TURN_STRAIGHT

    def _heading_vote(self, cfg, scale=1.0):
        if len(self._headings) < 3:
            return TURN_STRAIGHT
        thresh = float(cfg.get("intersection_heading_thresh", 0.12)) * float(scale)
        sign = float(cfg.get("intersection_heading_sign", -1.0))
        h = sign * float(np.mean(list(self._headings)[-5:]))
        if h >= thresh:
            return TURN_RIGHT
        if h <= -thresh:
            return TURN_LEFT
        return TURN_STRAIGHT

    def _aspect_drop_frac(self, cfg) -> Tuple[float, float, float]:
        """Return (fractional_drop, baseline_aspect, recent_aspect)."""
        aspects = list(self._aspects)
        if len(aspects) < 3:
            return 0.0, 0.0, 0.0
        n_base = max(2, int(cfg.get("intersection_aspect_baseline_frames", 4)))
        baseline = float(np.mean(aspects[: min(n_base, len(aspects))]))
        lookback = max(2, int(cfg.get("intersection_turn_infer_lookback", 8)))
        recent = float(np.mean(aspects[-min(lookback, len(aspects)) :]))
        if baseline < 1e-3:
            return 0.0, baseline, recent
        drop = max(0.0, (baseline - recent) / baseline)
        return drop, baseline, recent

    def _is_yawing(self, drop_frac: float, cfg) -> bool:
        return drop_frac >= float(cfg.get("intersection_aspect_drop_frac", 0.10))

    def infer(self, cfg):
        drop_frac, _baseline, _recent = self._aspect_drop_frac(cfg)
        yawing = self._is_yawing(drop_frac, cfg)

        if not yawing:
            drift = self._cx_drift(cfg)
            cx_vote = self._cx_vote(drift, cfg)
            heading_vote = self._heading_vote(cfg)
            if cx_vote == heading_vote and cx_vote != TURN_STRAIGHT:
                return cx_vote
            return TURN_STRAIGHT

        heading_vote = self._heading_vote(cfg, scale=1.0)
        if heading_vote != TURN_STRAIGHT:
            return heading_vote

        weak = float(cfg.get("intersection_heading_weak_scale", 0.55))
        heading_weak = self._heading_vote(cfg, scale=weak)
        if heading_weak != TURN_STRAIGHT:
            return heading_weak

        cx_weak = float(cfg.get("intersection_cx_drift_weak_scale", 0.5))
        drift = self._cx_drift(cfg)
        cx_vote = self._cx_vote(drift, cfg, scale=cx_weak)
        if cx_vote != TURN_STRAIGHT:
            return cx_vote

        return TURN_STRAIGHT

    def debug_votes(self, cfg):
        drift = self._cx_drift(cfg)
        drop_frac, baseline, recent = self._aspect_drop_frac(cfg)
        yawing = self._is_yawing(drop_frac, cfg)
        return {
            "cx_drift": drift,
            "cx_vote": self._cx_vote(drift, cfg),
            "heading_vote": self._heading_vote(cfg),
            "aspect_baseline": baseline,
            "aspect_recent": recent,
            "aspect_drop_frac": drop_frac,
            "yawing": yawing,
        }


def measure_red_at_line(frame_bgr, cfg):
    """Detect red in the bottom band, centered under the bot (not side paint)."""
    if frame_bgr is None:
        return RedLineProximity(0, 0.0, 0, False)
    try:
        _, _, mask_r = _lane_hsv.detect_lane_markings(frame_bgr)
    except Exception:
        return RedLineProximity(0, 0.0, 0, False)

    h, w = mask_r.shape[:2]
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

    center_left = float(cfg.get("intersection_red_center_left_frac", 0.40))
    center_right = float(cfg.get("intersection_red_center_right_frac", 0.60))
    center_left = min(max(center_left, 0.0), 0.49)
    center_right = min(max(center_right, center_left + 0.05), 1.0)
    x_left = int(near.shape[1] * center_left)
    x_right = max(x_left + 1, int(near.shape[1] * center_right))
    center_near_px = int(np.count_nonzero(near[:, x_left:x_right]))
    center_near_frac = center_near_px / max(1, near_px)

    ys, xs = np.nonzero(near)
    if len(xs) > 0:
        centroid_x_frac = float(np.mean(xs)) / max(1, near.shape[1] - 1)
    else:
        centroid_x_frac = 0.5

    min_near_px = int(cfg.get("intersection_red_near_min_pixels", 3500))
    min_near_frac = float(cfg.get("intersection_red_near_min_frac", 0.055))
    min_ratio = float(cfg.get("intersection_red_near_far_ratio", 2.0))

    ratio = near_px / max(1, far_px)
    base_at_line = (
        near_px >= min_near_px
        and near_frac >= min_near_frac
        and ratio >= min_ratio
    )
    at_line = base_at_line
    if base_at_line and bool(cfg.get("intersection_red_require_centered", True)):
        min_center_frac = float(cfg.get("intersection_red_center_min_frac", 0.40))
        max_offset = float(cfg.get("intersection_red_centroid_max_offset_frac", 0.18))
        centered = (
            center_near_frac >= min_center_frac
            or abs(centroid_x_frac - 0.5) <= max_offset
        )
        at_line = centered
    return RedLineProximity(
        near_px, float(near_frac), far_px, at_line,
        float(center_near_frac), float(centroid_x_frac),
    )


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


def intersection_straight_lane_pwm(lane_agent, frame_bgr, cfg):
    """Steer along the road using the white edge — parallel inside lane, not robot heading.

    Ignores red/yellow so intersection paint does not pull the robot sideways.
    Returns (left_pwm, right_pwm) or None to fall back to blind straight PWM.
    """
    if frame_bgr is None or lane_agent is None:
        return None
    try:
        import cv2
        from tasks.visual_lane_servoing.packages.agent import detect_lines_in_slices

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lane_agent.compute_commands(frame_rgb, use_red=False)
        debug = getattr(lane_agent, "last_debug_info", None) or {}
        mask_w = debug.get("white_mask")
        if mask_w is None:
            return None
        white_px = int(np.count_nonzero(mask_w))
        min_white = int(cfg.get("intersection_straight_min_white_px", 200))
        if white_px < min_white:
            return None

        h, w = mask_w.shape[:2]
        empty = np.zeros_like(mask_w)
        _yellow_xs, white_xs, _red_xs = detect_lines_in_slices(empty, mask_w, h, None)
        if not white_xs:
            return None

        half_w = float(
            cfg.get(
                "intersection_straight_lane_half_width_px",
                getattr(lane_agent, "_lane_half_width", 160.0),
            )
        )
        w_pos = lane_agent._weighted_line_x(white_xs)
        error = (w / 2.0 - (w_pos - half_w)) / max(1.0, w / 2.0)
        error = float(np.clip(error, -1.0, 1.0))
        steering = lane_agent._calculate_steering(error)

        speed = min(1.0, max(0.05, float(cfg.get("intersection_turn_speed", 0.15))))
        left = float(np.clip(speed - steering, 0.0, 1.0))
        right = float(np.clip(speed + steering, 0.0, 1.0))
        return left, right
    except Exception:
        return None


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
