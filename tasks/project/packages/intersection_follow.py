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


class _LastLeaderSample(object):
    __slots__ = ("center_x", "heading", "aspect", "source")

    def __init__(self, center_x, heading, aspect, source):
        self.center_x = center_x
        self.heading = heading
        self.aspect = aspect
        self.source = source


class LeaderTurnTracker(object):
    """Remember last leader sighting during red approach; infer turn from it.

    Grid gives heading when visible; blue body center-x is kept when the grid
    drops out (common right as the leader yaws at an intersection).
    """

    def __init__(self, window=20):
        self._window = max(3, int(window))
        self._cx = deque(maxlen=self._window)  # type: Deque[float]
        self._last = None  # type: Optional[_LastLeaderSample]
        self._approach_active = False

    def reset(self):
        self._cx.clear()
        self._last = None
        self._approach_active = False

    def update(self, det):
        if not det.found:
            return
        if det.center_x is not None:
            cx = float(det.center_x)
        elif det.centers is not None and len(det.centers) > 0:
            cx = float(det.centers[0, 0, 0])
        else:
            return
        self._cx.append(cx)
        aspect = grid_bbox_aspect(det)
        source = getattr(det, "source", None) or ("grid" if det.heading is not None else "blue")
        if source == "grid" and det.heading is not None:
            heading = float(det.heading)
            self._last = _LastLeaderSample(cx, heading, aspect, "grid")
        else:
            self._last = _LastLeaderSample(cx, None, aspect, "blue")

    def begin_approach_if_needed(self, red_prox, cfg):
        """Clear stale samples when red first shows in the far band."""
        min_far = int(cfg.get("intersection_red_approach_min_far_px", 600))
        approaching = red_prox.far_px >= min_far
        if approaching and not self._approach_active:
            self._cx.clear()
            self._last = None
        self._approach_active = approaching

    @property
    def approach_active(self):
        return self._approach_active

    @property
    def has_last_grid(self):
        return self._last is not None

    def _heading_turn(self, heading, cfg):
        if heading is None:
            return TURN_STRAIGHT
        thresh = float(cfg.get("intersection_heading_thresh", 0.12))
        sign = float(cfg.get("intersection_heading_sign", -1.0))
        h = sign * float(heading)
        if h >= thresh:
            return TURN_RIGHT
        if h <= -thresh:
            return TURN_LEFT
        return TURN_STRAIGHT

    def _cx_trend_turn(self, cfg):
        if len(self._cx) < 3:
            return TURN_STRAIGHT
        lookback = max(2, int(cfg.get("intersection_turn_infer_lookback", 8)))
        samples = list(self._cx)[-min(lookback, len(self._cx)) :]
        drift = float(samples[-1] - samples[0])
        thresh = float(cfg.get("intersection_cx_drift_px", 30.0))
        if drift >= thresh:
            return TURN_RIGHT
        if drift <= -thresh:
            return TURN_LEFT
        return TURN_STRAIGHT

    def _infer_from_center(self, last, cfg, frame_w):
        cx_offset_thresh = float(cfg.get("intersection_last_cx_offset_px", 28.0))
        straight_center = float(cfg.get("intersection_blue_straight_center_px", 18.0))
        offset = float(last.center_x) - (frame_w / 2.0)
        if abs(offset) <= straight_center:
            trend_vote = self._cx_trend_turn(cfg)
            if trend_vote == TURN_STRAIGHT:
                return TURN_STRAIGHT
        if offset >= cx_offset_thresh:
            return TURN_RIGHT
        if offset <= -cx_offset_thresh:
            return TURN_LEFT
        trend_vote = self._cx_trend_turn(cfg)
        if trend_vote != TURN_STRAIGHT:
            return trend_vote
        return TURN_STRAIGHT

    def infer(self, cfg, frame_w=640.0):
        """Use the last leader sighting to pick straight / left / right."""
        last = self._last
        if last is None:
            return TURN_STRAIGHT

        frame_w = max(1.0, float(frame_w))
        straight_aspect = float(cfg.get("intersection_straight_aspect_min", 2.2))
        heading_thresh = float(cfg.get("intersection_heading_thresh", 0.12))

        if last.source == "grid" and last.heading is not None:
            # Face-on grid + weak heading on the last frame -> straight through.
            if last.aspect is not None and last.aspect >= straight_aspect:
                if abs(float(last.heading)) < heading_thresh * 0.45:
                    return TURN_STRAIGHT

            heading_vote = self._heading_turn(last.heading, cfg)
            if heading_vote != TURN_STRAIGHT:
                return heading_vote

        return self._infer_from_center(last, cfg, frame_w)

    def debug_votes(self, cfg, frame_w=640.0):
        last = self._last
        frame_w = max(1.0, float(frame_w))
        if last is None:
            return {
                "has_last_grid": False,
                "last_source": None,
                "last_cx": None,
                "last_heading": None,
                "last_aspect": None,
                "cx_offset_px": 0.0,
                "heading_vote": TURN_STRAIGHT,
                "cx_trend_vote": TURN_STRAIGHT,
                "infer": TURN_STRAIGHT,
            }
        cx_offset = float(last.center_x) - (frame_w / 2.0)
        heading_vote = (
            self._heading_turn(last.heading, cfg)
            if last.source == "grid" and last.heading is not None
            else TURN_STRAIGHT
        )
        trend_vote = self._cx_trend_turn(cfg)
        return {
            "has_last_grid": True,
            "last_source": last.source,
            "last_cx": round(float(last.center_x), 1),
            "last_heading": (
                round(float(last.heading), 3) if last.heading is not None else None
            ),
            "last_aspect": (
                round(float(last.aspect), 2) if last.aspect is not None else None
            ),
            "cx_offset_px": round(cx_offset, 1),
            "heading_vote": heading_vote,
            "cx_trend_vote": trend_vote,
            "infer": self.infer(cfg, frame_w=frame_w),
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


def intersection_turn_schedule(direction, cfg):
    """Return preamble (straight), arc (~90 deg), and optional tail seconds."""
    if direction == TURN_STRAIGHT:
        return {
            "preamble_s": 0.0,
            "arc_s": max(0.2, float(cfg.get("intersection_turn_straight_s", 1.4))),
            "tail_s": 0.0,
        }
    if direction == TURN_RIGHT:
        return {
            "preamble_s": max(0.0, float(cfg.get("intersection_right_preamble_s", 0.35))),
            "arc_s": max(0.2, float(cfg.get("intersection_turn_right_s", 2.0))),
            "tail_s": max(0.0, float(cfg.get("intersection_turn_tail_straight_s", 0.0))),
        }
    if direction == TURN_LEFT:
        return {
            "preamble_s": max(0.0, float(cfg.get("intersection_left_preamble_s", 0.55))),
            "arc_s": max(0.2, float(cfg.get("intersection_turn_left_s", 2.1))),
            "tail_s": max(0.0, float(cfg.get("intersection_turn_tail_straight_s", 0.0))),
        }
    return {
        "preamble_s": 0.0,
        "arc_s": max(0.2, float(cfg.get("intersection_turn_straight_s", 1.4))),
        "tail_s": 0.0,
    }


def intersection_wheel_commands(direction, cfg, speed=None):
    """Fixed differential PWM for intersection arc at the given cruise speed."""
    if speed is None:
        turn_speed = float(cfg.get("intersection_turn_speed", 0.15))
    else:
        turn_speed = float(speed)
    turn_speed = min(1.0, max(0.05, turn_speed))
    inner_ratio = float(cfg.get("intersection_turn_inner_ratio", 0.27))
    outer_ratio = float(cfg.get("intersection_turn_outer_ratio", 1.0))
    inner = min(1.0, turn_speed * inner_ratio)
    outer = min(1.0, turn_speed * outer_ratio)
    if direction == TURN_LEFT:
        return inner, outer
    if direction == TURN_RIGHT:
        return outer, inner
    return turn_speed, turn_speed
