import os
import yaml
import numpy as np
import cv2
from collections import deque
from typing import Tuple

from tasks.visual_lane_servoing.packages import visual_servoing_activity as student
from tasks.visual_lane_servoing.packages.cuvrve_behavior import detect_curve

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_config.yaml'
))

_LINE_OFFSET = 160
_ROI_START   = 0.45
_NUM_SLICES  = 3
_SLICE_TOL   = 5
# Small nudge toward lane center (keep symmetric — large white bias caused false right steer).
_YELLOW_CENTER_BIAS_PX = 8
_YELLOW_FAR_SLICE_BIAS_PX = 10
_WHITE_CENTER_BIAS_PX = 3


def _largest_component_cols(strip: np.ndarray) -> np.ndarray:
    """Ignore small noise blobs; use the main lane marking in this row."""
    idx = np.where(strip > 0)[1]
    if len(idx) == 0:
        return idx
    if len(idx) < 25:
        return idx
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(strip, connectivity=8)
    if num_labels <= 1:
        return idx
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == best)[1]


def detect_lines_in_slices(
    mask_yellow: np.ndarray,
    mask_white:  np.ndarray,
    h: int,
    mask_red: np.ndarray = None,
) -> Tuple[list, list, list]:
    w = mask_yellow.shape[1]
    slice_height = int(h * 0.35 / _NUM_SLICES)
    start_y      = int(h * _ROI_START)
    yellow_xs, white_xs, red_xs = [], [], []

    for i in range(_NUM_SLICES):
        y = start_y + i * slice_height + slice_height // 2
        far_frac = (_NUM_SLICES - 1 - i) / max(1, _NUM_SLICES - 1)

        strip_y = mask_yellow[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx = _largest_component_cols(strip_y)
        if len(idx) > 0:
            y_edge = int(np.percentile(idx, 90))
            y_inset = int(_YELLOW_CENTER_BIAS_PX + far_frac * _YELLOW_FAR_SLICE_BIAS_PX)
            yellow_xs.append(min(w - 1, y_edge + y_inset))

        strip_w = mask_white[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx = _largest_component_cols(strip_w)
        if len(idx) > 0:
            white_xs.append(max(0, int(np.min(idx)) - _WHITE_CENTER_BIAS_PX))

        if mask_red is not None:
            strip_r = mask_red[y - _SLICE_TOL: y + _SLICE_TOL, :]
            idx = _largest_component_cols(strip_r)
            if len(idx) > 0:
                r_edge = int(np.percentile(idx, 90))
                r_inset = int(_YELLOW_CENTER_BIAS_PX + far_frac * _YELLOW_FAR_SLICE_BIAS_PX)
                red_xs.append(min(w - 1, r_edge + r_inset))

    return yellow_xs, white_xs, red_xs


class LaneServoingAgent:

    def __init__(self, config_path: str = None):
        path = config_path or _CONFIG_FILE
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        self.p_gain              = cfg.get('p_gain',              0.1)
        self.d_gain              = cfg.get('d_gain',              0.35)
        self.max_steer           = cfg.get('max_steer',           0.4)
        self.base_speed          = cfg.get('base_speed',          0.4)
        self.curve_speed         = cfg.get('curve_speed',         0.26)
        self.curve_threshold     = cfg.get('curve_threshold',     350)
        self.steering_threshold  = cfg.get('steering_threshold',  0.2)
        self.curve_boost         = cfg.get('curve_boost',         1.3)
        self.detection_threshold = cfg.get('detection_threshold', 500)

        self.frame_count        = 0
        self._prev_error        = 0.0
        self._filtered_error    = 0.0
        self._lane_half_width   = float(_LINE_OFFSET)
        self._left_history      = deque(maxlen=3)
        self._right_history     = deque(maxlen=3)
        self.last_debug_info    = self._empty_debug_info(480, 640)

    def _weighted_line_x(self, xs: list) -> float:
        """Weight lower slices (closer to robot) more than far slices."""
        if not xs:
            return 0.0
        if len(xs) == 1:
            return float(xs[0])
        weights = np.linspace(0.2, 1.0, num=len(xs), dtype=np.float64)
        weights /= weights.sum()
        return float(np.dot(weights, xs))

    def _calculate_error(self, yellow_xs, white_xs, left_det, right_det, w):
        if left_det and right_det and yellow_xs and white_xs:
            y_pos = self._weighted_line_x(yellow_xs)
            w_pos = self._weighted_line_x(white_xs)
            measured = (w_pos - y_pos) / 2.0
            if measured > 20:
                self._lane_half_width = 0.9 * self._lane_half_width + 0.1 * measured
            error = w / 2.0 - (y_pos + w_pos) / 2.0
        elif left_det and yellow_xs:
            y_pos = self._weighted_line_x(yellow_xs)
            error = w / 2.0 - (y_pos + self._lane_half_width)
        elif right_det and white_xs:
            w_pos = self._weighted_line_x(white_xs)
            error = w / 2.0 - (w_pos - self._lane_half_width)
        else:
            error = self._prev_error

        return float(np.clip(error / (w / 2.0), -1.0, 1.0))

    def _calculate_steering(self, error: float) -> float:
        error_diff       = error - self._prev_error
        self._prev_error = error
        steering = self.p_gain * error + self.d_gain * error_diff
        return float(np.clip(steering, -self.max_steer, self.max_steer))

    def _motor_commands(self, steering: float, recovery: bool, is_curve: bool, both_visible: bool):
        if recovery:
            return 0.0, 0.0

        speed = self.curve_speed if is_curve else self.base_speed
        
        if not both_visible:
            speed *= 0.8

        left  = speed - steering
        right = speed + steering

        if is_curve and abs(steering) > self.steering_threshold:
            if steering > 0:
                right *= 5
            else:
                left  *= self.curve_boost

        return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))

    def _smooth(self, left, right, both_visible):
        buf = 2 if both_visible else 1
        if self._left_history.maxlen != buf:
            self._left_history  = deque(maxlen=buf)
            self._right_history = deque(maxlen=buf)
        self._left_history.append(left)
        self._right_history.append(right)
        return (sum(self._left_history)  / len(self._left_history),
                sum(self._right_history) / len(self._right_history))

    def compute_commands(self, image: np.ndarray) -> Tuple[float, float]:
        self.frame_count += 1
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            mask_y, mask_w, mask_r = student.detect_lane_markings(bgr)
        except Exception as e:
            print(f"[Agent] detect_lane_markings error: {e}")
            return 0.0, 0.0

        mask_left = np.clip(mask_y + mask_r, 0, 1)
        mask_y_vis = (mask_y * 255).astype(np.uint8)
        mask_r_vis = (mask_r * 255).astype(np.uint8)
        mask_w = (mask_w * 255).astype(np.uint8)

        yellow_pixels = int(np.count_nonzero(mask_y_vis))
        red_pixels = int(np.count_nonzero(mask_r_vis))
        white_pixels  = int(np.count_nonzero(mask_w))
        total_pixels  = yellow_pixels + red_pixels + white_pixels

        combined = np.clip(mask_left + (mask_w // 255), 0, 1)
        self.last_debug_info = {
            'roi':               image,
            'frame_bgr':         bgr,
            'lane_mask':         (combined * 255).astype(np.uint8),
            'white_mask':        mask_w,
            'yellow_mask':       mask_y_vis,
            'red_mask':          mask_r_vis,
            'total_lane_pixels': total_pixels,
            'lateral_error':     float(np.clip(self._prev_error, -1.0, 1.0)),
            'lane_detected':     total_pixels >= self.detection_threshold,
            'frame_count':       self.frame_count,
        }

        h, w      = mask_y_vis.shape
        left_det  = (yellow_pixels + red_pixels) > 0
        right_det = white_pixels  > 0
        recovery  = total_pixels  < self.detection_threshold

        mask_left_u8 = (mask_left * 255).astype(np.uint8)
        yellow_xs, white_xs, red_xs = detect_lines_in_slices(mask_left_u8, mask_w, h, mask_r_vis)
        both_visible        = left_det and right_det and not recovery
        is_curve, curve_dir = detect_curve(yellow_xs, white_xs, self.curve_threshold, red_xs)

        raw_error            = self._calculate_error(yellow_xs, white_xs, left_det, right_det, w)
        self._filtered_error = 0.7 * self._filtered_error + 0.3 * raw_error
        steering             = self._calculate_steering(self._filtered_error)
        left, right          = self._motor_commands(steering, recovery, is_curve, both_visible)
        left, right          = self._smooth(left, right, both_visible)

        slice_height = int(h * 0.35 / _NUM_SLICES)
        start_y      = int(h * _ROI_START)
        self.last_debug_info.update({
            'yellow_xs': yellow_xs,
            'white_xs':  white_xs,
            'red_xs':    red_xs,
            'slice_ys':  [start_y + i * slice_height + slice_height // 2 for i in range(_NUM_SLICES)],
            'is_curve':  is_curve,
            'curve_dir': curve_dir,
            'lateral_error': float(np.clip(self._filtered_error, -1.0, 1.0)),
        })

        return left, right

    def step(self, image: np.ndarray, wheels_driver) -> Tuple[float, float]:
        left, right = self.compute_commands(image)
        wheels_driver.set_wheels_speed(left, right)
        return left, right

    def get_debug_info(self, image: np.ndarray) -> dict:
        return self.last_debug_info

    def _empty_debug_info(self, h, w):
        return {
            'roi':               np.zeros((h, w, 3), dtype=np.uint8),
            'lane_mask':         np.zeros((h, w),    dtype=np.uint8),
            'white_mask':        np.zeros((h, w),    dtype=np.uint8),
            'yellow_mask':       np.zeros((h, w),    dtype=np.uint8),
            'red_mask':          np.zeros((h, w),    dtype=np.uint8),
            'total_lane_pixels': 0,
            'lateral_error':     0.0,
            'lane_detected':     False,
            'frame_count':       0,
        }