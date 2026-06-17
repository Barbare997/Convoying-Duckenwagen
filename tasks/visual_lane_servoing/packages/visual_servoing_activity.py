from typing import Tuple
import os
import numpy as np
import cv2
import yaml

_HSV_FILE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_hsv_config.yaml')
try:
    with open(_HSV_FILE) as _f:
        _h = yaml.safe_load(_f) or {}
except FileNotFoundError:
    _h = {}

_yellow_lower = np.array([_h.get('yellow_lower_h', 0),  _h.get('yellow_lower_s', 0),  _h.get('yellow_lower_v', 0)])
_yellow_upper = np.array([_h.get('yellow_upper_h', 0),  _h.get('yellow_upper_s', 0), _h.get('yellow_upper_v', 0)])

_white_lower = np.array([_h.get('white_lower_h', 0),   _h.get('white_lower_s', 0), _h.get('white_lower_v', 0)])
_white_upper = np.array([_h.get('white_upper_h', 0), _h.get('white_upper_s', 0), _h.get('white_upper_v', 0)])

_red_lower = np.array([_h.get('red_lower_h', 0), _h.get('red_lower_s', 100), _h.get('red_lower_v', 100)])
_red_upper = np.array([_h.get('red_upper_h', 10), _h.get('red_upper_s', 255), _h.get('red_upper_v', 255)])


def _red_hsv_mask(hsv: np.ndarray) -> np.ndarray:
    """Red spans low and high hue in OpenCV HSV — OR both bands with shared S/V sliders."""
    mask = cv2.inRange(hsv, _red_lower, _red_upper)
    if int(_red_lower[0]) <= 20:
        wrap_lo = np.array([max(0, 179 - (10 - int(_red_lower[0]))), int(_red_lower[1]), int(_red_lower[2])])
        wrap_hi = np.array([179, int(_red_upper[1]), int(_red_upper[2])])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, wrap_lo, wrap_hi))
    return mask


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 1) Ignore sky / far field; keep road-focused band (lower ~72%).
    h, w = image.shape[:2]
    crop_top = int(h * 0.28)
    roi = image[crop_top:, :]

    # 2) Blur to reduce high-frequency noise before color thresholding.
    blurred = cv2.GaussianBlur(roi, (5, 5), 2)

    # 3) HSV makes color-based masking (yellow/white/red lanes) more stable.
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    yellow_mask_cfg = cv2.inRange(hsv, _yellow_lower, _yellow_upper)
    white_mask_cfg = cv2.inRange(hsv, _white_lower, _white_upper)
    red_mask_cfg = _red_hsv_mask(hsv)

    yellow_mask = yellow_mask_cfg
    white_mask = white_mask_cfg
    red_mask = red_mask_cfg

    # Geometric prior with overlap (less brittle in curves).
    left_limit = int(0.70 * w)
    right_limit = int(0.30 * w)
    yellow_mask[:, left_limit:] = 0
    white_mask[:, :right_limit] = 0
    red_mask[:, left_limit:] = 0

    kernel = np.ones((3, 3), dtype=np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    thin_kernel = np.ones((2, 2), dtype=np.uint8)
    yellow_mask = cv2.erode(yellow_mask, thin_kernel, iterations=1)
    white_mask = cv2.erode(white_mask, thin_kernel, iterations=1)
    red_mask = cv2.erode(red_mask, thin_kernel, iterations=1)

    full_yellow = np.zeros((h, w), dtype=np.uint8)
    full_white = np.zeros((h, w), dtype=np.uint8)
    full_red = np.zeros((h, w), dtype=np.uint8)
    full_yellow[crop_top:, :] = yellow_mask
    full_white[crop_top:, :] = white_mask
    full_red[crop_top:, :] = red_mask

    return (
        (full_yellow // 255).astype(np.uint8),
        (full_white // 255).astype(np.uint8),
        (full_red // 255).astype(np.uint8),
    )


def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper, red_lower=None, red_upper=None):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper, _red_lower, _red_upper
    _yellow_lower = np.array(yellow_lower)
    _yellow_upper = np.array(yellow_upper)
    _white_lower = np.array(white_lower)
    _white_upper = np.array(white_upper)
    if red_lower is not None:
        _red_lower = np.array(red_lower)
    if red_upper is not None:
        _red_upper = np.array(red_upper)


def get_hsv_bounds():
    return {
        'yellow_lower_h': int(_yellow_lower[0]),    'yellow_upper_h': int(_yellow_upper[0]),
        'yellow_lower_s': int(_yellow_lower[1]),    'yellow_upper_s': int(_yellow_upper[1]),
        'yellow_lower_v': int(_yellow_lower[2]),    'yellow_upper_v': int(_yellow_upper[2]),
        'white_lower_h':  int(_white_lower[0]), 'white_upper_h':  int(_white_upper[0]),
        'white_lower_s':  int(_white_lower[1]), 'white_upper_s':  int(_white_upper[1]),
        'white_lower_v':  int(_white_lower[2]), 'white_upper_v':  int(_white_upper[2]),
        'red_lower_h': int(_red_lower[0]), 'red_upper_h': int(_red_upper[0]),
        'red_lower_s': int(_red_lower[1]), 'red_upper_s': int(_red_upper[1]),
        'red_lower_v': int(_red_lower[2]), 'red_upper_v': int(_red_upper[2]),
    }
