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

def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 1) Ignore the horizon: lane pixels are expected in the lower image area.
    h, w = image.shape[:2]
    crop_top = int(h * 0.3)
    roi = image[crop_top:, :]

    # 2) Blur to reduce high-frequency noise before color thresholding.
    blurred = cv2.GaussianBlur(roi, (5, 5), 2)

    # 3) HSV makes color-based masking (yellow/white lanes) more stable.
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    yellow_mask_cfg = cv2.inRange(hsv, _yellow_lower, _yellow_upper)
    white_mask_cfg = cv2.inRange(hsv, _white_lower, _white_upper)

    # Robust priors keep lane extraction stable even with broad HSV sliders.
    yellow_mask = yellow_mask_cfg
    white_mask =  white_mask_cfg

    # Geometric prior with overlap (less brittle in curves).
    # Yellow mostly left; white mostly right, but keep central overlap.
    left_limit = int(0.70 * w)
    right_limit = int(0.30 * w)
    yellow_mask[:, left_limit:] = 0
    white_mask[:, :right_limit] = 0

    kernel = np.ones((3, 3), dtype=np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    # Slightly thin masks so centroids stay on marking cores.
    thin_kernel = np.ones((2, 2), dtype=np.uint8)
    yellow_mask = cv2.erode(yellow_mask, thin_kernel, iterations=1)
    white_mask = cv2.erode(white_mask, thin_kernel, iterations=1)

    # Pull both detections slightly toward lane center:
    # - yellow shifted right, white shifted left.
    # This reduces boundary hugging when one side dominates in curves.
    # yellow_center_bias_px = 6
    # white_center_bias_px = 16

    # if yellow_center_bias_px > 0:
    #     shifted_yellow = np.zeros_like(yellow_mask)
    #     shifted_yellow[:, yellow_center_bias_px:] = yellow_mask[:, :-yellow_center_bias_px]
    #     yellow_mask = shifted_yellow

    # if white_center_bias_px > 0:
    #     shifted_white = np.zeros_like(white_mask)
    #     shifted_white[:, :-white_center_bias_px] = white_mask[:, white_center_bias_px:]
    #     white_mask = shifted_white


    # 4) Rebuild full-size masks so downstream code keeps original coordinates.
    full_yellow = np.zeros((h, w), dtype=np.uint8)
    full_white = np.zeros((h, w), dtype=np.uint8)
    full_yellow[crop_top:, :] = yellow_mask
    full_white[crop_top:, :] = white_mask

    # Convert OpenCV masks (0/255) to binary masks (0/1).
    return (full_yellow // 255).astype(np.uint8), (full_white // 255).astype(np.uint8)




def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    _yellow_lower    = np.array(yellow_lower)
    _yellow_upper    = np.array(yellow_upper)
    _white_lower = np.array(white_lower)
    _white_upper = np.array(white_upper)

def get_hsv_bounds():
    return {
        'yellow_lower_h': int(_yellow_lower[0]),    'yellow_upper_h': int(_yellow_upper[0]),
        'yellow_lower_s': int(_yellow_lower[1]),    'yellow_upper_s': int(_yellow_upper[1]),
        'yellow_lower_v': int(_yellow_lower[2]),    'yellow_upper_v': int(_yellow_upper[2]),
        'white_lower_h':  int(_white_lower[0]), 'white_upper_h':  int(_white_upper[0]),
        'white_lower_s':  int(_white_lower[1]), 'white_upper_s':  int(_white_upper[1]),
        'white_lower_v':  int(_white_lower[2]), 'white_upper_v':  int(_white_upper[2]),
    }