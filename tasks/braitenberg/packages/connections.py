from typing import Tuple
import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """Left motor weight matrix: highest at bottom-left, decreasing toward top-right."""
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError("shape must be (height, width) with positive integers")

    # y: -1 at top, +1 at bottom
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    # x: +1 at left, -1 at right
    x = np.linspace(1.0, -1.0, w, dtype=np.float32)[None, :]

    # Bottom-left = +1, top-right = -1 (simple diagonal gradient)
    return (x + y) / 2.0


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """Right motor weight matrix: highest at bottom-right, decreasing toward top-left."""
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError("shape must be (height, width) with positive integers")

    # y: -1 at top, +1 at bottom
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    # x: -1 at left, +1 at right
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]

    # Bottom-right = +1, top-left = -1 (simple diagonal gradient)
    return (x + y) / 2.0
