"""Map stream clicks on the lane visualization to camera pixels and HSV samples."""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

VIZ_PANEL_W = 320
VIZ_GRID_ROWS = 3
INFO_STRIP_H = 120

# Tolerances for suggested HSV bounds around a clicked pixel.
_YELLOW_TOL = (10, 60, 60)  # h, s, v
_WHITE_TOL = (20, 50, 50)
_RED_TOL = (10, 60, 60)


def _display_h(frame_h: int, frame_w: int) -> int:
    return int(frame_h * VIZ_PANEL_W / max(1, frame_w))


def camera_pixel_from_stream_click(
    stream_x: float,
    stream_y: float,
    frame_h: int,
    frame_w: int,
) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
    """Top-left panel of the composite stream is the live camera view."""
    display_h = _display_h(frame_h, frame_w)
    sx = int(stream_x)
    sy = int(stream_y)

    if sx < 0 or sy < 0 or sx >= VIZ_PANEL_W or sy >= display_h:
        return None, (
            "Click the Camera panel (top-left of the stream). "
            "Other panels show masks, not raw colors."
        )

    fx = int(np.clip(sx * frame_w / VIZ_PANEL_W, 0, frame_w - 1))
    fy = int(np.clip(sy * frame_h / display_h, 0, frame_h - 1))
    return (fx, fy), None


def _clamp_hsv(lower: np.ndarray, upper: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lo = lower.astype(np.int32).copy()
    hi = upper.astype(np.int32).copy()
    lo[0] = int(np.clip(lo[0], 0, 179))
    hi[0] = int(np.clip(hi[0], 0, 179))
    lo[1:] = np.clip(lo[1:], 0, 255)
    hi[1:] = np.clip(hi[1:], 0, 255)
    if lo[0] > hi[0]:
        lo[0], hi[0] = hi[0], lo[0]
    for i in (1, 2):
        if lo[i] > hi[i]:
            lo[i], hi[i] = hi[i], lo[i]
    return lo.astype(np.int32), hi.astype(np.int32)


def _suggest_bounds(h: int, s: int, v: int, tol: Tuple[int, int, int]) -> Dict[str, int]:
    th, ts, tv = tol
    lo, hi = _clamp_hsv(
        np.array([h - th, s - ts, v - tv], dtype=np.int32),
        np.array([h + th, s + ts, v + tv], dtype=np.int32),
    )
    return {
        "lower_h": int(lo[0]), "lower_s": int(lo[1]), "lower_v": int(lo[2]),
        "upper_h": int(hi[0]), "upper_s": int(hi[1]), "upper_v": int(hi[2]),
    }


def _guess_line(fx: int, frame_w: int) -> str:
    if fx < int(0.40 * frame_w):
        return "yellow"
    if fx > int(0.60 * frame_w):
        return "white"
    return "red"


def sample_pixel_from_frame_bgr(
    frame_bgr: np.ndarray,
    stream_x: float,
    stream_y: float,
) -> Dict[str, Any]:
    if frame_bgr is None or frame_bgr.size == 0:
        return {"ok": False, "error": "No camera frame available yet."}

    frame_h, frame_w = frame_bgr.shape[:2]
    pixel, err = camera_pixel_from_stream_click(stream_x, stream_y, frame_h, frame_w)
    if err:
        return {"ok": False, "error": err}

    fx, fy = pixel
    b, g, r = [int(x) for x in frame_bgr[fy, fx]]
    hsv = cv2.cvtColor(
        np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV
    )[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    line_guess = _guess_line(fx, frame_w)

    return {
        "ok": True,
        "pixel": {"x": fx, "y": fy},
        "stream_click": {"x": float(stream_x), "y": float(stream_y)},
        "bgr": {"b": b, "g": g, "r": r},
        "rgb": {"r": r, "g": g, "b": b},
        "hsv": {"h": h, "s": s, "v": v},
        "line_guess": line_guess,
        "suggested_yellow": _suggest_bounds(h, s, v, _YELLOW_TOL),
        "suggested_white": _suggest_bounds(h, s, v, _WHITE_TOL),
        "suggested_red": _suggest_bounds(h, s, v, _RED_TOL),
        "frame_size": {"width": frame_w, "height": frame_h},
        "camera_panel": {"width": VIZ_PANEL_W, "height": _display_h(frame_h, frame_w)},
    }
