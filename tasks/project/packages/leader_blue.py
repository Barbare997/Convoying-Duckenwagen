"""Blue duckie body fallback when the rear dot grid is too small to track."""

import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from tasks.project.packages.leader_grid import GridDetection, _pattern_size


_cache_lock = threading.Lock()
_cache_ts = 0.0
_cache_result = None  # type: Optional[GridDetection]


def reset_leader_blue_cache() -> None:
    global _cache_result, _cache_ts
    with _cache_lock:
        _cache_result = None
        _cache_ts = 0.0


def _height_to_distance_signal(height_px: float, cfg: Dict[str, Any]) -> float:
    safe = float(cfg.get("leader_blue_safe_px", 14.0))
    stop = float(cfg.get("leader_blue_stop_px", 50.0))
    if stop <= safe:
        stop = safe + 1.0
    return float(np.clip((height_px - safe) / (stop - safe), 0.0, 1.0))


def _leader_blue_mask(frame_bgr: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h_min = int(cfg.get("leader_blue_h_min", 100))
    h_max = int(cfg.get("leader_blue_h_max", 125))
    s_min = int(cfg.get("leader_blue_s_min", 90))
    v_min = int(cfg.get("leader_blue_v_min", 75))
    mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, 255, 255))

    top = int(h * float(cfg.get("leader_blue_roi_top_frac", 0.22)))
    bot = int(h * float(cfg.get("leader_blue_roi_bottom_frac", 0.82)))
    top = min(max(top, 0), h - 2)
    bot = min(max(bot, top + 1), h)
    mask[:top, :] = 0
    mask[bot:, :] = 0

    k = max(3, int(cfg.get("leader_blue_morph_kernel", 5)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def leader_blue_debug_mask(frame_bgr, cfg: Dict[str, Any]):
    """Same HSV mask as detection — for dashboard tuning only."""
    if frame_bgr is None:
        return None
    return _leader_blue_mask(frame_bgr, cfg)


def leader_spacing_span_px(det, cfg):
    """Map blue body span into the grid spacing scale (span_target_px)."""
    if det is None or det.span_px is None:
        return None
    span = float(det.span_px)
    if getattr(det, "source", None) != "blue":
        return span
    grid_target = float(cfg.get("span_target_px", 7.0))
    blue_target = float(cfg.get("leader_blue_span_target_px", 36.0))
    return span * (grid_target / max(blue_target, 1.0))


def detect_leader_blue(frame_bgr, cfg: Dict[str, Any]) -> GridDetection:
    pattern = _pattern_size(cfg)
    empty = GridDetection(False, None, 0.0, None, pattern, source="blue")
    if frame_bgr is None or not bool(cfg.get("leader_blue_enabled", True)):
        return empty

    mask = _leader_blue_mask(frame_bgr, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return empty

    min_area = float(cfg.get("leader_blue_min_area", 600.0))
    max_area = float(cfg.get("leader_blue_max_area", 90000.0))
    best = None
    best_area = 0.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        if area > best_area:
            best_area = area
            best = cnt
    if best is None:
        return empty

    x, y, bw, bh = cv2.boundingRect(best)
    if bw < 8 or bh < 8:
        return empty

    cx = float(x) + float(bw) / 2.0
    cy = float(y) + float(bh) / 2.0
    span_scale = float(cfg.get("leader_blue_span_scale", 0.55))
    span_px = max(1.0, float(bh) * span_scale)
    signal = _height_to_distance_signal(float(bh), cfg)
    centers = np.array([[[cx, cy]]], dtype=np.float32)
    quality = min(1.0, best_area / max(min_area * 4.0, 1.0))
    return GridDetection(
        True,
        signal,
        quality,
        centers,
        pattern,
        span_px=span_px,
        bbox=(int(x), int(y), int(x + bw), int(y + bh)),
        heading=None,
        center_x=cx,
        source="blue",
    )


def fetch_leader_blue(
    frame_bgr,
    cfg: Dict[str, Any],
    *,
    force: bool = False,
) -> GridDetection:
    global _cache_ts, _cache_result
    if str(cfg.get("role", "leader")).lower() != "follower":
        return GridDetection(False, None, 0.0, None, _pattern_size(cfg), source="blue")

    min_dt = 1.0 / max(1.0, float(cfg.get("grid_detect_hz", 10)))
    now = time.time()
    with _cache_lock:
        if not force and _cache_result is not None and now - _cache_ts < min_dt:
            return _cache_result

    result = detect_leader_blue(frame_bgr, cfg)
    with _cache_lock:
        _cache_result = result
        _cache_ts = time.time()
    return result


def draw_blue_overlay(frame_bgr: np.ndarray, detection: GridDetection) -> np.ndarray:
    out = frame_bgr.copy()
    if not detection.found:
        return out

    bbox = detection.bbox_or_centers()
    if bbox is None:
        return out
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 0), 2)
    if detection.centers is not None:
        cx = int(detection.centers[0, 0, 0])
        cy = int(detection.centers[0, 0, 1])
        cv2.circle(out, (cx, cy), 5, (255, 80, 0), -1)
    parts = ["blue"]
    if detection.distance_signal is not None:
        parts.append(f"dist={detection.distance_signal:.2f}")
    if detection.span_px is not None:
        parts.append(f"span={detection.span_px:.0f}px")
    cv2.putText(
        out,
        " ".join(parts),
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 180, 0),
        1,
        cv2.LINE_AA,
    )
    return out
