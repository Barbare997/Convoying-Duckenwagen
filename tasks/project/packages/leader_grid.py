"""Circle-grid leader tracking — wraps MarkerGridTracker (NoMrBody/duckietown-convoy)."""

import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from tasks.project.packages.marker_grid import MarkerGridTracker


class GridDetection(object):
    """Leader grid detection result (plain class for Python 3.6 bot compatibility)."""

    __slots__ = (
        "found", "distance_signal", "quality", "centers", "pattern_size",
        "span_px", "bbox", "heading", "center_x",
    )

    def __init__(
        self,
        found,
        distance_signal,
        quality,
        centers,
        pattern_size,
        span_px=None,
        bbox=None,
        heading=None,
        center_x=None,
    ):
        self.found = found
        self.distance_signal = distance_signal
        self.quality = quality
        self.centers = centers
        self.pattern_size = pattern_size
        self.span_px = span_px
        self.bbox = bbox
        self.heading = heading
        self.center_x = center_x

    def bbox_or_centers(self):
        if self.bbox is not None:
            return self.bbox
        if self.centers is None or len(self.centers) == 0:
            return None
        xs = self.centers[:, 0, 0]
        ys = self.centers[:, 0, 1]
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


_cache_lock = threading.Lock()
_cache_ts = 0.0
_cache_result = None  # type: Optional[GridDetection]
_tracker = None  # type: Optional[MarkerGridTracker]
_tracker_cfg_key = None  # type: Optional[Tuple]


def _pattern_size(cfg):
    cols = max(2, int(cfg.get("grid_cols", 7)))
    rows = max(2, int(cfg.get("grid_rows", 3)))
    return cols, rows


def _span_to_distance_signal(span_px: float, cfg: Dict[str, Any]) -> float:
    """Map mean dot spacing (px) to 0..1 — higher = leader closer."""
    safe = float(cfg.get("grid_safe_px", 20.0))
    stop = float(cfg.get("grid_stop_px", 42.0))
    if stop <= safe:
        stop = safe + 1.0
    return float(np.clip((span_px - safe) / (stop - safe), 0.0, 1.0))


def _tracker_key(cfg: Dict[str, Any]) -> Tuple:
    keys = (
        "grid_cols", "grid_rows", "grid_downscale", "grid_roi_pad_px",
        "grid_roi_downscale", "grid_far_search", "grid_far_band_top_frac",
        "grid_far_band_bot_frac", "grid_lost_grace_frames", "grid_use_clustering",
        "grid_blob_min_area", "grid_blob_max_area", "grid_blob_min_circularity",
    )
    return tuple(cfg.get(k) for k in keys)


def _get_tracker(cfg: Dict[str, Any]) -> MarkerGridTracker:
    global _tracker, _tracker_cfg_key
    key = _tracker_key(cfg)
    if _tracker is None or _tracker_cfg_key != key:
        _tracker = MarkerGridTracker(cfg=cfg)
        _tracker_cfg_key = key
    return _tracker


def reset_grid_tracker() -> None:
    """Clear tracker state after sim reset."""
    global _tracker, _tracker_cfg_key, _cache_result, _cache_ts
    with _cache_lock:
        _tracker = None
        _tracker_cfg_key = None
        _cache_result = None
        _cache_ts = 0.0


def detect_leader_grid(frame_bgr, cfg: Dict[str, Any]) -> GridDetection:
    pattern = _pattern_size(cfg)
    empty = GridDetection(False, None, 0.0, None, pattern)
    if frame_bgr is None:
        return empty

    obs = _get_tracker(cfg).update(frame_bgr)
    if obs is None:
        return empty

    x1, y1, x2, y2 = obs.bbox
    cx, cy = obs.midpoint
    centers = np.array([[[cx, cy]]], dtype=np.float32)
    signal = _span_to_distance_signal(obs.span_px, cfg)
    return GridDetection(
        True,
        signal,
        float(obs.score),
        centers,
        pattern,
        span_px=float(obs.span_px),
        bbox=(int(x1), int(y1), int(x2), int(y2)),
        heading=obs.heading,
        center_x=float(cx),
    )


def fetch_leader_grid(
    frame_bgr,
    cfg: Dict[str, Any],
    *,
    force: bool = False,
) -> GridDetection:
    global _cache_ts, _cache_result
    if str(cfg.get("role", "leader")).lower() != "follower":
        return GridDetection(False, None, 0.0, None, _pattern_size(cfg))

    min_dt = 1.0 / max(1.0, float(cfg.get("grid_detect_hz", 10)))
    now = time.time()
    with _cache_lock:
        if not force and _cache_result is not None and now - _cache_ts < min_dt:
            return _cache_result

    result = detect_leader_grid(frame_bgr, cfg)
    with _cache_lock:
        _cache_result = result
        _cache_ts = time.time()
    return result


def get_cached_grid_detection() -> Optional[GridDetection]:
    with _cache_lock:
        return _cache_result


def draw_grid_overlay(frame_bgr: np.ndarray, detection: GridDetection) -> np.ndarray:
    out = frame_bgr.copy()
    if not detection.found:
        return out

    bbox = detection.bbox_or_centers()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if detection.centers is not None:
            cx = int(detection.centers[0, 0, 0])
            cy = int(detection.centers[0, 0, 1])
            cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
        parts = []
        if detection.distance_signal is not None:
            parts.append(f"dist={detection.distance_signal:.2f}")
        if detection.span_px is not None:
            parts.append(f"span={detection.span_px:.0f}px")
        if parts:
            cv2.putText(
                out,
                " ".join(parts),
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    return out
