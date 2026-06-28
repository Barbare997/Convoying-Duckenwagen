"""YOLO leader truck tracking for follower convoy (primary over grid fallback)."""

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from tasks.project.packages.leader_grid import GridDetection, _pattern_size

_DET_AGENT = None  # type: Optional[Any]
_DET_LOCK = threading.Lock()

_cache_lock = threading.Lock()
_cache_ts = 0.0
_cache_result = None  # type: Optional[GridDetection]
_last_raw_dets: List[Tuple] = []


def set_detector_agent(det_agent) -> None:
    """Wire ObjectDetectionAgent from project server startup."""
    global _DET_AGENT
    with _DET_LOCK:
        _DET_AGENT = det_agent


def get_detector_agent():
    with _DET_LOCK:
        return _DET_AGENT


def reset_leader_detector_cache() -> None:
    global _cache_result, _cache_ts, _last_raw_dets
    with _cache_lock:
        _cache_result = None
        _cache_ts = 0.0
        _last_raw_dets = []


def get_cached_leader_detector() -> Optional[GridDetection]:
    with _cache_lock:
        return _cache_result


def leader_detector_ready(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """True when YOLO leader detector is enabled and model weights are loaded."""
    c = cfg or {}
    if not bool(c.get("leader_detector_enabled", True)):
        return False
    det_agent = get_detector_agent()
    return det_agent is not None and bool(getattr(det_agent, "model_loaded", False))


def get_last_raw_detections() -> List[Tuple]:
    with _cache_lock:
        return list(_last_raw_dets)


def get_detector_status(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Status for web UI — model load state + last leader sighting."""
    det_agent = get_detector_agent()
    cached = get_cached_leader_detector()
    out: Dict[str, Any] = {
        "enabled": bool((cfg or {}).get("leader_detector_enabled", True)),
        "model_loaded": False,
        "trt_building": False,
        "trt_build_elapsed_s": 0,
        "load_error": None,
        "backend": None,
        "last_found": bool(cached.found) if cached is not None else False,
        "last_source": getattr(cached, "source", None) if cached is not None else None,
    }
    if det_agent is None:
        out["load_error"] = "Detector not started (restart project task)"
        return out

    out["model_loaded"] = bool(getattr(det_agent, "model_loaded", False))
    out["trt_building"] = bool(getattr(det_agent, "trt_building", False))
    out["trt_build_elapsed_s"] = int(getattr(det_agent, "trt_build_elapsed", 0))
    out["load_error"] = getattr(det_agent, "load_error", None)
    out["backend"] = getattr(det_agent, "_backend", None)
    out["conf_threshold"] = float(getattr(det_agent, "conf_threshold", 0.5))

    if cached is not None and cached.found:
        out["score"] = round(float(cached.quality or 0.0), 3)
        if cached.span_px is not None:
            out["span_px"] = round(float(cached.span_px), 1)
        if cached.center_x is not None:
            out["center_x"] = round(float(cached.center_x), 1)
    return out


def _height_to_distance_signal(height_px: float, cfg: Dict[str, Any]) -> float:
    safe = float(cfg.get("leader_detector_safe_px", 14.0))
    stop = float(cfg.get("leader_detector_stop_px", 50.0))
    if stop <= safe:
        stop = safe + 1.0
    return float(np.clip((height_px - safe) / (stop - safe), 0.0, 1.0))


def _bbox_heading(x1: int, y1: int, x2: int, y2: int, frame_w: float) -> float:
    """Rough yaw from bbox: offset of center from frame center, normalized."""
    cx = 0.5 * (float(x1) + float(x2))
    return float(np.clip((cx - frame_w / 2.0) / max(frame_w / 2.0, 1.0), -1.0, 1.0))


def _leader_target_class(cfg: Dict[str, Any]) -> int:
    """YOLO class id for convoy leader — 1=truck (0=duckie, 2=sign are ignored)."""
    return int(cfg.get("leader_detector_class", 1))


def _filter_leader_class(detections: List[Tuple], cfg: Dict[str, Any]) -> List[Tuple]:
    target_cls = _leader_target_class(cfg)
    return [d for d in detections if int(d[2]) == target_cls]


def _pick_leader_bbox(
    detections: List[Tuple],
    frame_h: int,
    frame_w: int,
    cfg: Dict[str, Any],
) -> Optional[Tuple[Tuple[int, int, int, int], float, int]]:
    """Choose the leader truck bbox ahead of us (not bottom-of-frame self)."""
    target_cls = _leader_target_class(cfg)
    min_area = float(cfg.get("leader_detector_min_area", 500.0))
    roi_top = int(frame_h * float(cfg.get("leader_detector_roi_top_frac", 0.0)))
    roi_bot = int(frame_h * float(cfg.get("leader_detector_roi_bottom_frac", 0.82)))

    best = None
    best_score = -1.0
    for bbox, score, cls_id in detections:
        if int(cls_id) != target_cls:
            continue
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = float(bw * bh)
        if area < min_area:
            continue
        cy = 0.5 * (y1 + y2)
        if cy < roi_top or cy > roi_bot:
            continue
        # Prefer confident, large, and nearer the horizon (smaller y).
        dist_prior = 1.0 - float(cy) / max(float(frame_h), 1.0)
        rank = float(score) * (area ** 0.35) * (0.55 + 0.45 * dist_prior)
        if rank > best_score:
            best_score = rank
            best = (bbox, float(score), int(cls_id))
    return best


def detect_leader_detector(frame_bgr, cfg: Dict[str, Any]) -> GridDetection:
    pattern = _pattern_size(cfg)
    empty = GridDetection(False, None, 0.0, None, pattern, source="detector")
    if frame_bgr is None or not bool(cfg.get("leader_detector_enabled", True)):
        return empty

    det_agent = get_detector_agent()
    if det_agent is None or not getattr(det_agent, "model_loaded", False):
        return empty

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]
    try:
        detections = det_agent.detect(frame_rgb)
    except Exception as exc:
        print(f"[Project][Detector] inference failed: {exc}", flush=True)
        return empty

    if detections is None:
        with _cache_lock:
            if _cache_result is not None and _cache_result.found:
                return _cache_result
        return empty

    leader_dets = _filter_leader_class(detections, cfg)
    with _cache_lock:
        _last_raw_dets[:] = leader_dets

    pick = _pick_leader_bbox(leader_dets, h, w, cfg)
    if pick is None:
        return empty

    (x1, y1, x2, y2), score, _cls = pick
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = 0.5 * (float(x1) + float(x2))
    cy = 0.5 * (float(y1) + float(y2))
    span_scale = float(cfg.get("leader_detector_span_scale", 0.55))
    span_px = max(1.0, float(bh) * span_scale)
    signal = _height_to_distance_signal(float(bh), cfg)
    aspect = float(bw) / float(bh)
    heading = _bbox_heading(x1, y1, x2, y2, float(w))
    centers = np.array([[[cx, cy]]], dtype=np.float32)

    return GridDetection(
        True,
        signal,
        float(score),
        centers,
        pattern,
        span_px=span_px,
        bbox=(int(x1), int(y1), int(x2), int(y2)),
        heading=heading,
        center_x=cx,
        source="detector",
    )


def fetch_leader_detector(
    frame_bgr,
    cfg: Dict[str, Any],
    *,
    force: bool = False,
) -> GridDetection:
    global _cache_ts, _cache_result
    if str(cfg.get("role", "leader")).lower() != "follower":
        return GridDetection(False, None, 0.0, None, _pattern_size(cfg), source="detector")

    hz = float(cfg.get("leader_detector_hz", cfg.get("grid_detect_hz", 8)))
    min_dt = 1.0 / max(1.0, hz)
    now = time.time()
    with _cache_lock:
        if not force and _cache_result is not None and now - _cache_ts < min_dt:
            return _cache_result

    result = detect_leader_detector(frame_bgr, cfg)
    with _cache_lock:
        _cache_result = result
        _cache_ts = time.time()
    return result


def render_leader_camera_overlay(frame_bgr, cfg: Dict[str, Any]) -> np.ndarray:
    """Draw YOLO boxes on the main camera; status text while model loads."""
    if frame_bgr is None:
        return frame_bgr
    out = frame_bgr.copy()
    if not bool(cfg.get("leader_detector_enabled", True)):
        return out

    det_agent = get_detector_agent()
    if det_agent is None:
        cv2.putText(
            out, "Detector starting...", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
        )
        return out
    if getattr(det_agent, "trt_building", False):
        elapsed = int(getattr(det_agent, "trt_build_elapsed", 0))
        cv2.putText(
            out, f"Loading model (TensorRT {elapsed}s)...", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA,
        )
        return out
    if not getattr(det_agent, "model_loaded", False):
        err = getattr(det_agent, "load_error", None) or "Model not loaded"
        cv2.putText(
            out, str(err)[:64], (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1, cv2.LINE_AA,
        )
        return out

    cached = get_cached_leader_detector()
    with _cache_lock:
        cache_age = time.time() - _cache_ts
    hz = float(cfg.get("leader_detector_hz", cfg.get("grid_detect_hz", 8)))
    min_dt = 1.0 / max(1.0, hz)
    if cached is None or cache_age >= min_dt:
        fetch_leader_detector(frame_bgr, cfg)
    dets = get_last_raw_detections()
    if dets:
        try:
            from servers.object_detection.visualization import draw_detections
            out = draw_detections(out, dets)
        except ImportError:
            for bbox, score, cls_id in dets:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 215, 255), 2)

    leader = get_cached_leader_detector()
    if leader is not None and leader.found and leader.bbox is not None:
        x1, y1, x2, y2 = leader.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = "LEADER"
        if leader.quality is not None:
            label += f" {leader.quality:.2f}"
        cv2.putText(
            out, label, (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA,
        )
    return out


def draw_detector_overlay(frame_bgr: np.ndarray, detection: GridDetection) -> np.ndarray:
    out = frame_bgr.copy()
    if not detection.found:
        return out
    bbox = detection.bbox_or_centers()
    if bbox is None:
        return out
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 215, 255), 2)
    if detection.centers is not None:
        cx = int(detection.centers[0, 0, 0])
        cy = int(detection.centers[0, 0, 1])
        cv2.circle(out, (cx, cy), 5, (0, 140, 255), -1)
    parts = ["detector"]
    if detection.quality is not None:
        parts.append(f"conf={detection.quality:.2f}")
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
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def leader_spacing_span_px(det, cfg):
    """Raw span_px for spacing — bbox height scale for detector, grid dot spacing for grid."""
    if det is None or det.span_px is None:
        return None
    return float(det.span_px)


def leader_spacing_target_px(det, cfg):
    """Desired follow gap in the same px units as span_px for this tracking source."""
    if getattr(det, "source", None) == "detector":
        return float(cfg.get("leader_detector_span_target_px", 38.0))
    return float(cfg.get("span_target_px", 7.0))


def leader_spacing_too_close_for_source(cfg, source):
    """Creep threshold for detector vs grid span units."""
    if source == "detector":
        explicit = cfg.get("leader_detector_span_too_close_px")
        if explicit is not None:
            return float(explicit)
        det_target = float(cfg.get("leader_detector_span_target_px", 38.0))
        grid_target = float(cfg.get("span_target_px", 7.0))
        grid_close = float(cfg.get("span_too_close_px", 18.0))
        return det_target * (grid_close / max(grid_target, 1.0))
    return float(cfg.get("span_too_close_px", 18.0))


def leader_spacing_too_close_px(det, cfg):
    """Creep speed threshold in the same px units as span_px for this tracking source."""
    source = getattr(det, "source", None) if det is not None else None
    return leader_spacing_too_close_for_source(cfg, source or "grid")
