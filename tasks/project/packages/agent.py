import math
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
 
import cv2
import yaml
from tasks.project.packages.follower_spacing import GridSpacingController
from tasks.project.packages.intersection_follow import (
    LeaderTurnTracker,
    intersection_phase_deadlines,
    intersection_turn_schedule,
    intersection_wheel_commands,
    measure_red_at_line,
    TURN_LEFT,
    TURN_RIGHT,
    TURN_STRAIGHT,
)
from tasks.project.packages.leader_grid import (
    GridDetection,
    draw_grid_overlay,
    fetch_leader_grid,
    fetch_leader_tracking,
    get_cached_grid_detection,
    get_cached_leader_tracking,
)
from tasks.project.packages.leader_detector import (
    fetch_leader_detector,
    get_cached_leader_detector,
    get_detector_status,
    leader_detector_ready,
    leader_spacing_span_px,
    leader_spacing_target_px,
    leader_spacing_too_close_for_source,
    leader_spacing_too_close_px,
    render_leader_camera_overlay,
    set_detector_agent,
)
from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent


def _resolve_project_config_path() -> str:
    """Canonical project_config.yaml (repo root, or KVATITOWN_ROOT on deployed bot)."""
    env_root = os.environ.get("KVATITOWN_ROOT", "").strip()
    if env_root:
        return os.path.normpath(os.path.join(env_root, "config", "project_config.yaml"))
    try:
        from launcher.config import CONFIG_DIR
        return str(CONFIG_DIR / "project_config.yaml")
    except Exception:
        return os.path.normpath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "config", "project_config.yaml",
            ),
        )


_CONFIG_PATH = _resolve_project_config_path()
 
# States
STATE_CRUISING = "CRUISING"
STATE_SLOW = "SLOW"
STATE_STOPPING = "STOPPING"
STATE_STOPPED = "STOPPED"
STATE_INTERSECTION = "INTERSECTION"

# Events
EVENT_NORMAL = "EVENT_NORMAL"
EVENT_SLOW_SIGN = "EVENT_SLOW_SIGN"
EVENT_STOP_SIGN = "EVENT_STOP_SIGN"
EVENT_LEADER_LOST = "EVENT_LEADER_LOST"

_status_lock = threading.Lock()
_convoy_ui_status: Dict[str, Any] = {
    "state": STATE_STOPPED,
    "speed": 0.0,
    "event": EVENT_NORMAL,
    "tag_ids": [],
    "dist_signal": None,
    "leader_visible": False,
}

_apriltag_detector = None
_apriltag_init_attempted = False
_sign_runtime = {
    "candidate_event": EVENT_NORMAL,
    "candidate_count": 0,
    "active_until": 0.0,
    # Stop sign: distance-based trigger + rearm after hold.
    "stop_confirm_count": 0,
    "stop_triggered_latch": False,
    "stop_pending_rearm": False,
    "stop_rearm_until": 0.0,
    # Slow sign: distance-based engage + loss confirm to resume.
    "slow_confirm_count": 0,
    "slow_engaged": False,
    "slow_loss_streak": 0,
}
_lane_agent: Optional[LaneServoingAgent] = None
_driving_enabled = False
_driving_lock = threading.Lock()

# Manual convoy commands (Plan B when AprilTags unavailable).
MANUAL_CRUISING = "CRUISING"
MANUAL_SLOW = "SLOW"
MANUAL_STOPPED = "STOPPED"
_VALID_MANUAL_COMMANDS = {MANUAL_CRUISING, MANUAL_SLOW, MANUAL_STOPPED}

_manual_convoy_cmd = MANUAL_CRUISING
_manual_lock = threading.Lock()
_manual_slow_until = 0.0

_test_turn_lock = threading.Lock()
_pending_test_turn: Optional[str] = None
_follower_test_turn_active = False

_config_io_lock = threading.Lock()
_last_good_raw: Dict[str, Any] = {}
_runtime_intersection_turn: Dict[str, float] = {}
_runtime_intersection_lock = threading.Lock()

_INTERSECTION_TURN_KEYS = (
    "intersection_left_preamble_s",
    "intersection_right_preamble_s",
    "intersection_turn_straight_s",
    "intersection_turn_left_s",
    "intersection_turn_right_s",
    "intersection_turn_tail_straight_s",
    "intersection_turn_speed",
    "intersection_turn_inner_ratio",
    "intersection_turn_outer_ratio",
)


def _read_project_config_raw() -> Dict[str, Any]:
    """Read project_config.yaml; keep last good copy if the file is briefly unreadable."""
    global _last_good_raw
    with _config_io_lock:
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            if isinstance(raw, dict) and raw:
                _last_good_raw = dict(raw)
            return dict(_last_good_raw)
        except Exception:
            return dict(_last_good_raw)


def get_project_config_path() -> str:
    return _CONFIG_PATH


def _merge_project_config_raw(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Atomically read-merge-write project_config.yaml (prevents slider races)."""
    global _last_good_raw
    if not updates:
        return dict(_last_good_raw)
    with _config_io_lock:
        raw: Dict[str, Any] = {}
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                raw = dict(loaded)
        except Exception as exc:
            if _last_good_raw:
                raw = dict(_last_good_raw)
            else:
                raise OSError(
                    f"Cannot read project config at {_CONFIG_PATH}: {exc}",
                ) from exc
        for key, val in updates.items():
            raw[key] = val
        tmp_path = f"{_CONFIG_PATH}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.dump(raw, f, default_flow_style=False)
            os.replace(tmp_path, _CONFIG_PATH)
        except Exception as exc:
            try:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            raise OSError(
                f"Cannot write project config at {_CONFIG_PATH}: {exc}",
            ) from exc
        _last_good_raw = dict(raw)
        _invalidate_config_cache()
        return dict(raw)


try:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as _bootstrap_f:
        _bootstrap_raw = yaml.safe_load(_bootstrap_f) or {}
    if isinstance(_bootstrap_raw, dict) and _bootstrap_raw:
        _last_good_raw = dict(_bootstrap_raw)
except Exception:
    pass


def _write_project_config_raw(raw: Dict[str, Any]) -> None:
    """Legacy full-file write — prefer _merge_project_config_raw for UI patches."""
    _merge_project_config_raw(dict(raw))


def _intersection_turn_from_cfg(cfg: Dict[str, Any]) -> Dict[str, float]:
    return {
        "intersection_left_preamble_s": float(cfg.get("intersection_left_preamble_s", 0.69)),
        "intersection_right_preamble_s": float(cfg.get("intersection_right_preamble_s", 0.44)),
        "intersection_turn_straight_s": float(cfg.get("intersection_turn_straight_s", 1.75)),
        "intersection_turn_left_s": float(cfg.get("intersection_turn_left_s", 2.38)),
        "intersection_turn_right_s": float(cfg.get("intersection_turn_right_s", 1.06)),
        "intersection_turn_tail_straight_s": float(cfg.get("intersection_turn_tail_straight_s", 0.0)),
        "intersection_turn_speed": float(cfg.get("intersection_turn_speed", 0.32)),
        "intersection_turn_inner_ratio": float(cfg.get("intersection_turn_inner_ratio", 0.38)),
        "intersection_turn_outer_ratio": float(cfg.get("intersection_turn_outer_ratio", 0.72)),
    }


def _publish_runtime_intersection_turn(params: Dict[str, float]) -> None:
    with _runtime_intersection_lock:
        _runtime_intersection_turn.clear()
        _runtime_intersection_turn.update(params)


_config_cache_base: Optional[Dict[str, Any]] = None
_config_cache_mtime: float = -1.0


def _invalidate_config_cache() -> None:
    global _config_cache_mtime
    _config_cache_mtime = -1.0


def load_config() -> Dict[str, Any]:
    # Load only the fields needed by the skeleton. Missing file -> safe defaults.
    global _config_cache_base, _config_cache_mtime
    try:
        mtime = os.path.getmtime(_CONFIG_PATH)
    except OSError:
        mtime = 0.0
    if _config_cache_base is None or mtime != _config_cache_mtime:
        cfg = _read_project_config_raw()
        _config_cache_base = _build_config_dict(cfg)
        _config_cache_mtime = mtime
    out = dict(_config_cache_base)
    with _runtime_intersection_lock:
        if _runtime_intersection_turn:
            out.update(_runtime_intersection_turn)
    return out


def _build_config_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": str(cfg.get("role", "leader")).strip().lower(),
        "cruise_speed": float(cfg.get("cruise_speed", 0.32)),
        "slow_speed": float(cfg.get("slow_speed", 0.16)),
        "follower_max_speed": float(cfg.get("follower_max_speed", 0.368)),
        "follower_min_speed": float(cfg.get("follower_min_speed", 0.0)),
        "follower_require_leader": bool(cfg.get("follower_require_leader", True)),
        "follower_spacing_warmup_frames": int(cfg.get("follower_spacing_warmup_frames", 8)),
        "follower_catchup_margin": float(cfg.get("follower_catchup_margin", 0.06)),
        "follower_lane_fallback_speed": float(cfg.get("follower_lane_fallback_speed", 0.32)),
        "span_target_px": float(cfg.get("span_target_px", 32.0)),
        "spacing_kp": float(cfg.get("spacing_kp", 0.010)),
        "spacing_kd": float(cfg.get("spacing_kd", 0.018)),
        "spacing_span_dot_alpha": float(cfg.get("spacing_span_dot_alpha", 0.35)),
        "spacing_leader_alpha": float(cfg.get("spacing_leader_alpha", 0.12)),
        "spacing_leader_ff_gain": float(cfg.get("spacing_leader_ff_gain", 0.012)),
        "spacing_steady_span_px": float(cfg.get("spacing_steady_span_px", 5.0)),
        "spacing_steady_span_dot": float(cfg.get("spacing_steady_span_dot", 10.0)),
        "span_too_close_px": float(cfg.get("span_too_close_px", 44.0)),
        "span_too_close_speed": float(cfg.get("span_too_close_speed", 0.05)),
        "grid_detect_hz": float(cfg.get("grid_detect_hz", 10)),
        "leader_loss_confirm_frames": int(cfg.get("leader_loss_confirm_frames", 3)),
        "grid_cols": int(cfg.get("grid_cols", 7)),
        "grid_rows": int(cfg.get("grid_rows", 3)),
        "grid_blob_min_area": float(cfg.get("grid_blob_min_area", 20)),
        "grid_blob_max_area": float(cfg.get("grid_blob_max_area", 12000)),
        "grid_downscale": float(cfg.get("grid_downscale", 0.5)),
        "grid_roi_pad_px": int(cfg.get("grid_roi_pad_px", 60)),
        "grid_roi_downscale": float(cfg.get("grid_roi_downscale", 1.0)),
        "grid_far_search": bool(cfg.get("grid_far_search", True)),
        "grid_far_band_top_frac": float(cfg.get("grid_far_band_top_frac", 0.10)),
        "grid_far_band_bot_frac": float(cfg.get("grid_far_band_bot_frac", 0.45)),
        "grid_lost_grace_frames": int(cfg.get("grid_lost_grace_frames", 6)),
        "grid_use_clustering": bool(cfg.get("grid_use_clustering", False)),
        "grid_blob_min_circularity": float(cfg.get("grid_blob_min_circularity", 0.6)),
        "grid_safe_px": float(cfg.get("grid_safe_px", 20.0)),
        "grid_stop_px": float(cfg.get("grid_stop_px", 42.0)),
        "leader_detector_enabled": bool(cfg.get("leader_detector_enabled", True)),
        "leader_detector_class": int(cfg.get("leader_detector_class", 1)),
        "leader_detector_hz": float(cfg.get("leader_detector_hz", 8.0)),
        "leader_detector_min_area": float(cfg.get("leader_detector_min_area", 500.0)),
        "leader_detector_roi_top_frac": float(cfg.get("leader_detector_roi_top_frac", 0.0)),
        "leader_detector_roi_bottom_frac": float(cfg.get("leader_detector_roi_bottom_frac", 0.82)),
        "leader_detector_span_scale": float(cfg.get("leader_detector_span_scale", 0.55)),
        "leader_detector_span_target_px": float(cfg.get("leader_detector_span_target_px", 36.0)),
        "leader_detector_safe_px": float(cfg.get("leader_detector_safe_px", 14.0)),
        "leader_detector_stop_px": float(cfg.get("leader_detector_stop_px", 50.0)),
        "leader_grid_fallback_enabled": bool(cfg.get("leader_grid_fallback_enabled", True)),
        "intersection_blue_straight_center_px": float(
            cfg.get("intersection_blue_straight_center_px", 18.0)
        ),
        "stop_hold_s": float(cfg.get("stop_hold_s", 2.0)),
        "slow_hold_s": float(cfg.get("slow_hold_s", 4.0)),
        "decel_time_s": float(cfg.get("decel_time_s", 1.5)),
        "decel_steps": int(cfg.get("decel_steps", 10)),
        "speed_ramp_s": float(cfg.get("speed_ramp_s", 1.25)),
        "stop_tag_ids": [int(x) for x in cfg.get("stop_tag_ids", [])],
        "slow_tag_ids": [int(x) for x in cfg.get("slow_tag_ids", [])],
        "sign_confirm_frames": int(cfg.get("sign_confirm_frames", 3)),
        "sign_cooldown_s": float(cfg.get("sign_cooldown_s", 2.0)),
        "sign_center_roi": float(cfg.get("sign_center_roi", 1.0)),
        # Stop sign: trigger when tag is within sign_stop_distance_m (pose / tag width).
        "sign_stop_distance_m": float(cfg.get("sign_stop_distance_m", 0.25)),
        "sign_stop_rearm_s": float(cfg.get("sign_stop_rearm_s", 6.0)),
        # Slow sign: distance (pose_t from dt_apriltags) or legacy delay mode.
        "sign_slow_mode": str(cfg.get("sign_slow_mode", "distance")).strip().lower(),
        "sign_slow_distance_m": float(cfg.get("sign_slow_distance_m", 0.40)),
        "sign_slow_loss_confirm_frames": int(cfg.get("sign_slow_loss_confirm_frames", 2)),
        "sign_slow_delay_s": float(cfg.get("sign_slow_delay_s", 2.5)),
        "sign_tag_size_m": float(cfg.get("sign_tag_size_m", 0.065)),
        "camera_fx": float(cfg.get("camera_fx", 337.8)),
        "camera_fy": float(cfg.get("camera_fy", 337.3)),
        "camera_cx": float(cfg.get("camera_cx", 324.0)),
        "camera_cy": float(cfg.get("camera_cy", 239.3)),
        "camera_calib_width": float(cfg.get("camera_calib_width", 640)),
        "camera_calib_height": float(cfg.get("camera_calib_height", 480)),
        # True: send LaneServoingAgent PWM as-is (same as visual_lane_servoing task).
        # False: scale PWM so max(left,right) = cruise/slow speed from this file.
        "lane_use_direct_pwm": bool(cfg.get("lane_use_direct_pwm", False)),
        "lane_ignore_bottom_frac": float(cfg.get("lane_ignore_bottom_frac", 0.0)),
        "lane_ignore_bottom_at_intersection_frac": float(
            cfg.get("lane_ignore_bottom_at_intersection_frac", 0.35)
        ),
        # Sign input: apriltag | manual | both (manual overrides when not CRUISING).
        "sign_source": str(cfg.get("sign_source", "both")).strip().lower(),
        # Red-line intersection (follower): trigger only on near (bottom-band) red paint.
        "intersection_red_near_top_frac": float(cfg.get("intersection_red_near_top_frac", 0.72)),
        "intersection_red_far_top_frac": float(cfg.get("intersection_red_far_top_frac", 0.35)),
        "intersection_red_near_min_pixels": int(cfg.get("intersection_red_near_min_pixels", 3500)),
        "intersection_red_near_min_frac": float(cfg.get("intersection_red_near_min_frac", 0.055)),
        "intersection_red_near_far_ratio": float(cfg.get("intersection_red_near_far_ratio", 2.0)),
        "intersection_red_confirm_frames": int(cfg.get("intersection_red_confirm_frames", 3)),
        "intersection_red_approach_min_far_px": int(cfg.get("intersection_red_approach_min_far_px", 600)),
        "intersection_turn_track_window": int(cfg.get("intersection_turn_track_window", 20)),
        "intersection_turn_infer_lookback": int(cfg.get("intersection_turn_infer_lookback", 8)),
        "intersection_cx_drift_px": float(cfg.get("intersection_cx_drift_px", 30.0)),
        "intersection_heading_thresh": float(cfg.get("intersection_heading_thresh", 0.12)),
        "intersection_heading_sign": float(cfg.get("intersection_heading_sign", -1.0)),
        "intersection_straight_aspect_min": float(cfg.get("intersection_straight_aspect_min", 2.2)),
        "intersection_last_cx_offset_px": float(cfg.get("intersection_last_cx_offset_px", 28.0)),
        "intersection_turn_speed": float(cfg.get("intersection_turn_speed", 0.32)),
        "intersection_turn_inner_ratio": float(cfg.get("intersection_turn_inner_ratio", 0.27)),
        "intersection_turn_outer_ratio": float(cfg.get("intersection_turn_outer_ratio", 1.0)),
        "intersection_left_preamble_s": float(cfg.get("intersection_left_preamble_s", 0.69)),
        "intersection_right_preamble_s": float(cfg.get("intersection_right_preamble_s", 0.44)),
        "intersection_turn_straight_s": float(cfg.get("intersection_turn_straight_s", 1.75)),
        "intersection_turn_left_s": float(cfg.get("intersection_turn_left_s", 2.38)),
        "intersection_turn_right_s": float(cfg.get("intersection_turn_right_s", 1.06)),
        "intersection_turn_tail_straight_s": float(cfg.get("intersection_turn_tail_straight_s", 1.6)),
    }


def get_manual_convoy_command() -> str:
    with _manual_lock:
        return _manual_convoy_cmd


def set_manual_convoy_command(command: str) -> Dict[str, Any]:
    """Leader UI: latch Normal / Slow / Sign Stop (instant, no confirm frames)."""
    cmd = str(command).strip().upper()
    if cmd not in _VALID_MANUAL_COMMANDS:
        raise ValueError(f"command must be one of {sorted(_VALID_MANUAL_COMMANDS)}")

    global _manual_convoy_cmd, _manual_slow_until
    with _manual_lock:
        prev = _manual_convoy_cmd
        _manual_convoy_cmd = cmd
        if cmd == MANUAL_STOPPED:
            _manual_slow_until = 0.0
            _sign_runtime["candidate_event"] = EVENT_STOP_SIGN
            _sign_runtime["candidate_count"] = max(
                1, int(_sign_runtime.get("candidate_count", 0))
            )
            _sign_runtime["active_until"] = 0.0
        elif cmd == MANUAL_SLOW:
            hold_s = float(load_config().get("slow_hold_s", 4.0))
            _manual_slow_until = time.time() + max(0.5, hold_s)
            _sign_runtime["candidate_event"] = EVENT_SLOW_SIGN
            _sign_runtime["candidate_count"] = max(
                1, int(_sign_runtime.get("candidate_count", 0))
            )
            _sign_runtime["active_until"] = 0.0
        else:
            _manual_slow_until = 0.0
            _sign_runtime["candidate_event"] = EVENT_NORMAL
            _sign_runtime["candidate_count"] = 0
            _sign_runtime["active_until"] = 0.0

    if prev != cmd:
        print(f"[Project][Leader] Manual convoy command: {prev} -> {cmd}", flush=True)
    return {"ok": True, "command": cmd, "previous": prev}


def _normalize_test_turn_direction(direction: str) -> str:
    d = str(direction or "").strip().lower()
    if d in ("straight", "forward", "fwd", "go"):
        return TURN_STRAIGHT
    if d == TURN_LEFT:
        return TURN_LEFT
    if d == TURN_RIGHT:
        return TURN_RIGHT
    raise ValueError("direction must be left, right, or straight")


def request_follower_test_turn(direction: str) -> Dict[str, Any]:
    """Queue a timed intersection maneuver (works while paused — no Start required)."""
    global _pending_test_turn
    cfg = load_config()
    if str(cfg.get("role", "leader")).lower() != "follower":
        return {"status": "error", "message": "follower role required"}
    turn_dir = _normalize_test_turn_direction(direction)
    with _test_turn_lock:
        _pending_test_turn = turn_dir
    return {"status": "ok", "direction": turn_dir, "queued": True}


def get_follower_test_turn_status() -> Dict[str, Any]:
    with _test_turn_lock:
        return {
            "queued": _pending_test_turn,
            "active": _follower_test_turn_active,
        }


def _pop_follower_test_turn() -> Optional[str]:
    global _pending_test_turn
    with _test_turn_lock:
        turn_dir = _pending_test_turn
        _pending_test_turn = None
        return turn_dir


def _set_follower_test_turn_active(active: bool) -> None:
    global _follower_test_turn_active
    with _test_turn_lock:
        _follower_test_turn_active = bool(active)


def _follower_wheels_allowed() -> bool:
    with _test_turn_lock:
        test_active = _follower_test_turn_active
    return is_driving_enabled() or test_active


def _maybe_auto_clear_manual_slow(now: float) -> None:
    """After slow_hold_s, return manual convoy command to CRUISING (like auto-resume from stop)."""
    global _manual_slow_until
    with _manual_lock:
        if _manual_convoy_cmd != MANUAL_SLOW:
            return
        if _manual_slow_until <= 0.0 or now < _manual_slow_until:
            return
        _manual_slow_until = 0.0
    set_manual_convoy_command(MANUAL_CRUISING)
    print("[Project][Leader] Slow hold expired — auto-resume CRUISING", flush=True)


def ensure_apriltag_probe() -> bool:
    """Load AprilTag backend once (call from leader loop only)."""
    return _get_apriltag_detector() is not None


def apriltag_detector_ready() -> bool:
    return _apriltag_detector is not None


def _resolve_leader_event(
    state: str,
    tag_event: str,
    cfg: Dict[str, Any],
) -> str:
    """Merge AprilTag events with manual UI commands (manual wins when latched)."""
    source = str(cfg.get("sign_source", "both")).lower()
    manual_cmd = get_manual_convoy_command()
    tags_ok = apriltag_detector_ready()

    use_tags = source in ("apriltag", "both") and tags_ok
    use_manual = source in ("manual", "both")
    if source == "manual" or (source == "both" and not tags_ok):
        use_tags = False
        use_manual = True

    tag_e = tag_event if use_tags else EVENT_NORMAL

    if not use_manual:
        return tag_e

    if manual_cmd == MANUAL_STOPPED:
        if state == STATE_STOPPED:
            return EVENT_NORMAL
        return EVENT_STOP_SIGN

    if manual_cmd == MANUAL_SLOW:
        if state in (STATE_STOPPED, STATE_STOPPING):
            return EVENT_NORMAL
        return EVENT_SLOW_SIGN

    return tag_e


def get_grid_status(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cols = int(cfg.get("grid_cols", 7))
    rows = int(cfg.get("grid_rows", 3))
    cached = get_cached_leader_tracking() or get_cached_grid_detection()
    out = {
        "ready": True,
        "method": getattr(cached, "source", None) or "none",
        "pattern": f"{cols}x{rows}",
        "last_found": bool(cached.found) if cached is not None else False,
    }
    if cached is not None and cached.span_px is not None:
        out["span_px"] = round(float(cached.span_px), 1)
    if cached is not None and getattr(cached, "quality", None) is not None:
        out["score"] = round(float(cached.quality), 3)
    return out


def get_runtime_status(cfg: Dict[str, Any]) -> Dict[str, Any]:
    role = str(cfg.get("role", "leader")).lower()
    out: Dict[str, Any] = {
        "sign_source": str(cfg.get("sign_source", "both")),
        "manual_command": get_manual_convoy_command(),
        "driving_enabled": is_driving_enabled(),
    }
    if role == "follower":
        out["grid"] = get_grid_status(cfg)
        out["detector"] = get_detector_status(cfg)
        out["test_turn"] = get_follower_test_turn_status()
    elif role == "leader":
        out["apriltag_available"] = apriltag_detector_ready()
    return out


def _update_convoy_ui_status(**fields) -> None:
    with _status_lock:
        _convoy_ui_status.update(fields)


def get_convoy_ui_status() -> Dict[str, Any]:
    with _status_lock:
        return dict(_convoy_ui_status)
 
 
def _safe_stop(wheels) -> None:
    # Centralized hard-stop helper used in both roles and on exit.
    try:
        if wheels is not None:
            wheels.set_wheels_speed(0.0, 0.0)
    except Exception as e:
        print(f"[Project] Wheels stop failed: {e}")


_LED_OFF = [0.0, 0.0, 0.0]
_LED_WHITE = [1.0, 1.0, 1.0]
_LED_DIM_WHITE = [0.35, 0.35, 0.35]
_LED_RED = [1.0, 0.0, 0.0]


def _convoy_leds_off(leds) -> None:
    if leds is None:
        return
    try:
        leds.all_off()
    except Exception as e:
        print(f"[Project][LED] all_off failed: {e}")


def _apply_convoy_leds(
    leds,
    *,
    state: str,
    current_speed: float,
    cruise_speed: float,
    slow_speed: float,
    driving_enabled: bool,
) -> None:
    """Car-like lighting: front headlights when moving, rear red when braking."""
    if leds is None:
        return
    try:
        if not driving_enabled:
            leds.all_off()
            return

        state_u = str(state).upper()
        cruise = max(1e-6, float(cruise_speed))
        slow = max(0.0, float(slow_speed))

        if state_u in (STATE_STOPPED, STATE_STOPPING):
            leds.set_rgb(0, _LED_OFF)
            leds.set_rgb(2, _LED_OFF)
            leds.set_all_back(_LED_RED)
            return

        if state_u == STATE_SLOW:
            leds.set_all_front(_LED_DIM_WHITE)
            span = max(1e-6, cruise - slow)
            brake_mix = 1.0 - min(1.0, max(0.0, (float(current_speed) - slow) / span))
            brake_level = 0.55 + 0.45 * brake_mix
            rear = [min(1.0, _LED_RED[i] * brake_level) for i in range(3)]
            leds.set_all_back(rear)
            return

        # CRUISING: headlights on; rear red only while still slowing toward cruise.
        if float(current_speed) < cruise - 0.03:
            leds.set_all_front(_LED_DIM_WHITE)
            leds.set_all_back([0.65, 0.0, 0.0])
        else:
            leds.set_all_front(_LED_WHITE)
            leds.set_rgb(3, _LED_OFF)
            leds.set_rgb(4, _LED_OFF)
    except Exception as e:
        print(f"[Project][LED] update failed: {e}")
 
 
def smooth_stop(
    wheels,
    current_speed: float,
    decel_time_s: float,
    decel_steps: int,
    stop_event,
    lane_agent=None,
) -> None:
    """Ramp last lane PWM (or scalar speed) down to zero over configured time/steps."""
    if wheels is None:
        return

    speed0 = max(0.0, float(current_speed))
    steps = max(1, int(decel_steps))
    step_dt = max(0.0, float(decel_time_s)) / steps
    left0 = float(getattr(lane_agent, "_last_left", speed0)) if lane_agent is not None else speed0
    right0 = float(getattr(lane_agent, "_last_right", speed0)) if lane_agent is not None else speed0

    try:
        for i in range(steps - 1, -1, -1):
            if stop_event.is_set():
                break
            factor = i / steps
            wheels.set_wheels_speed(left0 * factor, right0 * factor)
            if step_dt > 0:
                time.sleep(step_dt)
    except Exception as e:
        print(f"[Project] smooth_stop failed: {e}")
    finally:
        _safe_stop(wheels)


def _ramp_toward(current: float, target: float, max_delta: float) -> float:
    """Move current toward target by at most max_delta (non-blocking speed ramp)."""
    if max_delta <= 0.0:
        return target
    diff = float(target) - float(current)
    if abs(diff) <= max_delta:
        return float(target)
    return float(current) + (max_delta if diff > 0.0 else -max_delta)


def _speed_ramp_delta(cfg: Dict[str, Any], frame_dt: float) -> float:
    cruise = float(cfg.get("cruise_speed", 0.32))
    slow = float(cfg.get("slow_speed", 0.16))
    ramp_s = max(0.05, float(cfg.get("speed_ramp_s", 1.25)))
    span = max(0.05, abs(cruise - slow))
    return span / ramp_s * max(1e-3, float(frame_dt))


def next_state(current_state: str, event: str) -> str:
    if event == EVENT_LEADER_LOST:
        return STATE_STOPPING
    if event == EVENT_STOP_SIGN:
        return STATE_STOPPING
    if event == EVENT_SLOW_SIGN:
        return STATE_SLOW if current_state != STATE_STOPPED else STATE_STOPPED

    # EVENT_NORMAL
    if current_state == STATE_SLOW:
        return STATE_CRUISING
    if current_state == STATE_STOPPED:
        return STATE_STOPPED
    return current_state


class _TagDetection:
    __slots__ = ("tag_id", "center", "distance_m")

    def __init__(
        self,
        tag_id: int,
        center: Tuple[float, float],
        distance_m: Optional[float] = None,
    ):
        self.tag_id = tag_id
        self.center = center
        self.distance_m = distance_m


def _distance_from_pose_t(pose_t) -> Optional[float]:
    """Forward depth from dt_apriltags/apriltags pose_t (meters)."""
    try:
        if hasattr(pose_t, "reshape"):
            t = pose_t.reshape(-1)
            x, y, z = float(t[0]), float(t[1]), float(t[2])
        elif isinstance(pose_t, (list, tuple)) and len(pose_t) >= 3:
            x, y, z = float(pose_t[0]), float(pose_t[1]), float(pose_t[2])
        else:
            return None
        if z > 0.05:
            return z
        n = math.sqrt(x * x + y * y + z * z)
        return n if n > 0.05 else None
    except Exception:
        return None


def _distance_from_tag_width_px(tag_width_px: float, fx: float, tag_size_m: float) -> Optional[float]:
    if tag_width_px <= 1.0 or fx <= 0.0 or tag_size_m <= 0.0:
        return None
    return float(fx * tag_size_m / tag_width_px)


def _apriltag_camera_params(
    cfg: Dict[str, Any],
    frame_gray,
) -> Tuple[float, float, float, float]:
    fx = float(cfg.get("camera_fx", 337.8))
    fy = float(cfg.get("camera_fy", 337.3))
    cx = float(cfg.get("camera_cx", 324.0))
    cy = float(cfg.get("camera_cy", 239.3))
    ref_w = max(1.0, float(cfg.get("camera_calib_width", 640)))
    ref_h = max(1.0, float(cfg.get("camera_calib_height", 480)))
    if frame_gray is not None:
        h, w = frame_gray.shape[:2]
        sx, sy = float(w) / ref_w, float(h) / ref_h
        return fx * sx, fy * sy, cx * sx, cy * sy
    return fx, fy, cx, cy


class _WrappedAprilTagBackend:
    """Wraps apriltags / dt_apriltags / pupil_apriltags Detector APIs."""

    def __init__(self, detector):
        self._det = detector

    def _raw_detect(self, frame_gray, camera_params, tag_size_m):
        if camera_params is not None and tag_size_m is not None:
            params = list(camera_params)
            for kwargs in (
                {
                    "estimate_tag_pose": True,
                    "camera_params": params,
                    "tag_size": tag_size_m,
                },
                {
                    "estimate_tag_pose": True,
                    "camera_params": tuple(params),
                    "tag_size": tag_size_m,
                },
            ):
                try:
                    return self._det.detect(frame_gray, **kwargs)
                except TypeError:
                    continue
        return self._det.detect(frame_gray)

    def detect(
        self,
        frame_gray,
        *,
        camera_params: Optional[Tuple[float, float, float, float]] = None,
        tag_size_m: Optional[float] = None,
    ) -> List[_TagDetection]:
        fx = camera_params[0] if camera_params is not None else None
        out: List[_TagDetection] = []
        for det in self._raw_detect(frame_gray, camera_params, tag_size_m):
            parsed = _parse_tag_detection(det, fx=fx, tag_size_m=tag_size_m)
            if parsed is not None:
                out.append(parsed)
        return out


def _parse_tag_detection(
    det,
    *,
    fx: Optional[float] = None,
    tag_size_m: Optional[float] = None,
) -> Optional[_TagDetection]:
    c = getattr(det, "center", None)
    if c is None:
        return None
    tag_id = getattr(det, "tag_id", None)
    if tag_id is None:
        tag_id = getattr(det, "id", None)
    if tag_id is None:
        return None

    distance_m = None
    pose_t = getattr(det, "pose_t", None)
    if pose_t is not None:
        distance_m = _distance_from_pose_t(pose_t)
    if distance_m is None and fx is not None and tag_size_m is not None:
        corners = getattr(det, "corners", None)
        if corners is not None:
            try:
                pts = corners.reshape(-1, 2)
                widths = [
                    math.hypot(float(pts[i, 0] - pts[j, 0]), float(pts[i, 1] - pts[j, 1]))
                    for i in range(4)
                    for j in range(i + 1, 4)
                ]
                distance_m = _distance_from_tag_width_px(max(widths), fx, tag_size_m)
            except Exception:
                pass

    return _TagDetection(int(tag_id), (float(c[0]), float(c[1])), distance_m)


def _build_detector_from_module(mod_name: str):
    import importlib

    last_err = None
    mod_candidates = (mod_name, f"{mod_name}.apriltags")
    for candidate in mod_candidates:
        try:
            mod = importlib.import_module(candidate)
        except ImportError as e:
            last_err = e
            continue

        detector_cls = getattr(mod, "Detector", None)
        if detector_cls is None:
            continue

        opts_cls = getattr(mod, "DetectorOptions", None) or getattr(mod, "Detectoroptions", None)
        if opts_cls is not None:
            for factory in (
                lambda: detector_cls(opts_cls(families="tag36h11")),
                lambda: detector_cls(options=opts_cls(families="tag36h11")),
            ):
                try:
                    return factory()
                except (AttributeError, TypeError):
                    continue

        for kwargs in (
            {"families": "tag36h11", "quad_decimate": 1.0, "refine_edges": 1},
            {"families": "tag36h11"},
            {},
        ):
            try:
                return detector_cls(**kwargs)
            except (TypeError, AttributeError):
                continue

        raise RuntimeError(f"{candidate}.Detector could not be initialized")

    if last_err is not None:
        raise ImportError(f"{mod_name} not importable ({last_err})")
    raise ImportError(f"{mod_name} has no Detector class")


class _OpenCVAprilTagBackend:
    """Fallback when pupil_apriltags cannot be installed (common on Jetson)."""

    def __init__(self):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco not available")
        dict_id = getattr(cv2.aruco, "DICT_APRILTAG_36h11", None)
        if dict_id is None:
            raise RuntimeError("OpenCV lacks DICT_APRILTAG_36h11 (need OpenCV >= 4.7)")
        self._dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        if hasattr(cv2.aruco, "ArucoDetector"):
            params = cv2.aruco.DetectorParameters()
            self._detector = cv2.aruco.ArucoDetector(self._dictionary, params)
            self._legacy = False
        else:
            self._detector = None
            self._legacy = True

    def detect(
        self,
        frame_gray,
        *,
        camera_params: Optional[Tuple[float, float, float, float]] = None,
        tag_size_m: Optional[float] = None,
    ) -> List[_TagDetection]:
        if self._legacy:
            corners, ids, _rejected = cv2.aruco.detectMarkers(frame_gray, self._dictionary)
        else:
            corners, ids, _rejected = self._detector.detectMarkers(frame_gray)
        if ids is None or len(ids) == 0:
            return []
        fx = camera_params[0] if camera_params is not None else None
        out: List[_TagDetection] = []
        for i, tag_id in enumerate(ids.flatten()):
            pts = corners[i][0]
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            distance_m = None
            if fx is not None and tag_size_m is not None:
                edge_lens = [
                    math.hypot(float(pts[a, 0] - pts[b, 0]), float(pts[a, 1] - pts[b, 1]))
                    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))
                ]
                distance_m = _distance_from_tag_width_px(max(edge_lens), fx, tag_size_m)
            out.append(_TagDetection(int(tag_id), (cx, cy), distance_m))
        return out


_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")


def _probe_apriltag_sys_paths() -> None:
    for p in (
        "/usr/local/lib/python3.8/dist-packages",
        "/usr/local/lib/python3.6/dist-packages",
        "/usr/lib/python3/dist-packages",
        "/usr/local/lib/python3/dist-packages",
    ):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _log_apriltag_module_probe() -> None:
    import importlib

    for mod_name in ("apriltags", "dt_apriltags", "apriltag", "pupil_apriltags"):
        try:
            mod = importlib.import_module(mod_name)
            has_det = getattr(mod, "Detector", None) is not None
            _apriltag_log(
                f"[Project][AprilTag] import {mod_name}: ok (Detector={'yes' if has_det else 'no'})"
            )
        except ImportError as e:
            _apriltag_log(f"[Project][AprilTag] import {mod_name}: no ({e})")


def _try_install_pupil_from_vendor() -> bool:
    import glob

    wheels = sorted(glob.glob(os.path.join(_VENDOR_DIR, "pupil_apriltags*.whl")))
    if not wheels:
        return False
    wheel = wheels[-1]
    _apriltag_log(f"[Project][AprilTag] Trying bundled wheel {os.path.basename(wheel)} ...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--user", "--no-index", "--no-deps", wheel],
            timeout=120,
        )
        _apriltag_log("[Project][AprilTag] Bundled wheel install finished.")
        return True
    except Exception as e:
        _apriltag_log(f"[Project][AprilTag] Bundled wheel install failed ({e}).")
        return False


def _apriltag_log(msg: str) -> None:
    print(msg, flush=True)


def _create_pupil_apriltag_backend():
    import importlib

    detector_cls = importlib.import_module("pupil_apriltags").Detector
    det = detector_cls(families="tag36h11", quad_decimate=1.0, refine_edges=1)
    return _WrappedAprilTagBackend(det)


def _create_opencv_apriltag_backend():
    return _OpenCVAprilTagBackend()


def _create_module_backend(mod_name: str) -> _WrappedAprilTagBackend:
    return _WrappedAprilTagBackend(_build_detector_from_module(mod_name))


def _try_module_backend(mod_name: str, label: str):
    global _apriltag_detector
    try:
        _apriltag_detector = _create_module_backend(mod_name)
        _apriltag_log(f"[Project][AprilTag] Detector initialized ({label}).")
        return _apriltag_detector
    except ImportError as e:
        _apriltag_log(f"[Project][AprilTag] {mod_name} not usable ({e}).")
    except Exception as e:
        _apriltag_log(f"[Project][AprilTag] {mod_name} skipped ({e}).")
    return None


def _get_apriltag_detector():
    global _apriltag_detector, _apriltag_init_attempted
    if _apriltag_init_attempted:
        return _apriltag_detector

    _apriltag_init_attempted = True
    _probe_apriltag_sys_paths()
    cv_ver = getattr(cv2, "__version__", "unknown")
    _apriltag_log(f"[Project][AprilTag] OpenCV {cv_ver}; probing detectors (apriltags first)...")
    _log_apriltag_module_probe()

    # TA: `import apriltags` works on Duckiebot. Try before pupil/OpenCV/wrong `apriltag`.
    for mod_name, label in (
        ("apriltags", "apriltags"),
        ("dt_apriltags", "dt_apriltags"),
    ):
        det = _try_module_backend(mod_name, label)
        if det is not None:
            return det

    try:
        _apriltag_detector = _create_pupil_apriltag_backend()
        _apriltag_log("[Project][AprilTag] Detector initialized (pupil_apriltags).")
        return _apriltag_detector
    except ImportError as e:
        _apriltag_log(f"[Project][AprilTag] pupil_apriltags not installed ({e}).")
    except Exception as e:
        _apriltag_log(f"[Project][AprilTag] pupil_apriltags skipped ({e}).")

    try:
        _apriltag_detector = _create_opencv_apriltag_backend()
        _apriltag_log("[Project][AprilTag] Detector initialized (OpenCV tag36h11).")
        return _apriltag_detector
    except Exception as e:
        _apriltag_log(f"[Project][AprilTag] OpenCV tag36h11 unavailable ({e}).")

    # Singular `apriltag` on some bots is a different package without Detector — try last.
    det = _try_module_backend("apriltag", "apriltag")
    if det is not None:
        return det

    if _try_install_pupil_from_vendor():
        try:
            _apriltag_detector = _create_pupil_apriltag_backend()
            _apriltag_log("[Project][AprilTag] Detector initialized (pupil_apriltags from vendor wheel).")
            return _apriltag_detector
        except Exception as e:
            _apriltag_log(f"[Project][AprilTag] pupil_apriltags still unavailable ({e}).")

    _apriltag_detector = None
    _apriltag_log(
        "[Project][AprilTag] No detector on this bot. Use Convoy Sign Control buttons "
        "(sign_source: both). Ask TA which AprilTag module is installed (apriltags / dt_apriltags)."
    )
    return _apriltag_detector


#extracts the frame from the camera and converts it to grayscale
def _extract_frame_gray(camera) -> Tuple[Optional[Any], Optional[Any]]:
    if camera is None:
        return None, None
    try:
        ok, frame = camera.read()
    except Exception:
        return None, None
    if not ok or frame is None:
        return None, None

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = None
    return frame, gray


def _detect_apriltags(frame_gray, cfg: Optional[Dict[str, Any]] = None) -> List[_TagDetection]:
    detector = _get_apriltag_detector()
    if detector is None or frame_gray is None:
        return []
    camera_params = None
    tag_size_m = None
    if cfg is not None:
        tag_size_m = float(cfg.get("sign_tag_size_m", 0.065))
        camera_params = _apriltag_camera_params(cfg, frame_gray)
    try:
        return detector.detect(
            frame_gray,
            camera_params=camera_params,
            tag_size_m=tag_size_m,
        )
    except Exception as e:
        print(f"[Project][AprilTag] detect failed: {e}")
        return []


def _sign_tag_distance_from_detections(
    detections: List[Any],
    tag_ids: Set[int],
    frame_shape: Optional[Tuple[int, int]],
    center_roi: float,
) -> Tuple[bool, Optional[float]]:
    """Return (tag_visible_in_roi, nearest_distance_m)."""
    if not detections:
        return False, None

    cx_min, cx_max = -1.0, 2.0
    if frame_shape is not None:
        _h, w = frame_shape
        roi = max(0.2, min(1.0, float(center_roi)))
        margin = (1.0 - roi) * 0.5
        cx_min = margin * w
        cx_max = (1.0 - margin) * w

    nearest: Optional[float] = None
    for det in detections:
        try:
            tag_id = int(det.tag_id)
            c = det.center
            cx = float(c[0]) if c is not None else None
        except Exception:
            continue

        if tag_id not in tag_ids:
            continue
        if cx is not None and not (cx_min <= cx <= cx_max):
            continue

        dist = getattr(det, "distance_m", None)
        if dist is not None:
            nearest = float(dist) if nearest is None else min(nearest, float(dist))
        else:
            nearest = nearest if nearest is not None else float("inf")

    if nearest is None:
        return False, None
    if nearest == float("inf"):
        return True, None
    return True, nearest


def _sign_slow_from_detections(
    detections: List[Any],
    slow_ids: Set[int],
    frame_shape: Optional[Tuple[int, int]],
    center_roi: float,
) -> Tuple[bool, Optional[float]]:
    return _sign_tag_distance_from_detections(detections, slow_ids, frame_shape, center_roi)


def _sign_stop_from_detections(
    detections: List[Any],
    stop_ids: Set[int],
    frame_shape: Optional[Tuple[int, int]],
    center_roi: float,
) -> Tuple[bool, Optional[float]]:
    return _sign_tag_distance_from_detections(detections, stop_ids, frame_shape, center_roi)


def _reset_stop_tracker() -> None:
    _sign_runtime["stop_confirm_count"] = 0
    _sign_runtime["stop_triggered_latch"] = False
    _sign_runtime["stop_pending_rearm"] = False


def _stop_event_distance(
    seen_stop: bool,
    stop_distance_m: Optional[float],
    cfg: Dict[str, Any],
) -> str:
    """Trigger STOP when stop tag is within sign_stop_distance_m."""
    if bool(_sign_runtime.get("stop_triggered_latch", False)):
        return EVENT_NORMAL

    confirm_frames = max(1, int(cfg.get("sign_confirm_frames", 3)))
    threshold_m = max(0.05, float(cfg.get("sign_stop_distance_m", 0.25)))

    if not seen_stop or stop_distance_m is None:
        _sign_runtime["stop_confirm_count"] = 0
        return EVENT_NORMAL

    if float(stop_distance_m) > threshold_m:
        _sign_runtime["stop_confirm_count"] = 0
        return EVENT_NORMAL

    _sign_runtime["stop_confirm_count"] = int(_sign_runtime.get("stop_confirm_count", 0)) + 1
    if int(_sign_runtime["stop_confirm_count"]) < confirm_frames:
        return EVENT_NORMAL

    _sign_runtime["stop_confirm_count"] = 0
    _sign_runtime["stop_triggered_latch"] = True
    _sign_runtime["stop_pending_rearm"] = True
    print(
        f"[Project][Leader] Stop sign within {threshold_m:.2f}m "
        f"(d={float(stop_distance_m):.2f}m) — triggering STOP",
        flush=True,
    )
    return EVENT_STOP_SIGN


def _reset_slow_tracker() -> None:
    _sign_runtime["slow_confirm_count"] = 0
    _sign_runtime["slow_engaged"] = False
    _sign_runtime["slow_loss_streak"] = 0


def _mark_tag_slow_event(now: float, cooldown_s: float) -> None:
    _sign_runtime["candidate_event"] = EVENT_SLOW_SIGN
    _sign_runtime["candidate_count"] = max(1, int(_sign_runtime.get("candidate_count", 0)))
    _sign_runtime["active_until"] = now + cooldown_s


def reset_sign_detection_state() -> None:
    """Clear tag confirm/delay/arm timers — signs only accumulate while driving."""
    _sign_runtime["candidate_event"] = EVENT_NORMAL
    _sign_runtime["candidate_count"] = 0
    _sign_runtime["active_until"] = 0.0
    _sign_runtime["stop_rearm_until"] = 0.0
    _reset_stop_tracker()
    _reset_slow_tracker()


def _slow_event_distance(
    seen_slow: bool,
    slow_distance_m: Optional[float],
    cfg: Dict[str, Any],
    now: float,
    cooldown_s: float,
) -> str:
    """Start SLOW when slow tag is within sign_slow_distance_m; resume after tag loss."""
    confirm_frames = max(1, int(cfg.get("sign_confirm_frames", 3)))
    threshold_m = max(0.05, float(cfg.get("sign_slow_distance_m", 0.40)))
    loss_confirm = max(1, int(cfg.get("sign_slow_loss_confirm_frames", 2)))

    if bool(_sign_runtime.get("slow_engaged", False)):
        if seen_slow:
            _sign_runtime["slow_loss_streak"] = 0
        else:
            _sign_runtime["slow_loss_streak"] = int(_sign_runtime.get("slow_loss_streak", 0)) + 1

        if int(_sign_runtime["slow_loss_streak"]) >= loss_confirm:
            _reset_slow_tracker()
            return EVENT_NORMAL

        _mark_tag_slow_event(now, cooldown_s)
        return EVENT_SLOW_SIGN

    if not seen_slow or slow_distance_m is None:
        _sign_runtime["slow_confirm_count"] = 0
        return EVENT_NORMAL

    if float(slow_distance_m) > threshold_m:
        _sign_runtime["slow_confirm_count"] = 0
        return EVENT_NORMAL

    _sign_runtime["slow_confirm_count"] = int(_sign_runtime.get("slow_confirm_count", 0)) + 1
    if int(_sign_runtime["slow_confirm_count"]) < confirm_frames:
        return EVENT_NORMAL

    _sign_runtime["slow_confirm_count"] = 0
    _sign_runtime["slow_engaged"] = True
    _sign_runtime["slow_loss_streak"] = 0
    _mark_tag_slow_event(now, cooldown_s)
    print(
        f"[Project][Leader] Slow sign within {threshold_m:.2f}m "
        f"(d={float(slow_distance_m):.2f}m) — triggering SLOW",
        flush=True,
    )
    return EVENT_SLOW_SIGN


def _slow_event_with_delay(
    seen_slow: bool,
    cfg: Dict[str, Any],
    now: float,
    cooldown_s: float,
) -> str:
    """Legacy: confirm tag, cruise sign_slow_delay_s, then emit SLOW."""
    confirm_frames = max(1, int(cfg.get("sign_confirm_frames", 3)))
    delay_s = max(0.0, float(cfg.get("sign_slow_delay_s", 2.5)))
    pending_key = "slow_pending_until"
    pending_until = float(_sign_runtime.get(pending_key, 0.0))

    if pending_until > 0.0:
        if not seen_slow:
            _sign_runtime[pending_key] = 0.0
            _sign_runtime["slow_confirm_count"] = 0
            return EVENT_NORMAL
        if now >= pending_until:
            _sign_runtime[pending_key] = 0.0
            _sign_runtime["slow_engaged"] = True
            _mark_tag_slow_event(now, cooldown_s)
            print(
                f"[Project][Leader] Slow sign delay elapsed ({delay_s:.1f}s) — triggering SLOW",
                flush=True,
            )
            return EVENT_SLOW_SIGN
        return EVENT_NORMAL

    if not seen_slow:
        _sign_runtime["slow_confirm_count"] = 0
        return EVENT_NORMAL

    _sign_runtime["slow_confirm_count"] = int(_sign_runtime.get("slow_confirm_count", 0)) + 1
    if int(_sign_runtime["slow_confirm_count"]) < confirm_frames:
        return EVENT_NORMAL

    _sign_runtime["slow_confirm_count"] = 0
    if delay_s <= 0.0:
        _sign_runtime["slow_engaged"] = True
        _mark_tag_slow_event(now, cooldown_s)
        return EVENT_SLOW_SIGN

    _sign_runtime[pending_key] = now + delay_s
    print(
        f"[Project][Leader] Slow sign confirmed — waiting {delay_s:.1f}s before slowing",
        flush=True,
    )
    return EVENT_NORMAL


def _slow_event_from_detections(
    seen_slow: bool,
    slow_distance_m: Optional[float],
    cfg: Dict[str, Any],
    now: float,
    cooldown_s: float,
) -> str:
    mode = str(cfg.get("sign_slow_mode", "distance")).strip().lower()
    if mode == "delay":
        return _slow_event_with_delay(seen_slow, cfg, now, cooldown_s)
    return _slow_event_distance(seen_slow, slow_distance_m, cfg, now, cooldown_s)


def detect_sign_event(
    detections: List[Any],
    frame_shape: Optional[Tuple[int, int]],
    cfg: Dict[str, Any], 
) -> str:
    stop_ids = set(int(x) for x in cfg.get("stop_tag_ids", []))
    slow_ids = set(int(x) for x in cfg.get("slow_tag_ids", []))
    cooldown_s = max(0.0, float(cfg.get("sign_cooldown_s", 2.0)))
    center_roi = float(cfg.get("sign_center_roi", 1.0))
    now = time.time()
    if now < float(_sign_runtime["active_until"]):
        cached = str(_sign_runtime["candidate_event"])
        if cached != EVENT_SLOW_SIGN or not bool(_sign_runtime.get("slow_engaged")):
            return cached

    _seen_stop, stop_distance_m = _sign_stop_from_detections(
        detections=detections,
        stop_ids=stop_ids,
        frame_shape=frame_shape,
        center_roi=center_roi,
    )
    seen_slow, slow_distance_m = _sign_slow_from_detections(
        detections=detections,
        slow_ids=slow_ids,
        frame_shape=frame_shape,
        center_roi=center_roi,
    )

    if now < float(_sign_runtime.get("stop_rearm_until", 0.0)):
        _reset_stop_tracker()
        stop_event = EVENT_NORMAL
    else:
        stop_event = _stop_event_distance(_seen_stop, stop_distance_m, cfg)

    if stop_event == EVENT_STOP_SIGN:
        _reset_slow_tracker()
        _sign_runtime["active_until"] = now + cooldown_s
        _sign_runtime["candidate_event"] = EVENT_NORMAL
        _sign_runtime["candidate_count"] = 0
        return EVENT_STOP_SIGN

    return _slow_event_from_detections(seen_slow, slow_distance_m, cfg, now, cooldown_s)


def get_follower_grid_overlay(frame_bgr, cfg: Dict[str, Any]) -> Optional[GridDetection]:
    """Shared leader tracking cache for MJPEG preview (avoid duplicate inference)."""
    cached = get_cached_leader_tracking()
    if cached is not None:
        return cached
    if frame_bgr is None:
        return get_cached_grid_detection()
    return fetch_leader_tracking(frame_bgr, cfg)


def render_follower_grid_overlay(frame_bgr, cfg: Dict[str, Any]):
    """YOLO boxes on main camera; grid overlay only when detector lost the leader."""
    if frame_bgr is None:
        return frame_bgr
    out = render_leader_camera_overlay(frame_bgr, cfg)
    cached = get_cached_leader_tracking()
    if cached is None and frame_bgr is not None:
        cached = fetch_leader_tracking(frame_bgr, cfg)
    if (
        cached is not None
        and cached.found
        and getattr(cached, "source", None) == "grid"
    ):
        out = draw_grid_overlay(out, cached)
    return out


def enrich_follower_debug_masks(frame_bgr, debug, cfg=None):
    """Lane debug masks only — leader detections are drawn on the main camera."""
    return debug


def get_intersection_turn_params(cfg=None):
    """Timed intersection turn settings from project_config (for web UI)."""
    if cfg is None:
        params = _intersection_turn_from_cfg(load_config())
    else:
        params = _intersection_turn_from_cfg(cfg)
    return {"config_path": _CONFIG_PATH, **params}


def patch_intersection_turn_params(updates):
    """Merge intersection turn timing into project_config.yaml."""
    validated: Dict[str, Any] = {}
    for key in _INTERSECTION_TURN_KEYS:
        if key in (updates or {}):
            validated[key] = float(updates[key])

    for key in _INTERSECTION_TURN_KEYS:
        if key in validated:
            validated[key] = max(0.0, float(validated[key]))
    for key in ("intersection_turn_straight_s", "intersection_turn_left_s", "intersection_turn_right_s"):
        if key in validated:
            validated[key] = max(0.2, float(validated[key]))
    if "intersection_turn_speed" in validated:
        validated["intersection_turn_speed"] = min(
            1.0, max(0.05, float(validated["intersection_turn_speed"])),
        )
    for key in ("intersection_turn_inner_ratio", "intersection_turn_outer_ratio"):
        if key in validated:
            validated[key] = min(1.0, max(0.05, float(validated[key])))

    raw = _merge_project_config_raw(validated)
    params = _intersection_turn_from_cfg(raw)
    _publish_runtime_intersection_turn(params)
    print(
        f"[Project] Intersection turn saved to {_CONFIG_PATH}: "
        f"straight={params['intersection_turn_straight_s']:.2f}s "
        f"left_arc={params['intersection_turn_left_s']:.2f}s "
        f"right_arc={params['intersection_turn_right_s']:.2f}s "
        f"speed={params['intersection_turn_speed']:.2f}",
        flush=True,
    )
    return {"status": "ok", "config_path": _CONFIG_PATH, **params}


def get_spacing_params(cfg=None):
    """Follower convoy spacing settings (for web UI)."""
    c = load_config() if cfg is None else cfg
    return {
        "config_path": _CONFIG_PATH,
        "leader_detector_span_target_px": float(c.get("leader_detector_span_target_px", 38.0)),
        "span_target_px": float(c.get("span_target_px", 7.0)),
        "spacing_kp": float(c.get("spacing_kp", 0.012)),
        "spacing_kd": float(c.get("spacing_kd", 0.022)),
        "follower_catchup_margin": float(c.get("follower_catchup_margin", 0.06)),
        "span_too_close_px": float(c.get("span_too_close_px", 18.0)),
    }


def patch_spacing_params(updates):
    """Merge follower spacing into project_config.yaml (applies on next control frame)."""
    validated: Dict[str, Any] = {}
    float_keys = (
        "leader_detector_span_target_px",
        "span_target_px",
        "spacing_kp",
        "spacing_kd",
        "follower_catchup_margin",
        "span_too_close_px",
    )
    for key in float_keys:
        if key in (updates or {}):
            validated[key] = float(updates[key])

    if "leader_detector_span_target_px" in validated:
        validated["leader_detector_span_target_px"] = max(
            8.0, min(200.0, float(validated["leader_detector_span_target_px"])),
        )
    if "span_target_px" in validated:
        validated["span_target_px"] = max(2.0, float(validated["span_target_px"]))
    if "spacing_kp" in validated:
        validated["spacing_kp"] = min(0.08, max(0.001, float(validated["spacing_kp"])))
    if "spacing_kd" in validated:
        validated["spacing_kd"] = min(0.1, max(0.0, float(validated["spacing_kd"])))
    if "follower_catchup_margin" in validated:
        validated["follower_catchup_margin"] = min(
            0.35, max(0.0, float(validated["follower_catchup_margin"])),
        )
    if "span_too_close_px" in validated:
        validated["span_too_close_px"] = max(4.0, float(validated["span_too_close_px"]))

    raw = _merge_project_config_raw(validated)
    params = get_spacing_params(raw)
    print(
        f"[Project] Spacing saved to {_CONFIG_PATH}: "
        f"det_target={params['leader_detector_span_target_px']:.0f}px "
        f"grid_target={params['span_target_px']:.1f}px",
        flush=True,
    )
    return {"status": "ok", "config_path": _CONFIG_PATH, **params}


def set_lane_agent(lane_agent: LaneServoingAgent) -> None:
    """Use one LaneServoingAgent from the server (same as visual_lane_servoing)."""
    global _lane_agent
    _lane_agent = lane_agent


def set_driving_enabled(enabled: bool) -> None:
    global _driving_enabled
    with _driving_lock:
        was_enabled = _driving_enabled
        _driving_enabled = bool(enabled)
    if was_enabled != bool(enabled):
        reset_sign_detection_state()


def is_driving_enabled() -> bool:
    with _driving_lock:
        return _driving_enabled


def _get_lane_agent() -> LaneServoingAgent:
    global _lane_agent
    if _lane_agent is None:
        _lane_agent = LaneServoingAgent()
    return _lane_agent


def _remember_lane_pwm(lane_agent: LaneServoingAgent, left: float, right: float) -> None:
    lane_agent._last_left = float(left)
    lane_agent._last_right = float(right)


# Leader / follower cruise: steer yellow+white; red for UI + intersections (not steering).
def _project_lane_kwargs(cfg, red_prox=None):
    """Same masks as visual_lane_servoing on straights; trim bottom only on the red line."""
    ignore = float(cfg.get("lane_ignore_bottom_frac", 0.0))
    if red_prox is not None and red_prox.at_line:
        ignore = max(
            ignore,
            float(cfg.get("lane_ignore_bottom_at_intersection_frac", 0.35)),
        )
    return {"ignore_bottom_frac": ignore, "debug_red_mask": True}


def _leader_lane_pwm(lane_agent, frame_bgr, cfg=None, red_prox=None):
    """Yellow+white steer; red detected for debug/follower intersections (not steering)."""
    if frame_bgr is None:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    lane_cfg = cfg if cfg is not None else load_config()
    return lane_agent.compute_commands(frame_rgb, **_project_lane_kwargs(lane_cfg, red_prox))


def _drive_leader_wheels(wheels, lane_agent, frame_bgr, lane_direct, speed_cap, cfg=None, red_prox=None):
    if wheels is None:
        return
    if not is_driving_enabled():
        _safe_stop(wheels)
        return
    pwm = _leader_lane_pwm(lane_agent, frame_bgr, cfg=cfg, red_prox=red_prox)
    if pwm is None:
        return
    left, right = pwm
    _remember_lane_pwm(lane_agent, left, right)
    _apply_lane_wheels(wheels, left, right, speed_cap, lane_direct)


def _apply_lane_wheels(
    wheels,
    left: float,
    right: float,
    speed_cap: float,
    use_direct_pwm: bool,
) -> None:
    """Direct PWM = visual_lane_servoing. Otherwise scale steer to speed_cap."""
    if wheels is None:
        return
    left_f = float(left)
    right_f = float(right)
    if use_direct_pwm:
        wheels.set_wheels_speed(
            min(1.0, max(0.0, left_f)),
            min(1.0, max(0.0, right_f)),
        )
        return
    peak = max(1e-6, max(abs(left_f), abs(right_f)))
    scale = max(0.0, float(speed_cap)) / peak
    wheels.set_wheels_speed(
        min(1.0, left_f * scale),
        min(1.0, right_f * scale),
    )


def _sleep_if_no_frame(frame_bgr) -> None:
    """Match visual_lane_servoing: run per camera frame, brief wait only when no frame."""
    if frame_bgr is None:
        time.sleep(0.01)


def _drive_follower(
    wheels,
    lane_agent,
    frame_bgr,
    commanded_speed,
    target_speed,
    frame_dt,
    cfg,
    wheel_pwm=None,
    *,
    intersection_pwm=False,
    lane_kwargs=None,
) -> float:
    """Ramp to spacing target, then lane steer or intersection arc PWM."""
    commanded_speed = _ramp_toward(
        commanded_speed, target_speed, _speed_ramp_delta(cfg, frame_dt),
    )
    if wheel_pwm is not None:
        left_pwm, right_pwm = wheel_pwm
    elif frame_bgr is not None:
        kwargs = lane_kwargs or {"debug_red_mask": True, "ignore_bottom_frac": 0.0}
        left_pwm, right_pwm = lane_agent.compute_commands(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), **kwargs,
        )
    else:
        left_pwm, right_pwm = commanded_speed, commanded_speed

    if wheels is not None and _follower_wheels_allowed():
        # Cruise/convoy: scale lane steer to spacing speed. Intersection arcs: absolute PWM.
        use_direct = bool(intersection_pwm)
        _apply_lane_wheels(wheels, left_pwm, right_pwm, commanded_speed, use_direct)
        _remember_lane_pwm(lane_agent, left_pwm, right_pwm)
    elif wheels is not None:
        _safe_stop(wheels)
    return commanded_speed


def _follower_spacing_targets(cfg, leader_source):
    """Live spacing targets from project_config (refreshed every control frame)."""
    if leader_source == "detector":
        return (
            float(cfg.get("leader_detector_span_target_px", 38.0)),
            leader_spacing_too_close_for_source(cfg, "detector"),
        )
    return (
        float(cfg.get("span_target_px", 7.0)),
        leader_spacing_too_close_for_source(cfg, "grid"),
    )


def _follower_cruise_target_speed(
    spacing,
    cfg,
    follower_min,
    follower_max,
    cruise_speed,
    slow_speed,
    leader_visible,
    leader_grid_samples,
    last_span_px,
    leader_source=None,
):
    """Spacing speed for normal cruise — conservative at startup, never blind full throttle."""
    if not leader_visible:
        if bool(cfg.get("follower_require_leader", True)):
            return follower_min
        fallback = float(cfg.get("follower_lane_fallback_speed", cruise_speed))
        return max(follower_min, min(follower_max, fallback))

    span_target, span_too_close = _follower_spacing_targets(cfg, leader_source)
    target_speed = spacing.compute_target_speed(
        cfg, follower_min, follower_max,
        span_target=span_target, span_too_close=span_too_close,
    )

    warmup = max(1, int(cfg.get("follower_spacing_warmup_frames", 8)))
    if leader_grid_samples < warmup:
        return min(target_speed, slow_speed)

    catchup = max(0.0, float(cfg.get("follower_catchup_margin", 0.06)))
    target_speed = min(target_speed, cruise_speed + catchup)

    if last_span_px is not None and last_span_px >= span_too_close:
        target_speed = min(
            target_speed,
            float(cfg.get("span_too_close_speed", 0.05)),
        )
    return max(follower_min, min(follower_max, target_speed))


def _follower_intersection_turn_speed(cfg, cruise_speed):
    """Fixed PWM base during timed intersection — not spacing/throttle."""
    turn = float(cfg.get("intersection_turn_speed", cruise_speed))
    return min(1.0, max(0.05, turn))


def run_leader(camera, wheels, leds, stop_event, cfg: Dict[str, Any]) -> None:
    cruise_speed = float(cfg.get("cruise_speed", 0.32))
    slow_speed = float(cfg.get("slow_speed", 0.16))
    stop_hold_s = float(cfg.get("stop_hold_s", 2.0))
    decel_time_s = float(cfg.get("decel_time_s", 1.2))
    decel_steps = int(cfg.get("decel_steps", 10))
    lane_direct = bool(cfg.get("lane_use_direct_pwm", False))
    sign_source = str(cfg.get("sign_source", "both"))
    tags_ok = ensure_apriltag_probe()
    print("[Project][Leader] Control loop: one update per camera frame (like visual_lane_servoing).")
    print(
        f"[Project][Leader] sign_source={sign_source} apriltag={'yes' if tags_ok else 'no'} "
        f"manual={get_manual_convoy_command()}",
        flush=True,
    )
    if lane_direct:
        print("[Project][Leader] Lane PWM: direct (lane_servoing_config.yaml, like visual_lane_servoing).")
    else:
        print("[Project][Leader] Lane PWM: scaled to cruise_speed / slow_speed.")

    last_log = 0.0
    last_drive_ts = time.time()
    state = STATE_CRUISING
    current_speed = cruise_speed
    stop_until = 0.0
    last_event = EVENT_NORMAL
    last_tag_ids: List[int] = []

    while not stop_event.is_set():
        now = time.time()
        frame_dt = max(1e-3, now - last_drive_ts)
        last_drive_ts = now
        _maybe_auto_clear_manual_slow(now)
        frame_bgr, frame_gray = _extract_frame_gray(camera)
        frame_shape = None if frame_bgr is None else frame_bgr.shape[:2]
        red_prox = measure_red_at_line(frame_bgr, cfg)
        if sign_source == "manual" or not is_driving_enabled():
            detections = []
            tag_event = EVENT_NORMAL
        else:
            detections = _detect_apriltags(frame_gray, cfg)
            tag_event = detect_sign_event(detections, frame_shape, cfg)
        last_tag_ids = []
        for det in detections:
            try:
                last_tag_ids.append(int(det.tag_id))
            except Exception:
                pass

        event = _resolve_leader_event(state, tag_event, cfg)
        last_event = event
        if now >= stop_until:
            proposed = next_state(state, event)
            if state == STATE_SLOW and proposed == STATE_CRUISING:
                if get_manual_convoy_command() == MANUAL_SLOW:
                    proposed = STATE_SLOW
                elif tag_event == EVENT_SLOW_SIGN:
                    proposed = STATE_SLOW
                elif current_speed > slow_speed + 0.02:
                    proposed = STATE_SLOW
        else:
            proposed = state
 
        if proposed != state:
            print(f"[Project][Leader] transition {state} -> {proposed} ({event})")
            state = proposed

        if state == STATE_STOPPING:
            _apply_convoy_leds(
                leds,
                state=STATE_STOPPING,
                current_speed=current_speed,
                cruise_speed=cruise_speed,
                slow_speed=slow_speed,
                driving_enabled=is_driving_enabled(),
            )
            if is_driving_enabled():
                smooth_stop(
                    wheels, current_speed, decel_time_s, decel_steps, stop_event, _get_lane_agent()
                )
            else:
                _safe_stop(wheels)
            current_speed = 0.0
            state = STATE_STOPPED
            stop_until = time.time() + stop_hold_s
        elif state == STATE_STOPPED:
            current_speed = 0.0
            _safe_stop(wheels)
            _remember_lane_pwm(_get_lane_agent(), 0.0, 0.0)
            if now >= stop_until and event == EVENT_NORMAL:
                if _sign_runtime.pop("stop_pending_rearm", False):
                    rearm_s = float(cfg.get("sign_stop_rearm_s", 6.0))
                    _sign_runtime["stop_rearm_until"] = now + max(0.0, rearm_s)
                    _reset_stop_tracker()
                    print(
                        f"[Project][Leader] Tag stop hold done; ignore stop tags for {rearm_s:.1f}s",
                        flush=True,
                    )
                if get_manual_convoy_command() == MANUAL_STOPPED:
                    set_manual_convoy_command(MANUAL_CRUISING)
                state = STATE_CRUISING
        elif state == STATE_SLOW:
            speed_target = slow_speed
            current_speed = _ramp_toward(
                current_speed, speed_target, _speed_ramp_delta(cfg, frame_dt)
            )
            _drive_leader_wheels(
                wheels, _get_lane_agent(), frame_bgr, lane_direct=False, speed_cap=slow_speed,
                cfg=cfg, red_prox=red_prox,
            )
        else:
            state = STATE_CRUISING
            speed_target = cruise_speed
            current_speed = _ramp_toward(
                current_speed, speed_target, _speed_ramp_delta(cfg, frame_dt)
            )
            _drive_leader_wheels(
                wheels, _get_lane_agent(), frame_bgr, lane_direct=lane_direct, speed_cap=cruise_speed,
                cfg=cfg, red_prox=red_prox,
            )

        if now - last_log >= 2.0:
            print(
                f"[Project][Leader] state={state} speed={current_speed:.2f} "
                f"event={last_event} tags={last_tag_ids}"
            )
            last_log = now

        _update_convoy_ui_status(
            state=state,
            speed=current_speed,
            event=last_event,
            tag_ids=list(last_tag_ids),
        )

        _apply_convoy_leds(
            leds,
            state=state,
            current_speed=current_speed,
            cruise_speed=cruise_speed,
            slow_speed=slow_speed,
            driving_enabled=is_driving_enabled(),
        )
        _sleep_if_no_frame(frame_bgr)
 
    _update_convoy_ui_status(
        state=STATE_STOPPED,
        speed=0.0,
        event=EVENT_NORMAL,
        tag_ids=[],
    )
    _safe_stop(wheels)
    _convoy_leds_off(leds)
    print("[Project][Leader] Stopped.")
 
 
def _update_intersection_turn_tracker(turn_tracker, leader_det):
    """Intersection memory from the same leader sample used for spacing (no extra inference)."""
    if leader_det is None or not leader_det.found:
        return
    turn_tracker.update(leader_det)


def _intersection_turn_fallback_sample(cfg):
    """Last leader sample for red-line infer — never mix grid when detector is ready."""
    if leader_detector_ready(cfg):
        return get_cached_leader_detector()
    cached = get_cached_grid_detection()
    if cached is not None and cached.found:
        cached.source = "grid"
    return cached


def run_follower(camera, wheels, leds, stop_event, cfg: Dict[str, Any]) -> None:
    grid_hz = max(1.0, float(cfg.get("grid_detect_hz", 10)))
    grid_dt = 1.0 / grid_hz
    cruise_speed = float(cfg.get("cruise_speed", 0.32))
    slow_speed = float(cfg.get("slow_speed", 0.16))
    follower_max = float(cfg.get("follower_max_speed", 0.368))
    follower_min = float(cfg.get("follower_min_speed", 0.0))
    loss_confirm = max(1, int(cfg.get("leader_loss_confirm_frames", 3)))
    red_confirm = max(1, int(cfg.get("intersection_red_confirm_frames", 3)))
    span_target = float(cfg.get("span_target_px", 32.0))
    det_span_target = float(cfg.get("leader_detector_span_target_px", 38.0))
    print("[Project][Follower] Lane control: one update per camera frame (like visual_lane_servoing).")
    print(
        f"[Project][Follower] spacing grid_target={span_target:.0f}px "
        f"detector_target={det_span_target:.0f}px; "
        f"red-line intersection (YOLO truck turns; grid turns only if model unloaded).",
        flush=True,
    )

    spacing = GridSpacingController()
    turn_tracker = LeaderTurnTracker(window=int(cfg.get("intersection_turn_track_window", 20)))
    last_log = 0.0
    last_grid = 0.0
    last_drive_ts = time.time()
    mode = STATE_CRUISING
    commanded_speed = 0.0
    prev_mode = None
    leader_loss_streak = 0
    leader_visible = False
    leader_grid_samples = 0
    last_distance_signal = None
    last_span_px = None
    line_streak = 0
    intersection_phase = None  # None | "TURN"
    intersection_direction = None
    intersection_turn_start = 0.0
    intersection_preamble_end = 0.0
    intersection_arc_end = 0.0
    intersection_end = 0.0
    last_leader_source = None
    last_spacing_target = span_target
    last_leader_seen_ts = 0.0

    while not stop_event.is_set():
        cfg = load_config()
        cruise_speed = float(cfg.get("cruise_speed", 0.32))
        slow_speed = float(cfg.get("slow_speed", 0.16))
        follower_max = float(cfg.get("follower_max_speed", 0.368))
        follower_min = float(cfg.get("follower_min_speed", 0.0))
        now = time.time()
        frame_dt = max(1e-3, now - last_drive_ts)
        last_drive_ts = now
        frame_bgr, _ = _extract_frame_gray(camera)

        red_prox = measure_red_at_line(frame_bgr, cfg)
        turn_tracker.begin_approach_if_needed(red_prox, cfg)

        if now - last_grid >= grid_dt:
            leader_det = fetch_leader_tracking(frame_bgr, cfg) if frame_bgr is not None else None
            if leader_det is not None and leader_det.found:
                leader_loss_streak = 0
                leader_visible = True
                last_leader_seen_ts = now
                last_leader_source = getattr(leader_det, "source", None)
                leader_grid_samples = min(leader_grid_samples + 1, 1000)
                last_distance_signal = leader_det.distance_signal
                _update_intersection_turn_tracker(turn_tracker, leader_det)
                if leader_det.span_px is not None:
                    spacing_span = leader_spacing_span_px(leader_det, cfg)
                    if spacing_span is not None:
                        last_span_px = spacing_span
                        last_spacing_target = leader_spacing_target_px(leader_det, cfg)
                        spacing.observe(
                            spacing_span,
                            now,
                            cfg,
                            commanded_speed,
                            span_target=last_spacing_target,
                            span_too_close=leader_spacing_too_close_px(leader_det, cfg),
                        )
            else:
                leader_loss_streak += 1
                if leader_loss_streak >= loss_confirm:
                    leader_visible = False
            last_grid = now

        if red_prox.at_line:
            line_streak += 1
        else:
            line_streak = 0

        pending_turn = _pop_follower_test_turn()
        if pending_turn is not None:
            intersection_direction = pending_turn
            intersection_phase = "TURN"
            intersection_turn_start = now
            _set_follower_test_turn_active(True)
            (
                intersection_preamble_end,
                intersection_arc_end,
                intersection_end,
                schedule,
            ) = intersection_phase_deadlines(
                intersection_turn_start, intersection_direction, cfg,
            )
            line_streak = 0
            turn_tracker.reset()
            print(
                f"[Project][Follower] Manual test turn: {intersection_direction} "
                f"(no Start required) preamble={schedule['preamble_s']:.1f}s "
                f"arc={schedule['arc_s']:.1f}s tail={schedule['tail_s']:.1f}s",
                flush=True,
            )

        if (
            line_streak >= red_confirm
            and is_driving_enabled()
            and intersection_phase is None
            and turn_tracker.approach_active
        ):
            if not turn_tracker.has_last_grid:
                cached = _intersection_turn_fallback_sample(cfg)
                if cached is not None and cached.found:
                    turn_tracker.update(cached)
            frame_w = float(frame_bgr.shape[1]) if frame_bgr is not None else 640.0
            turn_dbg = turn_tracker.debug_votes(cfg, frame_w=frame_w)
            intersection_direction = turn_tracker.infer(cfg, frame_w=frame_w)
            intersection_phase = "TURN"
            intersection_turn_start = now
            (
                intersection_preamble_end,
                intersection_arc_end,
                intersection_end,
                schedule,
            ) = intersection_phase_deadlines(
                intersection_turn_start, intersection_direction, cfg,
            )
            print(
                f"[Project][Follower] Red line — turn {intersection_direction} "
                f"(last={turn_dbg['last_source']} "
                f"cx={turn_dbg['last_cx']} hdg={turn_dbg['last_heading']} "
                f"aspect={turn_dbg['last_aspect']} offset={turn_dbg['cx_offset_px']:.0f}px "
                f"hdg_vote={turn_dbg['heading_vote']} trend={turn_dbg['cx_trend_vote']}; "
                f"near_px={red_prox.near_px}) "
                f"preamble={schedule['preamble_s']:.1f}s "
                f"arc={schedule['arc_s']:.1f}s tail={schedule['tail_s']:.1f}s",
                flush=True,
            )

        lane_agent = _get_lane_agent()
        wheel_pwm = None
        follow_mode = "cruise"
        intersection_pwm = False

        if intersection_phase == "TURN":
            mode = STATE_INTERSECTION
            (
                intersection_preamble_end,
                intersection_arc_end,
                intersection_end,
                _ix_schedule,
            ) = intersection_phase_deadlines(
                intersection_turn_start, intersection_direction, cfg,
            )
            turn_speed = _follower_intersection_turn_speed(cfg, cruise_speed)
            target_speed = turn_speed
            if intersection_direction == TURN_STRAIGHT:
                wheel_pwm = intersection_wheel_commands(
                    TURN_STRAIGHT, cfg, speed=turn_speed,
                )
                intersection_pwm = True
                follow_mode = "turn_straight"
            elif now < intersection_preamble_end:
                wheel_pwm = intersection_wheel_commands(
                    TURN_STRAIGHT, cfg, speed=turn_speed,
                )
                intersection_pwm = True
                follow_mode = f"turn_preamble_{intersection_direction}"
            elif now < intersection_arc_end:
                wheel_pwm = intersection_wheel_commands(
                    intersection_direction, cfg, speed=turn_speed,
                )
                intersection_pwm = True
                follow_mode = f"turn_{intersection_direction}"
            elif now < intersection_end:
                wheel_pwm = intersection_wheel_commands(
                    TURN_STRAIGHT, cfg, speed=turn_speed,
                )
                intersection_pwm = True
                follow_mode = "turn_tail"
            if now >= intersection_end:
                intersection_phase = None
                intersection_direction = None
                line_streak = 0
                turn_tracker.reset()
                _set_follower_test_turn_active(False)
                mode = STATE_CRUISING
                wheel_pwm = None
                intersection_pwm = False
                if leader_visible:
                    follow_mode = "convoy"
                else:
                    follow_mode = "lane"
                print("[Project][Follower] Intersection done — resume cruise", flush=True)
        else:
            mode = STATE_CRUISING
            if leader_visible:
                follow_mode = "convoy"
            else:
                follow_mode = "lane"
            target_speed = _follower_cruise_target_speed(
                spacing, cfg, follower_min, follower_max, cruise_speed, slow_speed,
                leader_visible, leader_grid_samples, last_span_px,
                leader_source=last_leader_source,
            )

        commanded_speed = _drive_follower(
            wheels, lane_agent, frame_bgr, commanded_speed, target_speed, frame_dt, cfg, wheel_pwm,
            intersection_pwm=intersection_pwm,
            lane_kwargs=_project_lane_kwargs(cfg, red_prox) if wheel_pwm is None else None,
        )

        if mode != prev_mode:
            print(f"[Project][Follower] transition {prev_mode} -> {mode}")
            prev_mode = mode

        _update_convoy_ui_status(
            state=mode,
            speed=commanded_speed,
            event=EVENT_NORMAL if leader_visible else EVENT_LEADER_LOST,
            dist_signal=last_distance_signal if leader_visible else None,
            leader_visible=leader_visible,
            follow_mode=follow_mode,
            intersection_phase=intersection_phase or "-",
            intersection_turn=(
                intersection_direction
                if intersection_phase == "TURN"
                else "-"
            ),
            red_near_px=red_prox.near_px,
            red_at_line=red_prox.at_line,
            tag_ids=[],
            test_turn_active=_follower_wheels_allowed() and not is_driving_enabled(),
        )

        if now - last_log >= 2.0:
            dist_str = f"{last_distance_signal:.3f}" if last_distance_signal is not None else "-"
            span_str = f"{last_span_px:.1f}" if last_span_px is not None else "-"
            spacing_tgt, _ = _follower_spacing_targets(
                cfg, last_leader_source or "detector",
            )
            src_str = last_leader_source or "-"
            ix_extra = ""
            if intersection_phase == "TURN":
                _sched = intersection_turn_schedule(intersection_direction, cfg)
                ix_extra = (
                    f" ix_arc={_sched['arc_s']:.2f}s"
                    f" ix_left={float(cfg.get('intersection_turn_left_s', 0)):.2f}"
                    f" ix_right={float(cfg.get('intersection_turn_right_s', 0)):.2f}"
                )
            print(
                f"[Project][Follower] mode={mode} follow={follow_mode} "
                f"target={target_speed:.2f} cmd={commanded_speed:.2f} "
                f"span={span_str}/{spacing_tgt:.0f} src={src_str} "
                f"span_dot={spacing.span_dot:.1f} "
                f"v_leader={spacing.v_leader_est:.2f} dist={dist_str} "
                f"leader={leader_visible} red={red_prox.near_px} at_line={red_prox.at_line} "
                f"intersection={intersection_phase} turn={intersection_direction}{ix_extra}"
            )
            last_log = now

        _apply_convoy_leds(
            leds,
            state=mode,
            current_speed=commanded_speed,
            cruise_speed=cruise_speed,
            slow_speed=slow_speed,
            driving_enabled=is_driving_enabled(),
        )
        _sleep_if_no_frame(frame_bgr)

    _update_convoy_ui_status(
        state=STATE_STOPPED,
        speed=0.0,
        event=EVENT_LEADER_LOST,
        dist_signal=None,
        leader_visible=False,
        tag_ids=[],
    )
    _safe_stop(wheels)
    _set_follower_test_turn_active(False)
    _convoy_leds_off(leds)
    print("[Project][Follower] Stopped.")
 
 
def main(camera, wheels, leds, stop_event):
    # Role is selected from project_config.yaml so one codebase serves both bots.
    cfg = load_config()
    role = cfg.get("role", "leader")
    print(f"[Project] Loaded config from {_CONFIG_PATH}")
    print(f"[Project] Role: {role}")
 
    if role == "follower":
        run_follower(camera, wheels, leds, stop_event, cfg)
    else:
        run_leader(camera, wheels, leds, stop_event, cfg)