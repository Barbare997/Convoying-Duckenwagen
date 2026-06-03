import os
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
 
import cv2
import requests
import yaml
from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
 
 
_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "project_config.yaml")
)
 
# States
STATE_CRUISING = "CRUISING"
STATE_SLOW = "SLOW"
STATE_STOPPING = "STOPPING"
STATE_STOPPED = "STOPPED"

# Events
EVENT_NORMAL = "EVENT_NORMAL"
EVENT_SLOW_SIGN = "EVENT_SLOW_SIGN"
EVENT_STOP_SIGN = "EVENT_STOP_SIGN"
EVENT_TIMEOUT = "EVENT_TIMEOUT"

# Shared leader status contract (used by both teammates):
# {
#   "state": "STOPPED" | "CRUISING" | "SLOW",
#   "speed": <float>,            # commanded forward speed in [0, 1]
#   "ts": <float>                # Unix timestamp from time.time()
# }
#
# Follower-side rule:
# if current_time - payload["ts"] > leader_timeout_s -> treat as stale and stop.
 
_status_lock = threading.Lock()
_leader_status: Dict[str, Any] = {
    "state": "STOPPED",
    "speed": 0.0,
    "ts": float(time.time()),
}

_apriltag_detector = None
_apriltag_init_attempted = False
_sign_runtime = {
    "candidate_event": EVENT_NORMAL,
    "candidate_count": 0,
    "active_until": 0.0,
}
_lane_agent: Optional[LaneServoingAgent] = None
_detection_agent = None
_detection_init_attempted = False

# YOLO class id for leader Duckiebot (trained as "truck" in object_detection task).
LEADER_YOLO_CLASS_ID = 1


def load_config() -> Dict[str, Any]:
    # Load only the fields needed by the skeleton. Missing file -> safe defaults.
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
 
    return {
        "role": str(cfg.get("role", "leader")).strip().lower(),
        "loop_hz": float(cfg.get("loop_hz", 20)),
        "leader_host": str(cfg.get("leader_host", "127.0.0.1")).strip(),
        "leader_port": int(cfg.get("leader_port", 5055)),
        "cruise_speed": float(cfg.get("cruise_speed", 0.2)),
        "slow_speed": float(cfg.get("slow_speed", 0.12)),
        "follower_max_speed": float(cfg.get("follower_max_speed", 0.2)),
        "follower_min_speed": float(cfg.get("follower_min_speed", 0.0)),
        "distance_target": float(cfg.get("distance_target", 0.06)),
        "distance_kp": float(cfg.get("distance_kp", 0.6)),
        "status_publish_hz": float(cfg.get("status_publish_hz", 10)),
        "status_poll_hz": float(cfg.get("status_poll_hz", 10)),
        "request_timeout_s": float(cfg.get("request_timeout_s", 0.2)),
        "leader_timeout_s": float(cfg.get("leader_timeout_s", 0.4)),
        "stop_hold_s": float(cfg.get("stop_hold_s", 2.0)),
        "decel_time_s": float(cfg.get("decel_time_s", 0.8)),
        "decel_steps": int(cfg.get("decel_steps", 8)),
        "stop_tag_ids": [int(x) for x in cfg.get("stop_tag_ids", [])],
        "slow_tag_ids": [int(x) for x in cfg.get("slow_tag_ids", [])],
        "sign_confirm_frames": int(cfg.get("sign_confirm_frames", 3)),
        "sign_cooldown_s": float(cfg.get("sign_cooldown_s", 2.0)),
        "sign_center_roi": float(cfg.get("sign_center_roi", 1.0)),
        "leader_tag_ids": [int(x) for x in cfg.get("leader_tag_ids", [])],
        "leader_class_id": int(cfg.get("leader_class_id", LEADER_YOLO_CLASS_ID)),
        "leader_yolo_enabled": bool(cfg.get("leader_yolo_enabled", True)),
        "leader_center_roi": float(cfg.get("leader_center_roi", 0.65)),
        "leader_min_bbox_area": int(cfg.get("leader_min_bbox_area", 400)),
        "leader_min_y2_frac": float(cfg.get("leader_min_y2_frac", 0.25)),
    }
 
 
def build_status_payload(state: str, speed: float) -> Dict[str, Any]:
    """Create the canonical leader status payload shared by leader/follower code."""
    return {
        "state": str(state).upper(),
        "speed": float(speed),
        "ts": float(time.time()),
    }
 
 
def set_leader_status(payload: Dict[str, Any]) -> None:
    with _status_lock:
        _leader_status.update(payload)
 
 
def get_leader_status() -> Dict[str, Any]:
    with _status_lock:
        return dict(_leader_status)
 
 
def _safe_stop(wheels) -> None:
    # Centralized hard-stop helper used in both roles and on exit.
    try:
        if wheels is not None:
            wheels.set_wheels_speed(0.0, 0.0)
    except Exception as e:
        print(f"[Project] Wheels stop failed: {e}")
 
 
def smooth_stop(
    wheels,
    current_speed: float,
    decel_time_s: float,
    decel_steps: int,
    stop_event,
) -> None:
    """Ramp wheel commands down to zero over configured time/steps."""
    if wheels is None:
        return

    speed0 = max(0.0, float(current_speed))
    steps = max(1, int(decel_steps))
    step_dt = max(0.0, float(decel_time_s)) / steps

    try:
        for i in range(steps - 1, -1, -1):
            if stop_event.is_set():
                break
            s = speed0 * (i / steps)
            wheels.set_wheels_speed(s, s)
            if step_dt > 0:
                time.sleep(step_dt)
    except Exception as e:
        print(f"[Project] smooth_stop failed: {e}")
    finally:
        _safe_stop(wheels)


def next_state(current_state: str, event: str) -> str:
    if event == EVENT_TIMEOUT:
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


def _get_apriltag_detector():
    global _apriltag_detector, _apriltag_init_attempted
    if _apriltag_init_attempted:
        return _apriltag_detector

    _apriltag_init_attempted = True
    try:
        import importlib

        detector_cls = importlib.import_module("pupil_apriltags").Detector
        _apriltag_detector = detector_cls(families="tag36h11")
        print("[Project][AprilTag] Detector initialized (pupil_apriltags).")
    except Exception as e:
        _apriltag_detector = None
        print(f"[Project][AprilTag] Detector unavailable ({e}). Using EVENT_NORMAL fallback.")
    return _apriltag_detector


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


def _detect_apriltags(frame_gray) -> List[Any]:
    detector = _get_apriltag_detector()
    if detector is None or frame_gray is None:
        return []
    try:
        return detector.detect(frame_gray)
    except Exception as e:
        print(f"[Project][AprilTag] detect failed: {e}")
        return []


def _classify_event_from_detections(
    detections: List[Any],
    stop_ids: Set[int],
    slow_ids: Set[int],
    frame_shape: Optional[Tuple[int, int]],
    center_roi: float,
) -> str:
    if not detections:
        return EVENT_NORMAL

    cx_min, cx_max = -1.0, 2.0
    if frame_shape is not None:
        h, w = frame_shape
        roi = max(0.2, min(1.0, float(center_roi)))
        margin = (1.0 - roi) * 0.5
        cx_min = margin * w
        cx_max = (1.0 - margin) * w

    seen_stop = False
    seen_slow = False
    for det in detections:
        try:
            tag_id = int(det.tag_id)
            c = det.center
            cx = float(c[0]) if c is not None else None
        except Exception:
            continue

        if cx is not None and not (cx_min <= cx <= cx_max):
            continue
        if tag_id in stop_ids:
            seen_stop = True
        elif tag_id in slow_ids:
            seen_slow = True

    if seen_stop:
        return EVENT_STOP_SIGN
    if seen_slow:
        return EVENT_SLOW_SIGN
    return EVENT_NORMAL


def detect_sign_event(
    detections: List[Any],
    frame_shape: Optional[Tuple[int, int]],
    cfg: Dict[str, Any],
) -> str:
    stop_ids = set(int(x) for x in cfg.get("stop_tag_ids", []))
    slow_ids = set(int(x) for x in cfg.get("slow_tag_ids", []))
    confirm_frames = max(1, int(cfg.get("sign_confirm_frames", 3)))
    cooldown_s = max(0.0, float(cfg.get("sign_cooldown_s", 2.0)))
    center_roi = float(cfg.get("sign_center_roi", 1.0))

    raw_event = _classify_event_from_detections(
        detections=detections,
        stop_ids=stop_ids,
        slow_ids=slow_ids,
        frame_shape=frame_shape,
        center_roi=center_roi,
    )

    now = time.time()
    # Hold active sign event during cooldown to reduce flicker near signs.
    if now < float(_sign_runtime["active_until"]):
        return str(_sign_runtime["candidate_event"])

    if raw_event == EVENT_NORMAL:
        _sign_runtime["candidate_event"] = EVENT_NORMAL
        _sign_runtime["candidate_count"] = 0
        return EVENT_NORMAL

    if _sign_runtime["candidate_event"] == raw_event:
        _sign_runtime["candidate_count"] = int(_sign_runtime["candidate_count"]) + 1
    else:
        _sign_runtime["candidate_event"] = raw_event
        _sign_runtime["candidate_count"] = 1

    if int(_sign_runtime["candidate_count"]) >= confirm_frames:
        _sign_runtime["active_until"] = now + cooldown_s
        return raw_event
    return EVENT_NORMAL


def _estimate_tag_size_ratio(det: Any, frame_shape: Optional[Tuple[int, int]]) -> Optional[float]:
    if frame_shape is None:
        return None
    h, w = frame_shape
    try:
        corners = det.corners
        if corners is None or len(corners) != 4:
            return None
        p0, p1, p2, p3 = corners
        e1 = ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5
        e2 = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        side = 0.5 * (e1 + e2)
        return float(side / max(1.0, min(w, h)))
    except Exception:
        return None


def estimate_leader_distance_signal(
    detections: List[Any],
    frame_shape: Optional[Tuple[int, int]],
    cfg: Dict[str, Any],
) -> Tuple[Optional[float], Optional[int]]:
    leader_ids = set(int(x) for x in cfg.get("leader_tag_ids", []))
    if not detections or not leader_ids:
        return None, None

    best_ratio = None
    best_id = None
    for det in detections:
        try:
            tag_id = int(det.tag_id)
        except Exception:
            continue
        if tag_id not in leader_ids:
            continue
        ratio = _estimate_tag_size_ratio(det, frame_shape)
        if ratio is None:
            continue
        # Larger tag in image => closer; pick largest for strongest signal.
        if best_ratio is None or ratio > best_ratio:
            best_ratio = ratio
            best_id = tag_id
    return best_ratio, best_id


def _get_detection_agent():
    """Lazy-load YOLO agent for follower leader spacing (optional onnxruntime)."""
    global _detection_agent, _detection_init_attempted
    if _detection_init_attempted:
        return _detection_agent

    _detection_init_attempted = True
    try:
        from tasks.object_detection.packages.agent import ObjectDetectionAgent

        _detection_agent = ObjectDetectionAgent()
        if _detection_agent.model_loaded:
            print(
                f"[Project][YOLO] Leader spacing model ready "
                f"(img_size={_detection_agent.img_size})."
            )
        else:
            print(
                f"[Project][YOLO] Leader spacing unavailable: "
                f"{_detection_agent.load_error}"
            )
            _detection_agent = None
    except Exception as e:
        _detection_agent = None
        print(f"[Project][YOLO] Leader spacing unavailable ({e}).")
    return _detection_agent


def _leader_truck_in_roi(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
    cfg: Dict[str, Any],
) -> bool:
    h, w = frame_shape
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    if area < int(cfg.get("leader_min_bbox_area", 400)):
        return False
    if y2 / max(1.0, float(h)) < float(cfg.get("leader_min_y2_frac", 0.25)):
        return False

    roi = max(0.2, min(1.0, float(cfg.get("leader_center_roi", 0.65))))
    margin = (1.0 - roi) * 0.5
    cx = 0.5 * (x1 + x2)
    return margin * w <= cx <= (1.0 - margin) * w


def _leader_truck_distance_signal(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
) -> float:
    """Monocular proximity: larger + lower in frame => closer (same scale idea as AprilTag ratio)."""
    h, w = frame_shape
    x1, y1, x2, y2 = bbox
    area_norm = ((x2 - x1) * (y2 - y1)) / max(1.0, float(w * h))
    y2_frac = y2 / max(1.0, float(h))
    return 0.65 * area_norm + 0.35 * y2_frac


def estimate_leader_distance_from_yolo(
    frame_bgr,
    cfg: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Pick the best in-lane truck box as the leader; return (distance_signal, confidence).
    Used when leader_tag_ids is empty (no AprilTag on leader back).
    """
    if frame_bgr is None:
        return None, None

    det_agent = _get_detection_agent()
    if det_agent is None or not det_agent.model_loaded:
        return None, None

    leader_cls = int(cfg.get("leader_class_id", LEADER_YOLO_CLASS_ID))
    frame_shape = frame_bgr.shape[:2]
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = det_agent.detect(frame_rgb)
    except Exception as e:
        print(f"[Project][YOLO] detect failed: {e}")
        return None, None

    if not detections:
        return None, None

    best_signal = None
    best_score = None
    for bbox, conf, cls_id in detections:
        if int(cls_id) != leader_cls:
            continue
        if not _leader_truck_in_roi(bbox, frame_shape, cfg):
            continue
        signal = _leader_truck_distance_signal(bbox, frame_shape)
        if best_signal is None or signal > best_signal:
            best_signal = signal
            best_score = float(conf)

    return best_signal, best_score


def estimate_follower_distance_signal(
    frame_bgr,
    frame_gray,
    frame_shape: Optional[Tuple[int, int]],
    cfg: Dict[str, Any],
) -> Tuple[Optional[float], Optional[Any]]:
    """
    AprilTag on leader if configured; otherwise YOLO truck box (hybrid convoy spacing).
    """
    leader_tag_ids = set(int(x) for x in cfg.get("leader_tag_ids", []))
    if leader_tag_ids:
        detections = _detect_apriltags(frame_gray)
        signal, tag_id = estimate_leader_distance_signal(detections, frame_shape, cfg)
        return signal, tag_id

    if not cfg.get("leader_yolo_enabled", True):
        return None, None

    signal, conf = estimate_leader_distance_from_yolo(frame_bgr, cfg)
    return signal, conf


def _get_lane_agent() -> LaneServoingAgent:
    global _lane_agent
    if _lane_agent is None:
        _lane_agent = LaneServoingAgent()
    return _lane_agent


def run_leader(camera, wheels, leds, stop_event, cfg: Dict[str, Any]) -> None:
    loop_hz = max(1.0, float(cfg.get("loop_hz", 20)))
    dt = 1.0 / loop_hz
    status_hz = max(1.0, float(cfg.get("status_publish_hz", 10)))
    status_dt = 1.0 / status_hz
    cruise_speed = float(cfg.get("cruise_speed", 0.2))
    slow_speed = float(cfg.get("slow_speed", 0.12))
    stop_hold_s = float(cfg.get("stop_hold_s", 2.0))
    decel_time_s = float(cfg.get("decel_time_s", 0.8))
    decel_steps = int(cfg.get("decel_steps", 8))
    print(f"[Project][Leader] FSM loop started at {loop_hz:.1f} Hz.")
 
    last_log = 0.0
    last_status_pub = 0.0
    state = STATE_CRUISING
    current_speed = cruise_speed
    stop_until = 0.0
    last_event = EVENT_NORMAL
    last_tag_ids: List[int] = []

    while not stop_event.is_set():
        now = time.time()
        frame_bgr, frame_gray = _extract_frame_gray(camera)
        frame_shape = None if frame_bgr is None else frame_bgr.shape[:2]
        detections = _detect_apriltags(frame_gray)
        last_tag_ids = []
        for det in detections:
            try:
                last_tag_ids.append(int(det.tag_id))
            except Exception:
                pass

        event = detect_sign_event(detections, frame_shape, cfg)
        last_event = event
        if now >= stop_until:
            proposed = next_state(state, event)
        else:
            proposed = state
 
        if proposed != state:
            print(f"[Project][Leader] transition {state} -> {proposed} ({event})")
            state = proposed

        if state == STATE_STOPPING:
            smooth_stop(wheels, current_speed, decel_time_s, decel_steps, stop_event)
            current_speed = 0.0
            state = STATE_STOPPED
            stop_until = time.time() + stop_hold_s
        elif state == STATE_STOPPED:
            current_speed = 0.0
            _safe_stop(wheels)
            if now >= stop_until and event == EVENT_NORMAL:
                state = STATE_CRUISING
        elif state == STATE_SLOW:
            current_speed = max(0.0, slow_speed)
            if wheels is not None and frame_bgr is not None:
                lane_agent = _get_lane_agent()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                left, right = lane_agent.compute_commands(frame_rgb)
                scale = current_speed / max(1e-6, max(left, right))
                wheels.set_wheels_speed(min(1.0, left * scale), min(1.0, right * scale))
            elif wheels is not None:
                wheels.set_wheels_speed(current_speed, current_speed)
        else:
            state = STATE_CRUISING
            current_speed = max(0.0, cruise_speed)
            if wheels is not None and frame_bgr is not None:
                lane_agent = _get_lane_agent()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                left, right = lane_agent.compute_commands(frame_rgb)
                scale = current_speed / max(1e-6, max(left, right))
                wheels.set_wheels_speed(min(1.0, left * scale), min(1.0, right * scale))
            elif wheels is not None:
                wheels.set_wheels_speed(current_speed, current_speed)

        if now - last_status_pub >= status_dt:
            payload = build_status_payload(state, current_speed)
            payload.update(
                {
                    "event": last_event,
                    "tag_ids": last_tag_ids,
                }
            )
            set_leader_status(payload)
            last_status_pub = now
 
        if now - last_log >= 2.0:
            print(
                f"[Project][Leader] state={state} speed={current_speed:.2f} "
                f"event={last_event} tags={last_tag_ids}"
            )
            last_log = now
        time.sleep(dt)
 
    set_leader_status(build_status_payload(STATE_STOPPED, 0.0))
    _safe_stop(wheels)
    print("[Project][Leader] Stopped.")
 
 
def run_follower(camera, wheels, leds, stop_event, cfg: Dict[str, Any]) -> None:
    loop_hz = max(1.0, float(cfg.get("loop_hz", 20)))
    dt = 1.0 / loop_hz
    poll_hz = max(1.0, float(cfg.get("status_poll_hz", 10)))
    poll_dt = 1.0 / poll_hz
    request_timeout_s = float(cfg.get("request_timeout_s", 0.2))
    leader_timeout_s = float(cfg.get("leader_timeout_s", 0.4))
    leader_host = str(cfg.get("leader_host", "127.0.0.1")).strip()
    leader_port = int(cfg.get("leader_port", 5055))
    cruise_speed = float(cfg.get("cruise_speed", 0.2))
    slow_speed = float(cfg.get("slow_speed", 0.12))
    follower_max_speed = float(cfg.get("follower_max_speed", 0.2))
    follower_min_speed = float(cfg.get("follower_min_speed", 0.0))
    distance_target = float(cfg.get("distance_target", 0.06))
    distance_kp = float(cfg.get("distance_kp", 0.6))
    decel_time_s = float(cfg.get("decel_time_s", 0.8))
    decel_steps = int(cfg.get("decel_steps", 8))
    status_url = f"http://{leader_host}:{leader_port}/convoy/status"
    print(f"[Project][Follower] Polling loop started at {loop_hz:.1f} Hz.")
 
    last_log = 0.0
    last_poll = 0.0
    latest = build_status_payload(STATE_STOPPED, 0.0)
    mode = EVENT_TIMEOUT
    target_speed = 0.0
    commanded_speed = 0.0
    prev_mode = None
    last_distance_meta = None
    leader_tag_ids = set(int(x) for x in cfg.get("leader_tag_ids", []))
    use_yolo_spacing = not leader_tag_ids and bool(cfg.get("leader_yolo_enabled", True))
    if use_yolo_spacing:
        print("[Project][Follower] Leader spacing: HTTP + YOLO truck (no leader AprilTag).")
    elif leader_tag_ids:
        print(f"[Project][Follower] Leader spacing: HTTP + AprilTag ids={sorted(leader_tag_ids)}.")
    else:
        print("[Project][Follower] Leader spacing: HTTP only.")

    while not stop_event.is_set():
        now = time.time()
        frame_bgr, frame_gray = _extract_frame_gray(camera)
        frame_shape = None if frame_bgr is None else frame_bgr.shape[:2]

        distance_signal = None
        distance_meta = None
        distance_signal, distance_meta = estimate_follower_distance_signal(
            frame_bgr, frame_gray, frame_shape, cfg
        )
        if distance_meta is not None:
            last_distance_meta = distance_meta

        if now - last_poll >= poll_dt:
            try:
                resp = requests.get(status_url, timeout=request_timeout_s)
                if resp.ok:
                    data = resp.json()
                    latest = {
                        "state": str(data.get("state", "STOPPED")).upper(),
                        "speed": float(data.get("speed", 0.0)),
                        "ts": float(data.get("ts", 0.0)),
                    }
            except Exception:
                pass
            last_poll = now
 
        is_stale = (now - float(latest.get("ts", 0.0))) > leader_timeout_s
        state = str(latest.get("state", "STOPPED")).upper()
        leader_speed = max(0.0, float(latest.get("speed", 0.0)))
        if is_stale:
            mode, target_speed = EVENT_TIMEOUT, 0.0
        elif state == STATE_STOPPED:
            mode, target_speed = STATE_STOPPED, 0.0
        elif state == STATE_SLOW:
            mode, target_speed = STATE_SLOW, min(slow_speed, leader_speed, follower_max_speed)
        else:
            mode, target_speed = STATE_CRUISING, min(cruise_speed, leader_speed, follower_max_speed)

        if mode in (STATE_CRUISING, STATE_SLOW) and distance_signal is not None:
            # Smaller signal -> leader farther -> speed up; larger -> closer -> slow down.
            distance_error = distance_target - float(distance_signal)
            target_speed += distance_kp * distance_error
            target_speed = max(follower_min_speed, min(follower_max_speed, target_speed))

        if mode == EVENT_TIMEOUT:
            commanded_speed = 0.0
            _safe_stop(wheels)
        elif mode == STATE_STOPPED:
            if commanded_speed > 0.0:
                smooth_stop(wheels, commanded_speed, decel_time_s, decel_steps, stop_event)
            commanded_speed = 0.0
        else:
            commanded_speed = target_speed
            if wheels is not None and frame_bgr is not None:
                lane_agent = _get_lane_agent()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                left, right = lane_agent.compute_commands(frame_rgb)
                scale = commanded_speed / max(1e-6, max(left, right))
                wheels.set_wheels_speed(min(1.0, left * scale), min(1.0, right * scale))
            elif wheels is not None:
                wheels.set_wheels_speed(commanded_speed, commanded_speed)

        if mode != prev_mode:
            print(f"[Project][Follower] transition {prev_mode} -> {mode}")
            prev_mode = mode

        if now - last_log >= 2.0:
            age = now - float(latest.get("ts", 0.0))
            print(
                f"[Project][Follower] mode={mode} target_speed={target_speed:.2f} cmd={commanded_speed:.2f} "
                f"leader_state={state} leader_speed={leader_speed:.2f} age={age:.2f}s "
                f"dist_signal={distance_signal} dist_meta={last_distance_meta}"
            )
            last_log = now
        time.sleep(dt)
 
    _safe_stop(wheels)
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