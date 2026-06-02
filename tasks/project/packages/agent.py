import os
import threading
import time
from typing import Any, Dict
 
import requests
import yaml
 
 
_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "project_config.yaml")
)
 
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
        "status_publish_hz": float(cfg.get("status_publish_hz", 10)),
        "status_poll_hz": float(cfg.get("status_poll_hz", 10)),
        "request_timeout_s": float(cfg.get("request_timeout_s", 0.2)),
        "leader_timeout_s": float(cfg.get("leader_timeout_s", 0.4)),
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
 
 
def run_leader(camera, wheels, leds, stop_event, cfg: Dict[str, Any]) -> None:
    loop_hz = max(1.0, float(cfg.get("loop_hz", 20)))
    dt = 1.0 / loop_hz
    status_hz = max(1.0, float(cfg.get("status_publish_hz", 10)))
    status_dt = 1.0 / status_hz
    cruise_speed = float(cfg.get("cruise_speed", 0.2))
    slow_speed = float(cfg.get("slow_speed", 0.12))
    print(f"[Project][Leader] Safe idle loop started at {loop_hz:.1f} Hz.")
 
    last_log = 0.0
    last_status_pub = 0.0
    start_t = time.time()
    while not stop_event.is_set():
        # Skeleton mode: leader never drives until convoy logic is implemented.
        _safe_stop(wheels)
        now = time.time()
        elapsed = (now - start_t) % 11.0
        if elapsed < 3.0:
            state, speed = "STOPPED", 0.0
        elif elapsed < 8.0:
            state, speed = "CRUISING", cruise_speed
        else:
            state, speed = "SLOW", slow_speed
 
        if now - last_status_pub >= status_dt:
            set_leader_status(build_status_payload(state, speed))
            last_status_pub = now
 
        if now - last_log >= 2.0:
            print(f"[Project][Leader] state={state} speed={speed:.2f} (status publishing)")
            last_log = now
        time.sleep(dt)
 
    set_leader_status(build_status_payload("STOPPED", 0.0))
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
    status_url = f"http://{leader_host}:{leader_port}/convoy/status"
    print(f"[Project][Follower] Safe idle loop started at {loop_hz:.1f} Hz.")
 
    last_log = 0.0
    last_poll = 0.0
    latest = build_status_payload("STOPPED", 0.0)
    mode = "TIMEOUT"
    target_speed = 0.0
    while not stop_event.is_set():
        # Skeleton mode: follower also stays stopped; only structure is in place.
        now = time.time()
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
        if is_stale:
            mode, target_speed = "TIMEOUT", 0.0
        elif state == "STOPPED":
            mode, target_speed = "STOPPED", 0.0
        elif state == "SLOW":
            mode, target_speed = "SLOW", min(slow_speed, follower_max_speed)
        else:
            mode, target_speed = "CRUISING", min(cruise_speed, follower_max_speed)
 
        _safe_stop(wheels)
        if now - last_log >= 2.0:
            age = now - float(latest.get("ts", 0.0))
            print(
                f"[Project][Follower] mode={mode} target_speed={target_speed:.2f} "
                f"leader_state={state} age={age:.2f}s"
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