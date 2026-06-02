import os
import time
from typing import Any, Dict

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
    }


def build_status_payload(state: str, speed: float) -> Dict[str, Any]:
    """Create the canonical leader status payload shared by leader/follower code."""
    return {
        "state": str(state).upper(),
        "speed": float(speed),
        "ts": float(time.time()),
    }


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
    print(f"[Project][Leader] Safe idle loop started at {loop_hz:.1f} Hz.")

    last_log = 0.0
    while not stop_event.is_set():
        # Skeleton mode: leader never drives until convoy logic is implemented.
        _safe_stop(wheels)
        now = time.time()
        if now - last_log >= 2.0:
            print("[Project][Leader] state=STOPPED (skeleton mode)")
            last_log = now
        time.sleep(dt)

    _safe_stop(wheels)
    print("[Project][Leader] Stopped.")


def run_follower(camera, wheels, leds, stop_event, cfg: Dict[str, Any]) -> None:
    loop_hz = max(1.0, float(cfg.get("loop_hz", 20)))
    dt = 1.0 / loop_hz
    print(f"[Project][Follower] Safe idle loop started at {loop_hz:.1f} Hz.")

    last_log = 0.0
    while not stop_event.is_set():
        # Skeleton mode: follower also stays stopped; only structure is in place.
        _safe_stop(wheels)
        now = time.time()
        if now - last_log >= 2.0:
            print("[Project][Follower] waiting for leader state (skeleton mode)")
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