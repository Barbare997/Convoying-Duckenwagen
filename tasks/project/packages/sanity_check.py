import os
import sys
import threading
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tasks.project.packages import agent


class DummyWheels:
    def __init__(self):
        self.commands = []

    def set_wheels_speed(self, left: float, right: float):
        self.commands.append((left, right))


def _run_and_stop(loop_fn, timeout_s: float = 0.2):
    stop_event = threading.Event()
    wheels = DummyWheels()
    cfg = {}

    t = threading.Thread(
        target=loop_fn,
        args=(None, wheels, None, stop_event, cfg),
        daemon=True,
    )
    t.start()
    time.sleep(timeout_s)
    stop_event.set()
    t.join(timeout=1.0)

    assert not t.is_alive(), f"{loop_fn.__name__} did not exit after stop_event"
    assert wheels.commands, f"{loop_fn.__name__} never sent wheel command"
    assert wheels.commands[-1] == (0.0, 0.0), f"{loop_fn.__name__} final command was not stop"


def test_next_state_transitions():
    assert agent.next_state(agent.STATE_CRUISING, agent.EVENT_SLOW_SIGN) == agent.STATE_SLOW
    assert agent.next_state(agent.STATE_SLOW, agent.EVENT_NORMAL) == agent.STATE_CRUISING
    assert agent.next_state(agent.STATE_CRUISING, agent.EVENT_STOP_SIGN) == agent.STATE_STOPPING
    assert agent.next_state(agent.STATE_CRUISING, agent.EVENT_TIMEOUT) == agent.STATE_STOPPING
    print("OK: FSM transitions")


def test_smooth_stop():
    wheels = DummyWheels()
    stop_event = threading.Event()
    agent.smooth_stop(wheels, current_speed=0.4, decel_time_s=0.05, decel_steps=4, stop_event=stop_event)
    assert wheels.commands, "smooth_stop produced no wheel commands"
    assert wheels.commands[-1] == (0.0, 0.0), "smooth_stop did not end at full stop"
    print("OK: smooth_stop reaches zero")


def test_config_loads():
    cfg = agent.load_config()
    assert isinstance(cfg, dict), "Config must be a dictionary"
    assert "role" in cfg, "Config missing role"
    assert "leader_yolo_enabled" in cfg, "Config missing leader_yolo_enabled"
    print("OK: config loads")


def test_yolo_leader_distance_signal():
    cfg = {
        "role": "follower",
        "leader_yolo_enabled": True,
        "leader_class_id": 1,
        "leader_center_roi": 1.0,
        "leader_min_bbox_area": 100,
        "leader_min_y2_frac": 0.1,
    }
    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Farther truck: smaller, higher in image
    far = ((200, 80, 280, 200), 0.9, 1)
    # Closer truck: larger, lower in image
    near = ((220, 200, 420, 420), 0.9, 1)

    class FakeDet:
        model_loaded = True
        img_size = 640

        def detect(self, _rgb):
            return [far, near]

    agent._detection_agent = FakeDet()
    agent._detection_init_attempted = True
    signal, conf = agent.estimate_leader_distance_from_yolo(frame, cfg)
    assert signal is not None and conf == 0.9
    near_signal = agent._leader_truck_distance_signal(near[0], frame.shape[:2])
    assert abs(signal - near_signal) < 1e-6
    print("OK: YOLO leader distance picks nearest truck")


def test_role_dispatch():
    called = {"leader": 0, "follower": 0}
    original_load = agent.load_config
    original_leader = agent.run_leader
    original_follower = agent.run_follower

    def fake_leader(*args, **kwargs):
        called["leader"] += 1

    def fake_follower(*args, **kwargs):
        called["follower"] += 1

    try:
        agent.run_leader = fake_leader
        agent.run_follower = fake_follower

        agent.load_config = lambda: {"role": "leader"}
        agent.main(None, None, None, threading.Event())
        assert called["leader"] == 1 and called["follower"] == 0, "Leader dispatch failed"

        called["leader"] = 0
        called["follower"] = 0
        agent.load_config = lambda: {"role": "follower"}
        agent.main(None, None, None, threading.Event())
        assert called["follower"] == 1 and called["leader"] == 0, "Follower dispatch failed"
        print("OK: role dispatch works")
    finally:
        agent.load_config = original_load
        agent.run_leader = original_leader
        agent.run_follower = original_follower


def test_slow_sign_delay():
    agent._sign_runtime.update(
        {
            "candidate_event": agent.EVENT_NORMAL,
            "candidate_count": 0,
            "active_until": 0.0,
            "slow_confirm_count": 0,
            "slow_engaged": False,
            "slow_loss_streak": 0,
            "slow_pending_until": 0.0,
            "stop_visible_streak": 0,
            "stop_armed": False,
            "stop_loss_streak": 0,
            "stop_rearm_until": 0.0,
            "stop_tag_latch_resume": False,
        }
    )
    cfg = {
        "stop_tag_ids": [20],
        "slow_tag_ids": [39],
        "sign_confirm_frames": 2,
        "sign_cooldown_s": 0.0,
        "sign_center_roi": 1.0,
        "sign_stop_on_loss": True,
        "sign_stop_seen_min_frames": 99,
        "sign_stop_loss_confirm_frames": 2,
        "sign_slow_mode": "delay",
        "sign_slow_delay_s": 2.5,
    }
    slow_det = agent._TagDetection(39, (320.0, 240.0))
    shape = (480, 640)

    for _ in range(2):
        assert agent.detect_sign_event([slow_det], shape, cfg) == agent.EVENT_NORMAL

    assert agent.detect_sign_event([slow_det], shape, cfg) == agent.EVENT_NORMAL
    assert float(agent._sign_runtime.get("slow_pending_until", 0.0)) > time.time()

    agent._sign_runtime["slow_pending_until"] = time.time() - 0.01
    assert agent.detect_sign_event([slow_det], shape, cfg) == agent.EVENT_SLOW_SIGN
    assert agent._sign_runtime["candidate_event"] == agent.EVENT_SLOW_SIGN
    print("OK: slow sign waits before triggering")


def test_tag_slow_distance_engages():
    agent.reset_sign_detection_state()
    cfg = {
        "stop_tag_ids": [20],
        "slow_tag_ids": [39],
        "sign_confirm_frames": 2,
        "sign_cooldown_s": 2.0,
        "sign_center_roi": 1.0,
        "sign_stop_on_loss": True,
        "sign_stop_seen_min_frames": 99,
        "sign_stop_loss_confirm_frames": 2,
        "sign_slow_mode": "distance",
        "sign_slow_distance_m": 0.40,
        "sign_slow_loss_confirm_frames": 2,
    }
    far = agent._TagDetection(39, (320.0, 240.0), 0.55)
    near = agent._TagDetection(39, (320.0, 240.0), 0.35)
    shape = (480, 640)

    assert agent.detect_sign_event([far], shape, cfg) == agent.EVENT_NORMAL
    assert agent.detect_sign_event([near], shape, cfg) == agent.EVENT_NORMAL
    assert agent.detect_sign_event([near], shape, cfg) == agent.EVENT_SLOW_SIGN
    assert agent._sign_runtime.get("slow_engaged") is True
    # Tag briefly lost: stay in slow until loss confirm frames elapse.
    assert agent.detect_sign_event([], shape, cfg) == agent.EVENT_SLOW_SIGN
    assert agent.detect_sign_event([], shape, cfg) == agent.EVENT_NORMAL
    assert agent._sign_runtime.get("slow_engaged") is False
    print("OK: slow sign engages by distance and releases after tag loss")


def test_tag_slow_candidate_not_stale_normal():
    agent.reset_sign_detection_state()
    cfg = {
        "stop_tag_ids": [20],
        "slow_tag_ids": [39],
        "sign_confirm_frames": 1,
        "sign_cooldown_s": 2.0,
        "sign_center_roi": 1.0,
        "sign_stop_on_loss": True,
        "sign_stop_seen_min_frames": 99,
        "sign_stop_loss_confirm_frames": 2,
        "sign_slow_mode": "distance",
        "sign_slow_distance_m": 0.40,
        "sign_slow_loss_confirm_frames": 99,
    }
    near = agent._TagDetection(39, (320.0, 240.0), 0.30)
    shape = (480, 640)
    assert agent.detect_sign_event([near], shape, cfg) == agent.EVENT_SLOW_SIGN
    assert agent.detect_sign_event([], shape, cfg) == agent.EVENT_SLOW_SIGN
    assert agent._sign_runtime["candidate_event"] == agent.EVENT_SLOW_SIGN
    print("OK: engaged slow keeps EVENT_SLOW_SIGN during cooldown")


def test_stop_sign_on_loss():
    agent._sign_runtime.update(
        {
            "candidate_event": agent.EVENT_NORMAL,
            "candidate_count": 0,
            "active_until": 0.0,
            "stop_visible_streak": 0,
            "stop_armed": False,
            "stop_loss_streak": 0,
            "stop_rearm_until": 0.0,
            "stop_tag_latch_resume": False,
        }
    )
    cfg = {
        "stop_tag_ids": [20],
        "slow_tag_ids": [39],
        "sign_confirm_frames": 3,
        "sign_cooldown_s": 0.0,
        "sign_center_roi": 1.0,
        "sign_stop_on_loss": True,
        "sign_stop_seen_min_frames": 3,
        "sign_stop_loss_confirm_frames": 2,
        "sign_stop_rearm_s": 1.0,
    }
    stop_det = agent._TagDetection(20, (320.0, 240.0))
    shape = (480, 640)

    for _ in range(3):
        ev = agent.detect_sign_event([stop_det], shape, cfg)
        assert ev == agent.EVENT_NORMAL, "Stop tag visible should not stop yet"

    ev = agent.detect_sign_event([], shape, cfg)
    assert ev == agent.EVENT_NORMAL, "One loss frame should not stop yet"

    ev = agent.detect_sign_event([], shape, cfg)
    assert ev == agent.EVENT_STOP_SIGN, "Stop should trigger after tag leaves view"
    assert agent._sign_runtime.get("stop_tag_latch_resume") is True
    print("OK: stop sign triggers on tag loss")


def test_sign_state_cleared_on_pause():
    agent._sign_runtime["slow_engaged"] = True
    agent._sign_runtime["slow_confirm_count"] = 5
    agent._sign_runtime["stop_armed"] = True
    agent.set_driving_enabled(True)
    agent.set_driving_enabled(False)
    assert agent._sign_runtime.get("slow_engaged") is False
    assert int(agent._sign_runtime.get("slow_confirm_count", 0)) == 0
    assert agent._sign_runtime.get("stop_armed") is False
    print("OK: sign detection state cleared on pause/stop")


def test_follower_visual_target_speed():
    cfg = {"follower_spacing_mode": "visual"}
    assert agent._follower_spacing_mode(cfg) == agent.FOLLOWER_SPACING_VISUAL
    assert agent._follower_uses_http_convoy(cfg) is False
    no_truck = agent._follower_visual_target_speed(
        cfg,
        cruise_speed=0.4,
        follower_max_speed=0.4,
        follower_min_speed=0.0,
        distance_signal=None,
        distance_target=0.05,
        distance_kp=0.6,
    )
    assert abs(no_truck - 0.4) < 1e-6
    with_truck = agent._follower_visual_target_speed(
        cfg,
        cruise_speed=0.4,
        follower_max_speed=0.4,
        follower_min_speed=0.0,
        distance_signal=0.12,
        distance_target=0.05,
        distance_kp=0.6,
    )
    assert with_truck < no_truck
    print("OK: follower visual mode speed (lane without truck)")


def test_follower_http_link_ui():
    cfg = {"leader_timeout_s": 2.0, "leader_fallback_max_s": 3.0, "leader_fallback_enabled": False}
    assert agent._follower_http_link_phase(agent.STATE_CRUISING) == agent.HTTP_LINK_NORMAL
    assert agent._follower_http_link_phase(agent.FALLBACK_LANE) == agent.HTTP_LINK_FALLBACK
    assert agent._follower_http_link_phase(agent.EVENT_TIMEOUT) == agent.HTTP_LINK_TIMEOUT
    assert agent._follower_http_link_phase(agent.STATE_CRUISING, http_latched=True) == agent.HTTP_LINK_TIMEOUT
    agent._publish_follower_status(
        mode=agent.FALLBACK_LANE,
        target_speed=0.12,
        commanded_speed=0.10,
        leader_state=agent.STATE_CRUISING,
        leader_speed=0.40,
        status_age=2.5,
        is_stale=True,
        distance_signal=0.04,
        cfg={**cfg, "leader_fallback_enabled": True},
    )
    st = agent.get_leader_status()
    assert st.get("follower_mode") == agent.FALLBACK_LANE
    assert st.get("http_link") == agent.HTTP_LINK_FALLBACK
    assert "2.0" in str(st.get("http_link_label", ""))
    print("OK: follower HTTP link status for UI")


def test_loops_exit_cleanly():
    _run_and_stop(agent.run_leader)
    _run_and_stop(agent.run_follower)
    print("OK: loops exit cleanly with fake stop_event")


if __name__ == "__main__":
    test_next_state_transitions()
    test_smooth_stop()
    test_config_loads()
    test_yolo_leader_distance_signal()
    test_slow_sign_delay()
    test_tag_slow_distance_engages()
    test_tag_slow_candidate_not_stale_normal()
    test_stop_sign_on_loss()
    test_sign_state_cleared_on_pause()
    test_role_dispatch()
    test_follower_visual_target_speed()
    test_follower_http_link_ui()
    test_loops_exit_cleanly()
    print("All sanity checks passed.")
