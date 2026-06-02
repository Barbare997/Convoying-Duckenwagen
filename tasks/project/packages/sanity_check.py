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
    cfg = {"loop_hz": 20}

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


def test_config_loads():
    cfg = agent.load_config()
    assert isinstance(cfg, dict), "Config must be a dictionary"
    assert "role" in cfg, "Config missing role"
    assert "loop_hz" in cfg, "Config missing loop_hz"
    print("OK: config loads")


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

        agent.load_config = lambda: {"role": "leader", "loop_hz": 20}
        agent.main(None, None, None, threading.Event())
        assert called["leader"] == 1 and called["follower"] == 0, "Leader dispatch failed"

        called["leader"] = 0
        called["follower"] = 0
        agent.load_config = lambda: {"role": "follower", "loop_hz": 20}
        agent.main(None, None, None, threading.Event())
        assert called["follower"] == 1 and called["leader"] == 0, "Follower dispatch failed"
        print("OK: role dispatch works")
    finally:
        agent.load_config = original_load
        agent.run_leader = original_leader
        agent.run_follower = original_follower


def test_loops_exit_cleanly():
    _run_and_stop(agent.run_leader)
    _run_and_stop(agent.run_follower)
    print("OK: loops exit cleanly with fake stop_event")


if __name__ == "__main__":
    test_config_loads()
    test_role_dispatch()
    test_loops_exit_cleanly()
    print("All sanity checks passed.")
