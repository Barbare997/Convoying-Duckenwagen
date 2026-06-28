"""Load YOLO leader detector for project follower (background thread)."""

import threading


def start_leader_detector_background() -> None:
    """Non-blocking ObjectDetectionAgent load (TensorRT may compile ~1 min on Jetson)."""

    def _load():
        try:
            from tasks.object_detection.packages.agent import ObjectDetectionAgent
            from tasks.project.packages.leader_detector import set_detector_agent

            print("[Project] Loading leader detector (YOLO truck)...", flush=True)
            det = ObjectDetectionAgent()
            set_detector_agent(det)
            if det.model_loaded:
                backend = getattr(det, "_backend", "?")
                print(
                    f"[Project] Leader detector ready ({backend}, {det.img_size}px).",
                    flush=True,
                )
            elif getattr(det, "trt_building", False):
                print(
                    "[Project] Leader detector compiling TensorRT in background...",
                    flush=True,
                )
            else:
                print(
                    f"[Project] Leader detector unavailable: {det.load_error}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[Project] Leader detector init failed: {exc}", flush=True)

    threading.Thread(
        target=_load,
        name="LeaderDetectorLoad",
        daemon=True,
    ).start()
