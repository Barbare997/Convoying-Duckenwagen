import sys
import os
import signal
import threading
import argparse
import queue
import socket

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from flask import Flask, Response, jsonify, render_template_string
import numpy as np
import cv2

from duckiebot.camera_driver import CameraDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from duckiebot.led_driver import LEDDriver
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs

try:
    from servers.templates.project import get_template
except ModuleNotFoundError:
    get_template = None

try:
    from servers.visual_lane_servoing.visualization import create_lane_visualization
except ModuleNotFoundError:
    create_lane_visualization = None
from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent

import tasks.project.packages.agent as agent

app        = Flask(__name__)
camera     = None
wheels     = None
leds       = None
stop_event = threading.Event()
_preview_lane_agent = None
_frame_queue = queue.Queue(maxsize=1)


class QueuedCamera:
    """Convoy agent reads frames from the video stream (one hardware camera reader)."""

    def __init__(self, frame_queue: queue.Queue):
        self._q = frame_queue

    def start(self):
        return None

    def stop(self):
        return None

    def read(self):
        try:
            frame = self._q.get(timeout=0.25)
            return True, frame
        except queue.Empty:
            return False, None


def _visualize(frame_bgr):
    if frame_bgr is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for camera...", (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        return blank

    try:
        _frame_queue.put_nowait(frame_bgr.copy())
    except queue.Full:
        try:
            _frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _frame_queue.put_nowait(frame_bgr.copy())
        except queue.Full:
            pass

    if _preview_lane_agent is None or create_lane_visualization is None:
        return frame_bgr

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pwm_left, pwm_right = _preview_lane_agent.compute_commands(frame_rgb)
    return create_lane_visualization(
        frame_bgr, _preview_lane_agent.last_debug_info, pwm_left, pwm_right
    )


generate_frames = make_frame_generator(lambda: camera, _visualize, quality=50, rgb=False)


def _fallback_index_html(role: str) -> str:
    host = socket.gethostname()
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Convoy — {host}</title>
<style>body{{font-family:sans-serif;background:#1a1d23;color:#e6edf3;margin:12px}}
.grid{{display:grid;grid-template-columns:1fr 320px;gap:12px;max-width:1400px;margin:0 auto}}
img.stream{{width:100%;border-radius:6px}} .card{{background:#13161a;border:1px solid #30363d;padding:12px;border-radius:6px}}
.row{{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #30363d;font-size:13px}}</style></head>
<body><h2>Convoy Project — {host}</h2><p>{role} — real hardware</p>
<div class="grid"><div><img src="/video" class="stream"></div>
<div class="card"><div id="statusTable">Loading…</div></div></div>
<script>
function refresh(){{fetch('/status').then(r=>r.json()).then(d=>{{
  const rows=[['role',d.role],['state',d.state],['speed',d.speed!=null?Number(d.speed).toFixed(2):'-'],
    ['event',d.event||'-'],['tags',(d.tag_ids&&d.tag_ids.length)?d.tag_ids.join(', '):'-']];
  document.getElementById('statusTable').innerHTML=rows.map(([k,v])=>
    '<div class="row"><span>'+k+'</span><span>'+v+'</span></div>').join('');
}}).catch(()=>{{}});}}
refresh(); setInterval(refresh,400);
</script></body></html>"""


@app.route('/')
def index():
    cfg = agent.load_config()
    role = str(cfg.get('role', 'leader'))
    if get_template is not None:
        html = get_template(
            title=f"Convoy Project — {socket.gethostname()}",
            subtitle=f"{role} — real hardware",
        )
        return render_template_string(html)
    return _fallback_index_html(role)


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/shutdown')
def shutdown():
    shutdown_cleanup(wheels, camera, stop_event)
    return jsonify({'status': 'ok'})


@app.route('/convoy/status')
def convoy_status():
    return jsonify(agent.get_leader_status())


@app.route('/status')
def status():
    cfg = agent.load_config()
    payload = agent.get_leader_status()
    payload["role"] = cfg.get("role", "leader")
    return jsonify(payload)


def main():
    global camera, wheels, leds, stop_event, _preview_lane_agent

    ap = argparse.ArgumentParser(description='Project Server — Real Hardware')
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT SERVER — REAL HARDWARE')
    print('=' * 60)

    print('\n[1/4] Initializing LED driver...')
    try:
        leds = LEDDriver()
        leds.all_off()
        print('  LEDs: ok')
    except Exception as e:
        print(f'  LEDs: not available ({e})')
        leds = None

    print('\n[2/4] Initializing wheels driver...')
    wheels = DaguWheelsDriver(WheelPWMConfiguration(), WheelPWMConfiguration())
    print('  Wheels: ok')

    print('\n[3/4] Initializing camera driver...')
    camera = CameraDriver()
    camera.start()
    print('  Camera: ok')

    print('\n[4/5] Lane preview (stream masks, same as visual_lane_servoing)...')
    _preview_lane_agent = LaneServoingAgent()
    print(f'  p_gain={_preview_lane_agent.p_gain}, base_speed={_preview_lane_agent.base_speed}')

    print('\n[5/5] Starting convoy agent...')
    stop_event.clear()
    queued_cam = QueuedCamera(_frame_queue)
    threading.Thread(
        target=agent.main,
        args=(queued_cam, wheels, leds, stop_event),
        daemon=True,
        name='AgentThread',
    ).start()
    print('  agent.main() running')

    def _shutdown(signum, frame):
        print('\nShutting down...')
        if leds:
            try:
                leds.all_off()
                leds.release()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    web_port = find_available_port(args.port)
    print(f'\nWeb UI:  http://localhost:{web_port}/')
    print(f'Video:   http://localhost:{web_port}/video')
    print('Press Ctrl+C to stop\n')

    try:
        app.run(host='0.0.0.0', port=web_port, debug=False, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if leds:
            try:
                leds.all_off()
                leds.release()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())
