"""Leader YOLO detector status card for project follower UI."""

LEADER_DETECTOR_CARD = '''
            <div class="card" id="leaderDetectorCard" style="display:none">
                <div class="card-header">Leader Detector (follower)</div>
                <p style="font-size:12px;color:var(--text-muted);margin:0 0 12px">
                    YOLO truck detection for convoy follow and intersection turns.
                    Rear dot grid is fallback only when the model misses.
                </p>
                <div id="detectorStatusLine" style="font-size:13px;margin-bottom:10px">
                    <span class="ok">Checking model…</span>
                </div>
                <div style="font-size:12px;color:var(--text-muted);display:grid;gap:6px">
                    <div>Backend: <span id="detectorBackend" class="info-value">-</span></div>
                    <div>Leader seen: <span id="detectorLastSeen" class="info-value">-</span></div>
                    <div>Source: <span id="detectorSource" class="info-value">-</span></div>
                    <div>Confidence: <span id="detectorScore" class="info-value">-</span></div>
                </div>
            </div>
'''

LEADER_DETECTOR_JS = '''
    function renderDetectorStatus(det) {
        const card = document.getElementById('leaderDetectorCard');
        const line = document.getElementById('detectorStatusLine');
        if (!card || !line || !det) return;

        const backendEl = document.getElementById('detectorBackend');
        const seenEl = document.getElementById('detectorLastSeen');
        const srcEl = document.getElementById('detectorSource');
        const scoreEl = document.getElementById('detectorScore');

        let html = '';
        if (det.trt_building) {
            const sec = det.trt_build_elapsed_s != null ? det.trt_build_elapsed_s : 0;
            html = '<span style="color:#f39c12">Compiling TensorRT engine… ' + sec + 's (first run ~1 min)</span>';
        } else if (det.model_loaded) {
            html = '<span class="ok">Model loaded — tracking leader truck</span>';
        } else if (det.load_error) {
            html = '<span style="color:#e74c3c">Model not ready: ' + det.load_error + '</span>';
        } else {
            html = '<span style="color:var(--text-muted)">Loading model…</span>';
        }
        line.innerHTML = html;

        if (backendEl) {
            backendEl.textContent = det.backend || (det.model_loaded ? 'ready' : '-');
        }
        if (seenEl) {
            seenEl.textContent = det.last_found ? 'yes' : 'no';
        }
        if (srcEl) {
            srcEl.textContent = det.last_source || '-';
        }
        if (scoreEl) {
            scoreEl.textContent = det.score != null ? Number(det.score).toFixed(2) : '-';
        }
    }
'''
