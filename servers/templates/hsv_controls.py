"""Shared HSV calibration UI (HTML/CSS/JS) for lane-following tasks."""

HSV_EXTRA_CSS = '''
.info-value { color: var(--accent-blue); font-family: monospace; font-size: 13px; font-weight: 600; }
.hsv-section-title { font-size: 13px; font-weight: 600; color: var(--text-secondary); margin: 12px 0 8px; text-transform: uppercase; letter-spacing: 0.5px; }
.hsv-section-title.yellow { color: #f1c40f; }
.hsv-section-title.white  { color: #ecf0f1; }
.video-stack { flex-direction: column; align-items: stretch; }
.video-stack .stream { flex: 1; min-height: 0; }
.stream-pickable { cursor: crosshair; }
.pick-hint {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 8px;
    text-align: center;
}
.pick-result {
    margin-top: 12px;
    padding: 10px;
    background: var(--bg-sidebar);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    font-size: 12px;
    display: none;
}
.pick-result.visible { display: block; }
.pick-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid var(--border-color);
}
.pick-row:last-child { border-bottom: none; }
.pick-actions { display: flex; gap: 8px; margin-top: 10px; }
.pick-actions button { flex: 1; font-size: 12px; padding: 8px; }
'''

HSV_VIDEO_HINT = '''
            <p class="pick-hint">Click the <strong>Camera</strong> panel (top-left of the stream) to read HSV at that pixel.</p>
'''

HSV_CARD_HTML = '''
            <div class="card">
                <div class="card-header">HSV Color Calibration</div>

                <div class="hsv-section-title yellow">Yellow Line (left / dashed)</div>

                <div class="slider-group">
                    <div class="slider-label"><span>Hue Low</span><span style="color:var(--text-muted)">0-179</span></div>
                    <div class="slider-controls">
                        <input type="range" id="yLowH" min="0" max="179" value="20" class="slider">
                        <input type="number" id="yLowH-input" min="0" max="179" value="20" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Hue High</span><span style="color:var(--text-muted)">0-179</span></div>
                    <div class="slider-controls">
                        <input type="range" id="yHighH" min="0" max="179" value="40" class="slider">
                        <input type="number" id="yHighH-input" min="0" max="179" value="40" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Saturation Low</span><span style="color:var(--text-muted)">0-255</span></div>
                    <div class="slider-controls">
                        <input type="range" id="yLowS" min="0" max="255" value="80" class="slider">
                        <input type="number" id="yLowS-input" min="0" max="255" value="80" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Saturation High</span><span style="color:var(--text-muted)">0-255</span></div>
                    <div class="slider-controls">
                        <input type="range" id="yHighS" min="0" max="255" value="255" class="slider">
                        <input type="number" id="yHighS-input" min="0" max="255" value="255" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Value Low</span><span style="color:var(--text-muted)">0-255</span></div>
                    <div class="slider-controls">
                        <input type="range" id="yLowV" min="0" max="255" value="100" class="slider">
                        <input type="number" id="yLowV-input" min="0" max="255" value="100" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Value High</span><span style="color:var(--text-muted)">0-255</span></div>
                    <div class="slider-controls">
                        <input type="range" id="yHighV" min="0" max="255" value="255" class="slider">
                        <input type="number" id="yHighV-input" min="0" max="255" value="255" class="input-box">
                    </div>
                </div>

                <div class="hsv-section-title white" style="margin-top:20px">White Line (right / solid)</div>

                <div class="slider-group">
                    <div class="slider-label"><span>Hue Low</span><span style="color:var(--text-muted)">0-179</span></div>
                    <div class="slider-controls">
                        <input type="range" id="wLowH" min="0" max="179" value="0" class="slider">
                        <input type="number" id="wLowH-input" min="0" max="179" value="0" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Hue High</span><span style="color:var(--text-muted)">0-179</span></div>
                    <div class="slider-controls">
                        <input type="range" id="wHighH" min="0" max="179" value="179" class="slider">
                        <input type="number" id="wHighH-input" min="0" max="179" value="179" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Saturation Low</span><span style="color:var(--text-muted)">0-255</span></div>
                    <div class="slider-controls">
                        <input type="range" id="wLowS" min="0" max="255" value="0" class="slider">
                        <input type="number" id="wLowS-input" min="0" max="255" value="0" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Saturation High</span><span style="color:var(--text-muted)">0-255</span></div>
                    <div class="slider-controls">
                        <input type="range" id="wHighS" min="0" max="255" value="40" class="slider">
                        <input type="number" id="wHighS-input" min="0" max="255" value="40" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Value Low</span><span style="color:var(--text-muted)">0-255</span></div>
                    <div class="slider-controls">
                        <input type="range" id="wLowV" min="0" max="255" value="180" class="slider">
                        <input type="number" id="wLowV-input" min="0" max="255" value="180" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Value High</span><span style="color:var(--text-muted)">0-255</span></div>
                    <div class="slider-controls">
                        <input type="range" id="wHighV" min="0" max="255" value="255" class="slider">
                        <input type="number" id="wHighV-input" min="0" max="255" value="255" class="input-box">
                    </div>
                </div>

                <div id="pick-result" class="pick-result">
                    <div class="pick-row"><span>Pixel</span><span id="pick-pixel" class="info-value">-</span></div>
                    <div class="pick-row"><span>RGB</span><span id="pick-rgb" class="info-value">-</span></div>
                    <div class="pick-row"><span>HSV</span><span id="pick-hsv" class="info-value">-</span></div>
                    <div class="pick-row"><span>Guess</span><span id="pick-guess" class="info-value">-</span></div>
                    <div class="pick-row"><span>Suggested yellow</span><span id="pick-yellow" class="info-value">-</span></div>
                    <div class="pick-row"><span>Suggested white</span><span id="pick-white" class="info-value">-</span></div>
                    <div class="pick-actions">
                        <button type="button" class="button" style="background:#b8860b" onclick="applyPickedColor('yellow')">Apply as Yellow</button>
                        <button type="button" class="button" style="background:#5c6b7a" onclick="applyPickedColor('white')">Apply as White</button>
                    </div>
                </div>

                <div id="hsv-status" class="status"></div>
            </div>
'''

HSV_EXTRA_JS = '''
    fetch('/get_hsv')
        .then(r => r.json())
        .then(d => {
            setSliderValue('yLowH',  d.yellow_lower_h);
            setSliderValue('yHighH', d.yellow_upper_h);
            setSliderValue('yLowS',  d.yellow_lower_s);
            setSliderValue('yHighS', d.yellow_upper_s);
            setSliderValue('yLowV',  d.yellow_lower_v);
            setSliderValue('yHighV', d.yellow_upper_v);
            setSliderValue('wLowH',  d.white_lower_h);
            setSliderValue('wHighH', d.white_upper_h);
            setSliderValue('wLowS',  d.white_lower_s);
            setSliderValue('wHighS', d.white_upper_s);
            setSliderValue('wLowV',  d.white_lower_v);
            setSliderValue('wHighV', d.white_upper_v);
        });

    const hsvKeys = {
        'yLowH':  'yellow_lower_h', 'yHighH': 'yellow_upper_h',
        'yLowS':  'yellow_lower_s', 'yHighS': 'yellow_upper_s',
        'yLowV':  'yellow_lower_v', 'yHighV': 'yellow_upper_v',
        'wLowH':  'white_lower_h',  'wHighH': 'white_upper_h',
        'wLowS':  'white_lower_s',  'wHighS': 'white_upper_s',
        'wLowV':  'white_lower_v',  'wHighV': 'white_upper_v',
    };

    Object.entries(hsvKeys).forEach(([sliderId, key]) => {
        syncSliderInput(sliderId, () => {
            const payload = {};
            payload[key] = parseInt(document.getElementById(sliderId).value);
            postJSON('/update_hsv', payload)
                .then(() => showStatus('hsv-status', 'HSV Updated!', 'success'));
        });
    });

    let lastPick = null;

    function streamClickToNatural(img, clientX, clientY) {
        const rect = img.getBoundingClientRect();
        const naturalW = img.naturalWidth || 1;
        const naturalH = img.naturalHeight || 1;
        const scale = Math.min(rect.width / naturalW, rect.height / naturalH);
        const renderedW = naturalW * scale;
        const renderedH = naturalH * scale;
        const offsetX = (rect.width - renderedW) / 2;
        const offsetY = (rect.height - renderedH) / 2;
        const x = (clientX - rect.left - offsetX) / scale;
        const y = (clientY - rect.top - offsetY) / scale;
        if (x < 0 || y < 0 || x >= naturalW || y >= naturalH) {
            return null;
        }
        return { stream_x: x, stream_y: y };
    }

    function formatBounds(prefix, b) {
        return `H ${b.lower_h}-${b.upper_h}, S ${b.lower_s}-${b.upper_s}, V ${b.lower_v}-${b.upper_v}`;
    }

    function showPickResult(data) {
        lastPick = data;
        const box = document.getElementById('pick-result');
        box.classList.add('visible');
        document.getElementById('pick-pixel').textContent = `(${data.pixel.x}, ${data.pixel.y})`;
        document.getElementById('pick-rgb').textContent =
            `R${data.rgb.r} G${data.rgb.g} B${data.rgb.b}`;
        document.getElementById('pick-hsv').textContent =
            `H${data.hsv.h} S${data.hsv.s} V${data.hsv.v}`;
        document.getElementById('pick-guess').textContent = data.line_guess;
        document.getElementById('pick-yellow').textContent =
            formatBounds('y', data.suggested_yellow);
        document.getElementById('pick-white').textContent =
            formatBounds('w', data.suggested_white);
    }

    function applyPickedColor(which) {
        if (!lastPick) {
            showStatus('hsv-status', 'Click the camera panel first.', 'error');
            return;
        }
        const b = which === 'yellow' ? lastPick.suggested_yellow : lastPick.suggested_white;
        const prefix = which === 'yellow' ? 'y' : 'w';
        setSliderValue(prefix + 'LowH',  b.lower_h);
        setSliderValue(prefix + 'HighH', b.upper_h);
        setSliderValue(prefix + 'LowS',  b.lower_s);
        setSliderValue(prefix + 'HighS', b.upper_s);
        setSliderValue(prefix + 'LowV',  b.lower_v);
        setSliderValue(prefix + 'HighV', b.upper_v);

        const payload = {};
        if (which === 'yellow') {
            payload.yellow_lower_h = b.lower_h;
            payload.yellow_upper_h = b.upper_h;
            payload.yellow_lower_s = b.lower_s;
            payload.yellow_upper_s = b.upper_s;
            payload.yellow_lower_v = b.lower_v;
            payload.yellow_upper_v = b.upper_v;
        } else {
            payload.white_lower_h = b.lower_h;
            payload.white_upper_h = b.upper_h;
            payload.white_lower_s = b.lower_s;
            payload.white_upper_s = b.upper_s;
            payload.white_lower_v = b.lower_v;
            payload.white_upper_v = b.upper_v;
        }
        postJSON('/update_hsv', payload)
            .then(() => showStatus('hsv-status', which + ' HSV applied from click!', 'success'))
            .catch(() => showStatus('hsv-status', 'HSV apply failed.', 'error'));
    }

    const videoStream = document.getElementById('videoStream');
    if (videoStream) {
        videoStream.addEventListener('click', function(ev) {
            const coords = streamClickToNatural(this, ev.clientX, ev.clientY);
            if (!coords) {
                showStatus('hsv-status', 'Click inside the video image.', 'error');
                return;
            }
            postJSON('/sample_pixel', coords)
                .then(data => {
                    if (!data.ok) {
                        showStatus('hsv-status', data.error || 'Sample failed.', 'error');
                        return;
                    }
                    showPickResult(data);
                    showStatus('hsv-status', 'Color sampled — see HSV box below.', 'success');
                })
                .catch(() => showStatus('hsv-status', 'Sample request failed.', 'error'));
        });
    }
'''
