"""Leader-blue HSV sliders for project convoy UI (follower tuning)."""

LEADER_BLUE_HSV_CARD = '''
            <div class="card" id="leaderBlueHsvCard">
                <div class="card-header">Leader Blue HSV (follower)</div>
                <p style="font-size:12px;color:var(--text-muted);margin:0 0 12px">
                    Matches the blue mask panel on the stream. Upper S/V are fixed at 255.
                    Saved to <code>config/project_config.yaml</code>; mask updates live on save.
                </p>
                <div class="hsv-color-col blue" style="border-color:rgba(52,152,219,0.45);padding:10px;margin-bottom:8px">
                <div class="hsv-section-title blue" style="color:#3498db;margin-top:0">Duckie body (leader tracking)</div>

                <div class="slider-group">
                    <div class="slider-label"><span>Hue Low</span><span style="color:var(--text-muted)">0-179</span></div>
                    <div class="slider-controls">
                        <input type="range" id="bLowH" min="0" max="179" value="100" class="slider">
                        <input type="number" id="bLowH-input" min="0" max="179" value="100" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Hue High</span><span style="color:var(--text-muted)">0-179</span></div>
                    <div class="slider-controls">
                        <input type="range" id="bHighH" min="0" max="179" value="125" class="slider">
                        <input type="number" id="bHighH-input" min="0" max="179" value="125" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Saturation Min</span><span style="color:var(--text-muted)">0-255</span></div>
                    <div class="slider-controls">
                        <input type="range" id="bLowS" min="0" max="255" value="90" class="slider">
                        <input type="number" id="bLowS-input" min="0" max="255" value="90" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Value Min</span><span style="color:var(--text-muted)">0-255</span></div>
                    <div class="slider-controls">
                        <input type="range" id="bLowV" min="0" max="255" value="75" class="slider">
                        <input type="number" id="bLowV-input" min="0" max="255" value="75" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Ignore top ROI</span><span style="color:var(--text-muted)">0-45%</span></div>
                    <div class="slider-controls">
                        <input type="range" id="bRoiTop" min="0" max="45" value="22" class="slider">
                        <input type="number" id="bRoiTop-input" min="0" max="45" value="22" class="input-box">
                    </div>
                </div>
                </div>
                <div id="leader-blue-hsv-status" class="status"></div>
            </div>
'''

LEADER_BLUE_HSV_JS = '''
    let _leaderBlueHsvReady = false;

    const leaderBlueKeys = {
        'bLowH':  'leader_blue_h_min',
        'bHighH': 'leader_blue_h_max',
        'bLowS':  'leader_blue_s_min',
        'bLowV':  'leader_blue_v_min',
        'bRoiTop': 'leader_blue_roi_top_pct',
    };

    function formatLeaderBluePick(b) {
        if (!b) return '-';
        return `H ${b.h_min}-${b.h_max}, S≥${b.s_min}, V≥${b.v_min}`;
    }

    function loadLeaderBlueHsv() {
        fetch('/get_leader_blue_hsv')
            .then(r => r.json())
            .then(d => {
                setSliderValue('bLowH',  d.leader_blue_h_min);
                setSliderValue('bHighH', d.leader_blue_h_max);
                setSliderValue('bLowS',  d.leader_blue_s_min);
                setSliderValue('bLowV',  d.leader_blue_v_min);
                setSliderValue('bRoiTop', d.leader_blue_roi_top_pct);
                _leaderBlueHsvReady = true;
            })
            .catch(() => {});
    }

    function leaderBluePayloadFromSliders() {
        const payload = {};
        for (const [sliderId, key] of Object.entries(leaderBlueKeys)) {
            const el = document.getElementById(sliderId);
            if (!el) continue;
            const val = parseInt(el.value, 10);
            if (Number.isNaN(val)) {
                return null;
            }
            payload[key] = val;
        }
        return payload;
    }

    function pushLeaderBlueHsv(payload) {
        if (!_leaderBlueHsvReady) {
            return;
        }
        postJSON('/update_leader_blue_hsv', payload)
            .then(d => {
                if (d.status === 'error') {
                    showStatus('leader-blue-hsv-status', d.message || 'Save failed.', 'error');
                    return;
                }
                if (d.leader_blue_h_min != null) {
                    setSliderValue('bLowH',  d.leader_blue_h_min);
                    setSliderValue('bHighH', d.leader_blue_h_max);
                    setSliderValue('bLowS',  d.leader_blue_s_min);
                    setSliderValue('bLowV',  d.leader_blue_v_min);
                    setSliderValue('bRoiTop', d.leader_blue_roi_top_pct);
                }
                if (d.status === 'warning') {
                    showStatus('leader-blue-hsv-status', d.message || 'Applied live; yaml not saved.', 'error');
                    return;
                }
                showStatus('leader-blue-hsv-status', 'Leader blue HSV saved!', 'success');
            })
            .catch(() => showStatus('leader-blue-hsv-status', 'Leader blue HSV save failed.', 'error'));
    }

    Object.keys(leaderBlueKeys).forEach(sliderId => {
        syncSliderInput(sliderId, () => {
            const payload = leaderBluePayloadFromSliders();
            if (payload) {
                pushLeaderBlueHsv(payload);
            }
        });
    });

    loadLeaderBlueHsv();

    function applyLeaderBlueFromPick() {
        if (!window._lastPickSample || !window._lastPickSample.suggested_leader_blue) {
            showStatus('leader-blue-hsv-status', 'Click the camera panel first.', 'error');
            return;
        }
        const b = window._lastPickSample.suggested_leader_blue;
        setSliderValue('bLowH',  b.h_min);
        setSliderValue('bHighH', b.h_max);
        setSliderValue('bLowS',  b.s_min);
        setSliderValue('bLowV',  b.v_min);
        pushLeaderBlueHsv({
            leader_blue_h_min: b.h_min,
            leader_blue_h_max: b.h_max,
            leader_blue_s_min: b.s_min,
            leader_blue_v_min: b.v_min,
        });
    }

    window.onPixelSample = function(data) {
        window._lastPickSample = data;
        const el = document.getElementById('pick-leader-blue');
        if (el) {
            el.textContent = formatLeaderBluePick(data.suggested_leader_blue);
        }
    };

    (function addLeaderBluePickUi() {
        const box = document.getElementById('pick-result');
        if (!box || document.getElementById('pick-leader-blue')) return;
        const row = document.createElement('div');
        row.className = 'pick-row';
        row.innerHTML = '<span>Suggested leader blue</span><span id="pick-leader-blue" class="info-value">-</span>';
        const actions = box.querySelector('.pick-actions');
        box.insertBefore(row, actions);
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'button';
        btn.style.background = '#2980b9';
        btn.style.flex = '1';
        btn.textContent = 'Apply as Leader Blue';
        btn.onclick = applyLeaderBlueFromPick;
        actions.appendChild(btn);
    })();
'''
