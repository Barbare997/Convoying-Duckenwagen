"""Intersection turn timing sliders for project follower UI."""

INTERSECTION_TURN_CARD = '''
            <div class="card" id="intersectionTurnCard" style="display:none">
                <div class="card-header">Intersection Turns (follower)</div>
                <p style="font-size:12px;color:var(--text-muted);margin:0 0 12px">
                    Each button uses <strong>its own</strong> sliders below. Saves to
                    <code>project_config.yaml</code> on the machine running the task
                    (path shown after save — sim = your repo; hardware = the Duckiebot).
                </p>

                <div class="hsv-section-title" style="margin-top:0">Straight through → Go Straight</div>
                <div class="slider-group">
                    <div class="slider-label"><span>Straight forward (s)</span></div>
                    <div class="slider-controls">
                        <input type="range" id="ixStraight" min="0.2" max="3" step="0.05" value="1.75" class="slider">
                        <input type="number" id="ixStraight-input" min="0.2" max="3" step="0.05" value="1.75" class="input-box">
                    </div>
                </div>

                <div class="hsv-section-title">Turn left → Turn Left</div>
                <div class="slider-group">
                    <div class="slider-label"><span>Forward before turn (s)</span></div>
                    <div class="slider-controls">
                        <input type="range" id="ixLeftPre" min="0" max="2" step="0.05" value="0.69" class="slider">
                        <input type="number" id="ixLeftPre-input" min="0" max="2" step="0.05" value="0.69" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Turn arc (s)</span></div>
                    <div class="slider-controls">
                        <input type="range" id="ixLeftArc" min="0.2" max="4" step="0.05" value="2.38" class="slider">
                        <input type="number" id="ixLeftArc-input" min="0.2" max="4" step="0.05" value="2.38" class="input-box">
                    </div>
                </div>

                <div class="hsv-section-title">Turn right → Turn Right</div>
                <div class="slider-group">
                    <div class="slider-label"><span>Forward before turn (s)</span></div>
                    <div class="slider-controls">
                        <input type="range" id="ixRightPre" min="0" max="2" step="0.05" value="0.44" class="slider">
                        <input type="number" id="ixRightPre-input" min="0" max="2" step="0.05" value="0.44" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Turn arc (s)</span></div>
                    <div class="slider-controls">
                        <input type="range" id="ixRightArc" min="0.2" max="4" step="0.05" value="1.06" class="slider">
                        <input type="number" id="ixRightArc-input" min="0.2" max="4" step="0.05" value="1.06" class="input-box">
                    </div>
                </div>

                <div class="hsv-section-title">Arc PWM (all turns)</div>
                <div class="slider-group">
                    <div class="slider-label"><span>Turn speed (0–1)</span></div>
                    <div class="slider-controls">
                        <input type="range" id="ixTurnSpeed" min="0.05" max="1" step="0.01" value="0.32" class="slider">
                        <input type="number" id="ixTurnSpeed-input" min="0.05" max="1" step="0.01" value="0.32" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Inner wheel ratio</span></div>
                    <div class="slider-controls">
                        <input type="range" id="ixInner" min="0.05" max="1" step="0.01" value="0.38" class="slider">
                        <input type="number" id="ixInner-input" min="0.05" max="1" step="0.01" value="0.38" class="input-box">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label"><span>Outer wheel ratio</span></div>
                    <div class="slider-controls">
                        <input type="range" id="ixOuter" min="0.05" max="1" step="0.01" value="0.72" class="slider">
                        <input type="number" id="ixOuter-input" min="0.05" max="1" step="0.01" value="0.72" class="input-box">
                    </div>
                </div>

                <button type="button" onclick="pushIntersectionTurnParams()" class="button success" style="width:100%;margin-top:4px">Apply turn params</button>
                <div id="intersection-turn-status" class="status"></div>

                <div class="hsv-section-title">Test drive (no Start)</div>
                <p style="font-size:12px;color:var(--text-muted);margin:0 0 10px">
                    Saves sliders, then runs the timed sequence — works while <strong>paused</strong>.
                </p>
                <div style="display:flex;gap:8px;flex-wrap:wrap">
                    <button type="button" onclick="testIntersectionTurn('left')" class="button" style="flex:1;min-width:88px;background:#2980b9;color:#fff">Turn Left</button>
                    <button type="button" onclick="testIntersectionTurn('straight')" class="button success" style="flex:1;min-width:88px">Go Straight</button>
                    <button type="button" onclick="testIntersectionTurn('right')" class="button" style="flex:1;min-width:88px;background:#8e44ad;color:#fff">Turn Right</button>
                </div>
            </div>
'''

INTERSECTION_TURN_JS = '''
    const intersectionTurnKeys = {
        'ixStraight': 'intersection_turn_straight_s',
        'ixLeftPre': 'intersection_left_preamble_s',
        'ixLeftArc': 'intersection_turn_left_s',
        'ixRightPre': 'intersection_right_preamble_s',
        'ixRightArc': 'intersection_turn_right_s',
        'ixTurnSpeed': 'intersection_turn_speed',
        'ixInner': 'intersection_turn_inner_ratio',
        'ixOuter': 'intersection_turn_outer_ratio',
    };

    function intersectionTurnPayload() {
        const payload = {};
        Object.entries(intersectionTurnKeys).forEach(([sliderId, key]) => {
            payload[key] = parseFloat(document.getElementById(sliderId).value);
        });
        return payload;
    }

    function applyIntersectionTurnResponse(d) {
        if (d.intersection_turn_straight_s == null) return;
        setSliderValue('ixStraight', d.intersection_turn_straight_s);
        setSliderValue('ixLeftPre', d.intersection_left_preamble_s);
        setSliderValue('ixLeftArc', d.intersection_turn_left_s);
        setSliderValue('ixRightPre', d.intersection_right_preamble_s);
        setSliderValue('ixRightArc', d.intersection_turn_right_s);
        setSliderValue('ixTurnSpeed', d.intersection_turn_speed);
        setSliderValue('ixInner', d.intersection_turn_inner_ratio);
        setSliderValue('ixOuter', d.intersection_turn_outer_ratio);
    }

    function loadIntersectionTurnParams() {
        fetch('/get_intersection_turn')
            .then(r => r.json())
            .then(d => applyIntersectionTurnResponse(d))
            .catch(() => {});
    }

    function pushIntersectionTurnParams() {
        return postJSON('/update_intersection_turn', intersectionTurnPayload())
            .then(d => {
                if (d.status === 'error') {
                    showStatus('intersection-turn-status', d.message || 'Save failed.', 'error');
                    throw new Error('save failed');
                }
                applyIntersectionTurnResponse(d);
                const where = d.config_path ? (' File: ' + d.config_path) : '';
                showStatus(
                    'intersection-turn-status',
                    'Saved — left arc ' + d.intersection_turn_left_s + 's, right arc ' +
                    d.intersection_turn_right_s + 's, straight ' + d.intersection_turn_straight_s +
                    's.' + where,
                    'success',
                );
                return d;
            })
            .catch((err) => {
                if (err && err.message === 'save failed') return;
                showStatus('intersection-turn-status', 'Save failed (network or server error).', 'error');
                throw err;
            });
    }

    function bindIntersectionTurnSlider(sliderId) {
        const slider = document.getElementById(sliderId);
        const input = document.getElementById(sliderId + '-input');
        let timeout = null;
        const saveSoon = () => {
            clearTimeout(timeout);
            timeout = setTimeout(pushIntersectionTurnParams, 80);
        };
        slider.addEventListener('input', function() {
            input.value = this.value;
            saveSoon();
        });
        slider.addEventListener('change', pushIntersectionTurnParams);
        input.addEventListener('input', function() {
            let val = parseFloat(this.value);
            if (!Number.isFinite(val)) return;
            val = Math.max(parseFloat(this.min), Math.min(parseFloat(this.max), val));
            this.value = val;
            slider.value = val;
            saveSoon();
        });
        input.addEventListener('change', pushIntersectionTurnParams);
    }

    Object.keys(intersectionTurnKeys).forEach(bindIntersectionTurnSlider);
    loadIntersectionTurnParams();

    function testIntersectionTurn(direction) {
        pushIntersectionTurnParams()
            .then(() => postJSON('/intersection/test_turn', { direction: direction }))
            .then(d => {
                if (d.status === 'error') {
                    showStatus('intersection-turn-status', d.message || 'Test turn failed.', 'error');
                    return;
                }
                showStatus(
                    'intersection-turn-status',
                    'Running test turn: ' + d.direction + ' (check console for arc timing)',
                    'success',
                );
            })
            .catch(() => showStatus('intersection-turn-status', 'Test turn request failed.', 'error'));
    }
'''
