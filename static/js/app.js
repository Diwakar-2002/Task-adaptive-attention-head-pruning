/* HeadRouter dashboard. Every value shown comes from the API (measured live or read from artifacts). */
const $ = (s) => document.querySelector(s);
const api = async (path, body) => {
    const r = await fetch(path, body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {});
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
};
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : Number(v).toFixed(d);
const pct = (v, d = 1) => (v === null || v === undefined) ? '–' : `${(v * 100).toFixed(d)}%`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));

let STATUS = null;
let IMPORTANCE = null;
const HISTORY = [];

/* ------------------------------------------------------------------ tabs */
document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('active', x === b));
    document.querySelectorAll('.panel').forEach((p) => p.classList.toggle('active', p.id === `tab-${b.dataset.tab}`));
    if (b.dataset.tab === 'masks') loadMasks();
    if (b.dataset.tab === 'sweep') renderSweep();
    if (b.dataset.tab === 'bench') loadBench();
}));

/* ---------------------------------------------------------------- status */
async function loadStatus() {
    STATUS = await api('/api/status');
    const a = STATUS.arch, art = STATUS.artifacts;
    $('#status-pills').innerHTML = [
        `<span class="pill">model <b>${esc(STATUS.model.split('/').pop())}</b></span>`,
        `<span class="pill">${a.layers}×${a.heads} heads <b>(${a.kv_heads} KV)</b></span>`,
        `<span class="pill">device <b>${esc(STATUS.gpu)}</b></span>`,
        `<span class="pill">energy <b>${esc(STATUS.energy_mode)}</b></span>`,
        `<span class="pill ${art.classifier ? 'ok' : 'no'}">classifier ${art.classifier ? '✓' : '✗'}</span>`,
        `<span class="pill ${art.registry ? 'ok' : 'no'}">masks ${art.registry ? '✓' : '✗'}</span>`,
        `<span class="pill ${art.benchmark ? 'ok' : 'no'}">benchmark ${art.benchmark ? '✓' : '✗'}</span>`,
    ].join('');
    const missing = [];
    if (!art.classifier) missing.push('<code>scripts/01_train_classifier.py</code>');
    if (!art.registry) missing.push('<code>scripts/02_profile.py</code> + <code>scripts/03_build_masks.py</code>');
    const ban = $('#artifact-banner');
    if (missing.length || STATUS.warnings.length) {
        ban.innerHTML = (missing.length ? `HeadRouter needs ${missing.join(' and ')} — until then the pruned side runs dense. ` : '') + STATUS.warnings.map(esc).join(' ');
        ban.classList.remove('hidden');
    }
    const sel = $('#force-task');
    Object.entries(STATUS.tasks).forEach(([k, t]) => {
        const sp = t.profile ? ` – ${pct(t.profile.actual_sparsity, 0)} pruned` : '';
        sel.insertAdjacentHTML('beforeend', `<option value="${k}">Force: ${esc(t.name)}${sp}</option>`);
    });
}

async function loadExamples() {
    const ex = await api('/api/examples');
    const box = $('#examples');
    Object.entries(ex).forEach(([task, prompts]) => prompts.slice(0, 2).forEach((p, i) => {
        const b = document.createElement('button');
        b.className = 'chip';
        b.textContent = `${task} #${i + 1}`;
        b.title = p.slice(0, 300);
        b.onclick = () => { $('#prompt').value = p; };
        box.appendChild(b);
    }));
    if (!$('#prompt').value && ex.sentiment) $('#prompt').value = ex.sentiment[0];
}

/* ------------------------------------------------------------------ live */
function metricsHTML(m, hr) {
    const cells = [
        ['End-to-end', `${fmt(m.e2e_ms, 0)}<small> ms</small>`],
        ['TTFT (prefill)', `${fmt(m.ttft_ms, 1)}<small> ms</small>`],
        ['Per token', `${fmt(m.tpot_ms, 2)}<small> ms</small>`],
        ['Throughput', `${fmt(m.tokens_per_s, 1)}<small> tok/s</small>`],
        ['GPU energy', m.energy_j === null ? '<small>run &lt; 0.5 s</small>' : `${fmt(m.energy_j, 2)}<small> J</small>`],
        ['Energy / token', `${fmt(m.energy_per_token_mj, 0)}<small> mJ</small>`],
        ['Tokens', `${m.new_tokens}<small> / ${m.input_tokens} in</small>`],
        [hr ? 'Classifier' : 'Working mem', hr ? `${fmt(m.classifier_ms, 1)}<small> ms</small>` : `${fmt(m.working_mem_mb, 0)}<small> MB</small>`],
    ];
    return cells.map(([k, v]) => `<div class="m"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
}

function renderRouting(r, maskInfo) {
    if (!r) return;
    $('#route-ms').textContent = r.forced ? 'forced' : `${fmt(r.classifier_ms, 1)} ms`;
    const probs = Object.entries(r.probs || {}).sort((a, b) => b[1] - a[1]);
    const rows = probs.map(([k, v], i) => `<div class="prob-row ${i === 0 ? 'top' : ''}"><span>${esc(k)}</span>
        <div class="bar"><i style="width:${(v * 100).toFixed(1)}%"></i></div><span>${pct(v, 1)}</span></div>`).join('');
    const pruned = r.variant !== 'dense';
    const st = maskInfo;
    const decision = pruned
        ? `<div class="decision pruned"><b>Pruned execution:</b> mask <code>${esc(r.variant)}</code> — ${st.pruned_heads}/${st.total_heads} query heads
           (${pct(st.sparsity, 1)}) and ${st.total_kv_heads - st.kept_kv_heads}/${st.total_kv_heads} KV heads physically removed.</div>`
        : `<div class="decision dense"><b>Dense fallback:</b> ${esc(r.fallback || 'no mask')}.</div>`;
    $('#routing-body').innerHTML = `<div>Predicted task: <b>${esc(r.task ?? '–')}</b>
        (confidence ${pct(r.confidence, 1)}, threshold ${pct(STATUS.threshold, 0)})</div>
        <div class="probs">${rows}</div>${decision}`;
}

function renderMaskMini(mask) {
    const box = $('#mask-mini');
    box.style.gridTemplateColumns = `repeat(${mask[0].length}, 1fr)`;
    box.innerHTML = mask.map((row) => row.map((k) => `<div class="c ${k ? '' : 'off'}"></div>`).join('')).join('');
    box.title = 'Rows = layers, columns = query heads. Grey = pruned.';
}

function lcsRatio(a, b) {
    const x = a.split(/\s+/).filter(Boolean), y = b.split(/\s+/).filter(Boolean);
    if (!x.length && !y.length) return 1;
    const dp = new Array(y.length + 1).fill(0);
    for (let i = 1; i <= x.length; i++) {
        let prev = 0;
        for (let j = 1; j <= y.length; j++) {
            const tmp = dp[j];
            dp[j] = x[i - 1] === y[j - 1] ? prev + 1 : Math.max(dp[j], dp[j - 1]);
            prev = tmp;
        }
    }
    return (2 * dp[y.length]) / (x.length + y.length);
}

function deltaCard(k, v, cls, s) {
    return `<div class="d"><div class="k">${k}</div><div class="v ${cls}">${v}</div><div class="s">${s}</div></div>`;
}

function renderDelta(d, h) {
    const a = d.metrics, b = h.metrics;
    const sp = (x, y) => (x && y) ? (x / y - 1) : null;
    const cls = (v) => v === null ? '' : (v >= 0 ? 'pos' : 'neg');
    const e2e = sp(a.e2e_ms, b.e2e_ms), tpot = sp(a.tpot_ms, b.tpot_ms), ttft = sp(a.ttft_ms, b.ttft_ms);
    const eSave = (a.energy_per_token_mj && b.energy_per_token_mj) ? 1 - b.energy_per_token_mj / a.energy_per_token_mj : null;
    const same = d.text === h.text;
    const sim = lcsRatio(d.text, h.text);
    $('#delta-grid').innerHTML = [
        deltaCard('End-to-end speedup', pct(e2e), cls(e2e), `${fmt(a.e2e_ms, 0)} → ${fmt(b.e2e_ms, 0)} ms`),
        deltaCard('Decode speedup / token', pct(tpot), cls(tpot), `${fmt(a.tpot_ms, 2)} → ${fmt(b.tpot_ms, 2)} ms`),
        deltaCard('Prefill (TTFT) speedup', pct(ttft), cls(ttft), `${fmt(a.ttft_ms, 1)} → ${fmt(b.ttft_ms, 1)} ms`),
        deltaCard('Energy saved / token', pct(eSave), cls(eSave), `${fmt(a.energy_per_token_mj, 0)} → ${fmt(b.energy_per_token_mj, 0)} mJ`),
        deltaCard('Output agreement', same ? 'identical' : pct(sim, 0), same ? 'pos' : '', same ? 'token-for-token' : 'word-level LCS similarity'),
        deltaCard('Heads computed', `${h.mask_stats.kept_heads}/${h.mask_stats.total_heads}`, '', `KV heads ${h.mask_stats.kept_kv_heads}/${h.mask_stats.total_kv_heads}`),
    ].join('');
    const notes = [`Medians over ${a.runs} run(s) each; GPU timings synchronised.`];
    if (a.new_tokens !== b.new_tokens) notes.push(`Outputs have different lengths (${a.new_tokens} vs ${b.new_tokens} tokens) — compare the per-token numbers.`);
    if (h.variant === 'dense') notes.push('The router fell back to the dense model, so both sides run the same weights; any difference is measurement noise.');
    $('#delta-note').textContent = notes.join(' ');

    HISTORY.unshift({task: h.routing?.task, variant: h.variant, sp: h.mask_stats.sparsity, tok: `${a.new_tokens}/${b.new_tokens}`,
        de: a.e2e_ms, he: b.e2e_ms, e2e, dt: a.tpot_ms, ht: b.tpot_ms, eSave, same: same ? 'yes' : pct(sim, 0)});
    renderHistory();
}

function renderHistory() {
    $('#history tbody').innerHTML = HISTORY.map((r) => `<tr><td>${esc(r.task ?? '–')}</td><td>${esc(r.variant)}</td><td>${pct(r.sp, 0)}</td>
        <td>${r.tok}</td><td>${fmt(r.de, 0)}</td><td>${fmt(r.he, 0)}</td><td>${pct(r.e2e)}</td><td>${fmt(r.dt, 2)}</td>
        <td>${fmt(r.ht, 2)}</td><td>${pct(r.eSave)}</td><td>${r.same}</td></tr>`).join('');
}
$('#clear-history').onclick = () => { HISTORY.length = 0; renderHistory(); };

async function runCompare() {
    const prompt = $('#prompt').value.trim();
    if (!prompt) return;
    const body = {prompt, force_task: $('#force-task').value || null,
        max_new_tokens: parseInt($('#max-tokens').value, 10) || 128, runs: parseInt($('#runs').value, 10)};
    const btn = $('#run-btn');
    btn.disabled = true;
    $('#dense-text').textContent = ''; $('#hr-text').textContent = '';
    $('#dense-metrics').innerHTML = ''; $('#hr-metrics').innerHTML = '';
    $('#delta-grid').innerHTML = '<span class="muted">Waiting for both runs.</span>';
    try {
        const route = await api('/api/classify', {prompt, force_task: body.force_task});
        $('#hr-sub').textContent = route.variant === 'dense' ? '(dense fallback)' : `(${route.variant} mask)`;
        $('#dense-state').innerHTML = '<span class="spin"></span> generating';
        const d = await api('/api/generate', {...body, mode: 'dense'});
        $('#dense-state').textContent = `${fmt(d.metrics.e2e_ms, 0)} ms`;
        $('#dense-text').textContent = d.text;
        $('#dense-metrics').innerHTML = metricsHTML(d.metrics, false);

        $('#hr-state').innerHTML = '<span class="spin"></span> generating';
        const h = await api('/api/generate', {...body, mode: 'headrouter'});
        $('#hr-state').textContent = `${fmt(h.metrics.e2e_ms, 0)} ms`;
        $('#hr-text').textContent = h.text;
        $('#hr-metrics').innerHTML = metricsHTML(h.metrics, true);
        renderRouting(h.routing, h.mask_stats);
        renderMaskMini(h.mask);
        renderDelta(d, h);
    } catch (e) {
        $('#delta-grid').innerHTML = `<span class="muted">Error: ${esc(e.message)}</span>`;
        $('#dense-state').textContent = $('#hr-state').textContent = 'error';
    } finally {
        btn.disabled = false;
    }
}
$('#run-btn').onclick = runCompare;
$('#prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runCompare(); });

/* ----------------------------------------------------------------- masks */
const RAMP = ['#0d366b', '#104281', '#184f95', '#1c5cab', '#256abf', '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef', '#9ec5f4', '#b7d3f6', '#cde2fb'];

async function loadMasks() {
    if (!IMPORTANCE) {
        try { IMPORTANCE = await api('/api/importance'); } catch (e) {
            $('#mask-desc').textContent = e.message; return;
        }
        const seg = $('#mask-task-seg');
        [...IMPORTANCE.tasks, 'static'].forEach((t, i) => {
            const b = document.createElement('button');
            b.textContent = t === 'static' ? 'static (baseline)' : t;
            b.onclick = () => { seg.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b)); drawHeat(t); };
            if (i === 0) b.classList.add('on');
            seg.appendChild(b);
        });
        drawHeat(IMPORTANCE.tasks[0]);
    }
}

function drawHeat(task) {
    const I = IMPORTANCE;
    const isStatic = task === 'static';
    const imp = isStatic ? meanRank(I.importance) : I.importance[I.tasks.indexOf(task)];
    const keep = isStatic ? I.static_keep : I.keep[task];
    const L = imp.length, H = imp[0].length;
    const order = imp.flat().map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
    const rank = new Array(order.length);
    order.forEach(([, i], r) => { rank[i] = r / (order.length - 1); });
    const heat = $('#heat');
    heat.style.gridTemplateColumns = `24px repeat(${H}, 1fr)`;
    let html = '';
    for (let l = 0; l < L; l++) {
        html += `<div class="ax">${l}</div>`;
        for (let h = 0; h < H; h++) {
            const t = rank[l * H + h];
            const col = RAMP[Math.round(t * (RAMP.length - 1))];
            const cls = [keep[l][h] ? '' : 'pruned', I.protected[l][h] ? 'protected' : ''].join(' ');
            html += `<div class="c ${cls}" style="background:${col}" data-l="${l}" data-h="${h}" data-v="${imp[l][h]}" data-k="${keep[l][h] ? 1 : 0}"></div>`;
        }
    }
    html += '<div></div>' + Array.from({length: H}, (_, h) => `<div class="ax">${h}</div>`).join('');
    heat.innerHTML = html;
    const pruned = keep.flat().filter((k) => !k).length;
    const st = STATUS.tasks[task];
    $('#mask-desc').textContent = isStatic
        ? 'Static baseline: one task-agnostic mask (mean rank of importance across tasks) applied to every request.'
        : `Importance = ${I.scorer} scorer${I.normalized ? ', layer-normalised' : ''} (colour = rank). Mask chosen as the highest sparsity whose dev score stayed within the accuracy budget.`;
    const rows = [['heads pruned', `${pruned} / ${L * H} (${pct(pruned / (L * H), 1)})`]];
    if (st && st.profile) {
        rows.push(['dense dev score', fmt(st.profile.dense_dev, 4)], ['pruned dev score', fmt(st.profile.pruned_dev, 4)], ['metric', st.metric]);
        if (st.stats) rows.push(['KV heads kept', `${st.stats.kept_kv_heads} / ${st.stats.total_kv_heads}`], ['extra weights', `${fmt(st.extra_weight_mb, 0)} MB`]);
    }
    $('#mask-stats').innerHTML = rows.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join('');
}

function meanRank(stack) {
    const L = stack[0].length, H = stack[0][0].length;
    const out = Array.from({length: L}, () => new Array(H).fill(0));
    stack.forEach((m) => {
        const flat = m.flat().map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
        flat.forEach(([, i], r) => { out[Math.floor(i / H)][i % H] += r / (flat.length - 1) / stack.length; });
    });
    return out;
}

const tip = $('#heat-tip');
$('#heat').addEventListener('mousemove', (e) => {
    const c = e.target.closest('.c');
    if (!c) { tip.classList.add('hidden'); return; }
    tip.innerHTML = `layer ${c.dataset.l}, head ${c.dataset.h}<br>importance ${Number(c.dataset.v).toFixed(4)}<br>${c.dataset.k === '1' ? 'kept' : 'PRUNED'}${c.classList.contains('protected') ? ' · protected' : ''}`;
    tip.style.left = `${e.clientX + 14}px`; tip.style.top = `${e.clientY + 14}px`;
    tip.classList.remove('hidden');
});
$('#heat').addEventListener('mouseleave', () => tip.classList.add('hidden'));

/* ----------------------------------------------------------------- sweep */
function renderSweep() {
    const tasks = Object.entries(STATUS.tasks).filter(([, t]) => t.profile);
    if (!tasks.length) { $('#sweep-table').innerHTML = '<tr><td>No registry yet — run scripts/03_build_masks.py</td></tr>'; return; }
    drawSweepChart('sweep-chart', tasks, STATUS.registry);
    const levels = [...new Set(tasks.flatMap(([, t]) => t.profile.sweep.map((p) => p.sparsity)))].sort((a, b) => a - b);
    $('#sweep-table').innerHTML = `<thead><tr><th>Task</th><th>Metric</th>${levels.map((l) => `<th>${pct(l, 0)}</th>`).join('')}<th>Chosen</th></tr></thead><tbody>` +
        tasks.map(([k, t]) => `<tr><td>${esc(k)}</td><td>${t.metric}</td>${levels.map((l) => {
            const p = t.profile.sweep.find((x) => x.sparsity === l);
            return `<td>${p ? fmt(p.score, 4) + (p.ok ? '' : ' ✗') : '·'}</td>`;
        }).join('')}<td>${pct(t.profile.sparsity, 0)}</td></tr>`).join('') + '</tbody>';
}

/* ----------------------------------------------------------------- bench */
const CFG_LABEL = {dense: 'Dense', static: 'Static pruning', random: 'Random pruning', oracle: 'HeadRouter (oracle task)', headrouter: 'HeadRouter'};

async function loadBench() {
    let B;
    try { B = await api('/api/benchmark'); } catch (e) {
        $('#bench-body').innerHTML = `<div class="glass card"><p>No benchmark results yet. Run:</p>
            <pre class="cmd">.venv/bin/python scripts/04_benchmark.py          # full\n.venv/bin/python scripts/04_benchmark.py --quick  # smoke run</pre></div>`;
        return;
    }
    const tasks = Object.keys(B.registry);
    const m = B.meta;
    let html = `<div class="glass card"><div class="card-h"><h3>Run</h3><span class="tag">${esc(m.created)}${m.quick ? ' · quick' : ''}</span></div>
        <p class="muted">${esc(m.model)} (${m.dtype}) on ${esc(m.gpu)} · torch ${m.torch} · energy via NVML ${m.energy_mode} · ${m.speed_new_tokens} new tokens per speed run</p></div>`;
    if (B.quality) {
        html += `<div class="glass card"><div class="card-h"><h3>Quality on the test split</h3></div><div class="table-wrap"><table class="tbl"><thead><tr><th>Task</th><th>Metric</th>
            ${Object.values(CFG_LABEL).map((l) => `<th>${l}</th>`).join('')}<th>Classifier acc</th><th>Dense fallback</th></tr></thead><tbody>` +
            tasks.map((t) => `<tr><td>${t}</td><td>${m.metric_names[t]}</td>${Object.keys(CFG_LABEL).map((c) => `<td>${fmt(B.quality[t][c], 4)}</td>`).join('')}
            <td>${pct(B.routing[t].classifier_accuracy)}</td><td>${pct(B.routing[t].fallback_rate)}</td></tr>`).join('') + '</tbody></table></div></div>';
    }
    if (B.speed) {
        html += `<div class="glass card"><div class="card-h"><h3>Latency per task (median, ${m.speed_new_tokens} tokens)</h3></div><div class="chart-box"><canvas id="bench-lat"></canvas></div>
            <div class="table-wrap"><table class="tbl"><thead><tr><th>Task</th><th>Config</th><th>Latency ms</th><th>p95</th><th>TTFT ms</th><th>ms/token</th><th>tok/s</th><th>Energy J</th><th>Speedup</th></tr></thead><tbody>` +
            tasks.flatMap((t) => Object.entries(B.speed[t]).map(([c, v]) => {
                const d = B.speed[t].dense.total_ms.median;
                const g = (k) => v[k] ? v[k].median : null;
                return `<tr><td>${t}</td><td>${CFG_LABEL[c]}</td><td>${fmt(g('total_ms'), 0)}</td><td>${fmt(v.total_ms.p95, 0)}</td><td>${fmt(g('ttft_ms'), 1)}</td>
                    <td>${fmt(g('tpot_ms'), 2)}</td><td>${fmt(g('tokens_per_s'), 1)}</td><td>${fmt(g('energy_j'), 2)}</td><td>${fmt(d / g('total_ms'), 3)}×</td></tr>`;
            })).join('') + '</tbody></table></div></div>';
    }
    const imgs = ['context_sweep.png', 'sparsity_sweep.png', 'quality.png', 'energy.png', 'importance_heatmaps.png', 'accuracy_vs_sparsity.png', 'task_similarity.png', 'study_scorer_compensation.png', 'study_iterative.png'];
    html += `<div class="bench-grid">${imgs.map((p) => `<div class="glass card"><img class="plot-img" src="/plots/${p}?t=${Date.now()}" alt="${p}" onerror="this.parentElement.remove()"></div>`).join('')}</div>`;
    $('#bench-body').innerHTML = html;
    if (B.speed) drawLatencyChart('bench-lat', B, tasks);
}

/* ------------------------------------------------------------------ boot */
(async () => {
    try {
        await loadStatus();
        await loadExamples();
    } catch (e) {
        $('#status-pills').innerHTML = `<span class="pill no">API unreachable: ${esc(e.message)}</span>`;
    }
})();
