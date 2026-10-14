/* Chart.js helpers. Colours follow a fixed entity order (dark-mode steps). */
const SERIES_DARK = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
const CONFIG_COLOR = {dense: '#3987e5', headrouter: '#d95926', static: '#199e70', random: '#c98500', oracle: '#d55181'};
const INK2 = '#c3c2b7', GRIDC = 'rgba(255,255,255,0.07)';
const charts = {};

function baseOptions(yTitle, xTitle) {
    return {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: {mode: 'index', intersect: false},
        plugins: {
            legend: {labels: {color: INK2, usePointStyle: true, boxWidth: 8}},
            tooltip: {backgroundColor: '#1a1a19', borderColor: 'rgba(255,255,255,.1)', borderWidth: 1, titleColor: '#f0efec', bodyColor: INK2},
        },
        scales: {
            x: {title: {display: !!xTitle, text: xTitle, color: INK2}, ticks: {color: INK2}, grid: {color: GRIDC}},
            y: {title: {display: !!yTitle, text: yTitle, color: INK2}, ticks: {color: INK2}, grid: {color: GRIDC}},
        },
    };
}

function drawSweepChart(id, tasks, reg) {
    charts[id]?.destroy();
    const datasets = tasks.map(([k, t], i) => {
        const d = t.profile.sweep[0].score || 1e-9;
        return {label: k, data: t.profile.sweep.map((p) => ({x: p.sparsity * 100, y: p.score / d * 100})),
            borderColor: SERIES_DARK[i], backgroundColor: SERIES_DARK[i], borderWidth: 2, pointRadius: 4, pointHoverRadius: 6};
    });
    if (reg && reg.budget_type === 'relative') {
        datasets.push({label: `budget (−${(reg.accuracy_budget * 100).toFixed(1)}%)`, data: [{x: 0, y: 100 - reg.accuracy_budget * 100}, {x: 50, y: 100 - reg.accuracy_budget * 100}],
            borderColor: '#8b8a84', borderDash: [5, 4], borderWidth: 1, pointRadius: 0});
    }
    const opt = baseOptions('dev score (% of dense)', 'heads pruned (%)');
    opt.scales.x.type = 'linear';
    opt.interaction = {mode: 'nearest', intersect: false};
    opt.plugins.tooltip.callbacks = {label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(1)}% at ${c.parsed.x.toFixed(0)}% pruned`};
    charts[id] = new Chart(document.getElementById(id), {type: 'line', data: {datasets}, options: opt});
}

function drawLatencyChart(id, B, tasks) {
    charts[id]?.destroy();
    const configs = Object.keys(B.speed[tasks[0]]);
    const datasets = configs.map((c) => ({
        label: CFG_LABEL[c], backgroundColor: CONFIG_COLOR[c], borderRadius: 4, borderSkipped: 'bottom',
        data: tasks.map((t) => B.speed[t][c].total_ms.median),
    }));
    const opt = baseOptions('median latency (ms)', '');
    opt.plugins.tooltip.callbacks = {label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(0)} ms`};
    charts[id] = new Chart(document.getElementById(id), {type: 'bar', data: {labels: tasks, datasets}, options: opt});
}
