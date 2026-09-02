/* ==========================================================================
   Chart.js Rendering Engine for CausalPrune-AI
   ========================================================================== */

let accuracyChart = null;
let klChart = null;
let latencyChart = null;
let radarChart = null;

function initBenchmarkCharts() {
    const accuracyCtx = document.getElementById('accuracyChart')?.getContext('2d');
    const klCtx = document.getElementById('klChart')?.getContext('2d');
    const latencyCtx = document.getElementById('latencyChart')?.getContext('2d');
    const radarCtx = document.getElementById('radarChart')?.getContext('2d');

    if (!accuracyCtx) return;

    // Common styling
    Chart.defaults.color = '#9CA3AF';
    Chart.defaults.font.family = "'Inter', sans-serif";

    // 1. Accuracy Retention Chart
    accuracyChart = new Chart(accuracyCtx, {
        type: 'line',
        data: {
            labels: ['0%', '10%', '20%', '30%', '40%', '50%'],
            datasets: [
                {
                    label: 'Task-Aware Dynamic (Ours)',
                    data: [100, 99.8, 99.7, 99.6, 99.5, 99.2],
                    borderColor: '#10B981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Static Causal Pruning (Ours)',
                    data: [100, 99.5, 99.1, 98.8, 98.4, 97.1],
                    borderColor: '#8B5CF6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Magnitude Pruning (Baseline)',
                    data: [100, 96.2, 91.5, 87.1, 84.2, 73.0],
                    borderColor: '#F43F5E',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: { min: 70, max: 100, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });

    // 2. Logit KL Divergence Chart
    klChart = new Chart(klCtx, {
        type: 'bar',
        data: {
            labels: ['Full Model', 'Task Dynamic', 'Static Causal', 'Magnitude Baseline'],
            datasets: [{
                label: 'Logit KL Divergence (Lower is Better)',
                data: [0.000, 0.008, 0.021, 0.185],
                backgroundColor: ['#06B6D4', '#10B981', '#8B5CF6', '#F43F5E']
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });

    // 3. Inference Latency Chart
    latencyChart = new Chart(latencyCtx, {
        type: 'bar',
        data: {
            labels: ['Full Model', 'Magnitude (40%)', 'Static Causal (40%)', 'Task Dynamic (55% Cut)'],
            datasets: [{
                label: 'ms / token (Lower is Faster)',
                data: [18.4, 12.1, 11.8, 8.7],
                backgroundColor: ['#6B7280', '#F43F5E', '#8B5CF6', '#10B981']
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });

    // 4. Multi-Metric Radar Chart
    radarChart = new Chart(radarCtx, {
        type: 'radar',
        data: {
            labels: ['Accuracy %', 'Speedup Factor', 'FLOP Efficiency', 'Safety Fidelity', 'Memory Savings'],
            datasets: [
                {
                    label: 'Task-Aware Dynamic',
                    data: [99, 95, 90, 98, 88],
                    borderColor: '#10B981',
                    backgroundColor: 'rgba(16, 185, 129, 0.2)'
                },
                {
                    label: 'Magnitude Pruning',
                    data: [75, 70, 70, 45, 60],
                    borderColor: '#F43F5E',
                    backgroundColor: 'rgba(244, 63, 94, 0.2)'
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255,255,255,0.1)' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    pointLabels: { color: '#9CA3AF' }
                }
            }
        }
    });
}

function updateBenchmarkChartsWithData(data) {
    if (!data || !data.strategies) return;
    const strats = data.strategies;

    if (latencyChart) {
        latencyChart.data.datasets[0].data = [
            strats.unpruned.ms_per_token,
            strats.magnitude.ms_per_token,
            strats.static_causal.ms_per_token,
            strats.task_dynamic.ms_per_token
        ];
        latencyChart.update();
    }

    if (klChart) {
        klChart.data.datasets[0].data = [
            strats.unpruned.kl_divergence,
            strats.task_dynamic.kl_divergence,
            strats.static_causal.kl_divergence,
            strats.magnitude.kl_divergence
        ];
        klChart.update();
    }
}
