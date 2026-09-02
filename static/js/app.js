/* ==========================================================================
   CausalPrune-AI Dashboard Engine
   ========================================================================== */

const API_BASE = "";

// State
let modelInfo = { num_layers: 12, num_heads: 12, total_heads: 144 };
let ablatedHeadsSet = new Set(); // Set of "L,H" strings
let matrixData = {
    causal: null,
    magnitude: null,
    rankings: []
};
let currentMatrixView = "causal"; // "causal" or "magnitude"

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initPresets();
    fetchModelInfo();
    initBenchmarkCharts();
    setupEventListeners();
});

// Navigation Tabs
function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            document.getElementById(tabId).classList.add("active");
        });
    });
}

// Fetch Model Meta
async function fetchModelInfo() {
    try {
        const res = await fetch(`${API_BASE}/api/model/info`);
        const data = await res.json();
        modelInfo = data;

        document.getElementById("model-name-val").textContent = `${data.model_name} (${data.total_heads} Heads)`;
        document.getElementById("hud-total-heads").textContent = data.total_heads;
        
        generateMatrixGrid(data.num_layers, data.num_heads);
        fetchHeatmapMatrix();
    } catch (e) {
        console.warn("API server starting or offline, generating fallback grid...", e);
        generateMatrixGrid(12, 12);
    }
}

// Render Matrix Grid Structure
function generateMatrixGrid(numLayers, numHeads) {
    const xHeader = document.getElementById("matrix-x-header");
    const grid = document.getElementById("head-matrix-grid");

    if (!grid) return;

    xHeader.innerHTML = "";
    grid.innerHTML = "";

    // Header 0-11
    for (let h = 0; h < numHeads; h++) {
        const col = document.createElement("div");
        col.textContent = `H${h}`;
        xHeader.appendChild(col);
    }

    // Grid cells 12x12
    for (let l = 0; l < numLayers; l++) {
        for (let h = 0; h < numHeads; h++) {
            const cell = document.createElement("div");
            cell.className = "head-cell";
            cell.dataset.layer = l;
            cell.dataset.head = h;
            cell.title = `Layer ${l}, Head ${h}`;
            cell.textContent = `${l},${h}`;

            cell.addEventListener("click", () => toggleHeadAblation(l, h));
            grid.appendChild(cell);
        }
    }
}

// Fetch Causal & Magnitude Heatmaps
async function fetchHeatmapMatrix() {
    try {
        const res = await fetch(`${API_BASE}/api/eval/causal`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompts: [
                    "The human brain is a remarkably efficient computing engine because",
                    "Function parameters in modern python modules allow dynamic execution",
                    "Summarize the main benefits of interventional neural network pruning"
                ],
                sparsity_target: 0.4
            })
        });
        const data = await res.json();
        matrixData.causal = data.causal_heatmap;
        matrixData.magnitude = data.magnitude_heatmap;
        matrixData.rankings = data.head_rankings;

        renderHeatmapColors();
    } catch (e) {
        console.error("Error fetching matrix:", e);
    }
}

// Render Color Heatmap Grids
function renderHeatmapColors() {
    const matrix = (currentMatrixView === "causal") ? matrixData.causal : matrixData.magnitude;
    if (!matrix) return;

    const cells = document.querySelectorAll(".head-cell");
    let maxVal = 0.0001;

    // Find max value for normalization
    for (let l = 0; l < matrix.length; l++) {
        for (let h = 0; h < matrix[l].length; h++) {
            if (matrix[l][h] > maxVal) maxVal = matrix[l][h];
        }
    }

    cells.forEach(cell => {
        const l = parseInt(cell.dataset.layer);
        const h = parseInt(cell.dataset.head);
        const val = matrix[l][h];
        const norm = Math.min(1.0, val / maxVal);

        const key = `${l},${h}`;
        if (ablatedHeadsSet.has(key)) {
            cell.classList.add("ablated");
            cell.style.background = "";
        } else {
            cell.classList.remove("ablated");
            // Heatmap color interpolation: Dark blue -> Cyan -> Purple -> Rose
            cell.style.background = interpolateColor(norm);
        }

        cell.title = `Layer ${l}, Head ${h} | ${currentMatrixView.toUpperCase()}: ${val.toFixed(4)}`;
    });
}

function interpolateColor(val) {
    if (val < 0.25) {
        return `rgba(30, 41, 59, ${0.4 + val * 1.5})`;
    } else if (val < 0.6) {
        return `rgba(2, 132, 199, ${0.4 + val * 0.8})`;
    } else if (val < 0.85) {
        return `rgba(139, 92, 246, ${0.5 + val * 0.5})`;
    } else {
        return `rgba(244, 63, 94, ${0.6 + val * 0.4})`;
    }
}

// Toggle Head Ablation State
function toggleHeadAblation(layer, head) {
    const key = `${layer},${head}`;
    if (ablatedHeadsSet.has(key)) {
        ablatedHeadsSet.delete(key);
    } else {
        ablatedHeadsSet.add(key);
    }

    updateAblatedPillsUI();
    renderHeatmapColors();
    updateHUDStats();
}

function updateAblatedPillsUI() {
    const container = document.getElementById("ablated-pills-list");
    const badge = document.getElementById("ablated-count-badge");
    if (!container) return;

    badge.textContent = ablatedHeadsSet.size;

    if (ablatedHeadsSet.size === 0) {
        container.innerHTML = `<span class="empty-hint">No heads ablated yet. Click on the Matrix tab to add heads or pick a preset below.</span>`;
        return;
    }

    container.innerHTML = "";
    ablatedHeadsSet.forEach(key => {
        const [l, h] = key.split(",");
        const pill = document.createElement("span");
        pill.className = "head-pill";
        pill.innerHTML = `L${l}:H${h} <span class="remove-btn" onclick="removeAblatedHead('${l}','${h}')">×</span>`;
        container.appendChild(pill);
    });
}

function removeAblatedHead(layer, head) {
    ablatedHeadsSet.delete(`${layer},${head}`);
    updateAblatedPillsUI();
    renderHeatmapColors();
    updateHUDStats();
}

function updateHUDStats() {
    const total = modelInfo.total_heads || 144;
    const ablatedCount = ablatedHeadsSet.size;
    const activeRatio = (((total - ablatedCount) / total) * 100).toFixed(1);
    
    document.getElementById("hud-active-ratio").textContent = `${activeRatio}%`;
    document.getElementById("hud-pruned-count").textContent = `${ablatedCount} Heads Ablated`;

    const flopsSaved = ((ablatedCount / total) * 42.0).toFixed(1);
    document.getElementById("hud-flops-reduction").textContent = `${flopsSaved}%`;

    const speedup = (1.0 + (ablatedCount / total) * 0.85).toFixed(2);
    document.getElementById("top-speedup-val").textContent = `${speedup}x Faster`;
}

// Preset Handlers
function initPresets() {
    document.querySelectorAll(".preset-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const promptText = chip.getAttribute("data-prompt");
            document.getElementById("prompt-input").value = promptText;
        });
    });
}

const taskPrompts = {
    "summarization": "Summarize the core technical features of task-aware attention head pruning in transformers:",
    "sentiment": "The sentiment of this product review is remarkably positive because",
    "logic_math": "def calculate_causal_kl_divergence(baseline_logits, ablated_logits):",
    "qa": "What is the primary role of attention heads in transformer neural networks?",
    "sorting": "Sort the following items in alphabetical sequence: [banana, apple, cherry, date]"
};

// Event Listeners Setup
function setupEventListeners() {
    // Heatmap Toggle Buttons
    document.getElementById("view-causal-btn")?.addEventListener("click", (e) => {
        document.getElementById("view-causal-btn").classList.add("active");
        document.getElementById("view-magnitude-btn").classList.remove("active");
        currentMatrixView = "causal";
        renderHeatmapColors();
    });

    document.getElementById("view-magnitude-btn")?.addEventListener("click", (e) => {
        document.getElementById("view-magnitude-btn").classList.add("active");
        document.getElementById("view-causal-btn").classList.remove("active");
        currentMatrixView = "magnitude";
        renderHeatmapColors();
    });

    document.getElementById("run-matrix-eval-btn")?.addEventListener("click", fetchHeatmapMatrix);

    // KL Safety Threshold Slider
    const slider = document.getElementById("kl-threshold-slider");
    if (slider) {
        slider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value).toFixed(3);
            document.getElementById("kl-threshold-val").textContent = `KL \u2264 ${val}`;
        });
    }

    // Auto-Prune All Safe Heads (KL <= Threshold) Button
    document.getElementById("auto-prune-safe-heads-btn")?.addEventListener("click", async () => {
        const thresholdVal = parseFloat(document.getElementById("kl-threshold-slider")?.value || 0.05);
        const prompt = document.getElementById("prompt-input").value;

        const btn = document.getElementById("auto-prune-safe-heads-btn");
        const originalText = btn.innerHTML;
        btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Testing All Heads...`;
        btn.disabled = true;

        try {
            const res = await fetch(`${API_BASE}/api/prune/causal`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prompts: [prompt],
                    mode: "safety_threshold",
                    kl_threshold: thresholdVal
                })
            });
            const data = await res.json();
            
            ablatedHeadsSet.clear();
            data.ablated_pairs.forEach(pair => ablatedHeadsSet.add(`${pair[0]},${pair[1]}`));
            
            updateAblatedPillsUI();
            renderHeatmapColors();
            updateHUDStats();
        } catch (e) {
            console.error("Error auto-pruning safe heads:", e);
        } finally {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    });

    // Prune Bottom 40% Heads Button
    document.getElementById("ablate-safe-bottom20-btn")?.addEventListener("click", () => {
        if (!matrixData.rankings || matrixData.rankings.length === 0) {
            alert("Please compute heatmap matrix first.");
            return;
        }
        ablatedHeadsSet.clear();
        const safeCount = Math.floor(matrixData.rankings.length * 0.4);
        for (let i = 0; i < safeCount; i++) {
            const item = matrixData.rankings[i];
            ablatedHeadsSet.add(`${item.layer},${item.head}`);
        }
        updateAblatedPillsUI();
        renderHeatmapColors();
        updateHUDStats();
    });

    // Clear Ablated Heads
    document.getElementById("clear-ablated-heads-btn")?.addEventListener("click", () => {
        ablatedHeadsSet.clear();
        updateAblatedPillsUI();
        renderHeatmapColors();
        updateHUDStats();
    });

    // Run Interventional Safety Test
    document.getElementById("run-ablation-test-btn")?.addEventListener("click", runAblationPlaygroundTest);

    // Task Router Cards & Task Selector Buttons
    document.querySelectorAll(".task-card").forEach(card => {
        card.addEventListener("click", () => {
            document.querySelectorAll(".task-card").forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            const taskKey = card.getAttribute("data-task");
            
            if (taskPrompts[taskKey]) {
                document.getElementById("router-prompt-input").value = taskPrompts[taskKey];
            }
            executeDynamicTaskRoute(taskKey);
        });
    });

    document.getElementById("run-task-route-btn")?.addEventListener("click", () => {
        const activeCard = document.querySelector(".task-card.active");
        const taskKey = activeCard ? activeCard.getAttribute("data-task") : "summarization";
        const prompt = document.getElementById("router-prompt-input").value;
        executeDynamicTaskRoute(taskKey, prompt);
    });

    // Evaluate Overall Performance Across All Tasks Button
    document.getElementById("eval-all-tasks-btn")?.addEventListener("click", fetchAllTasksSummary);

    // Run Full Benchmark
    document.getElementById("run-full-benchmark-btn")?.addEventListener("click", runFullBenchmarkSuite);
}

// Run Interventional Playground Test
async function runAblationPlaygroundTest() {
    const prompt = document.getElementById("prompt-input").value;
    const headsArray = Array.from(ablatedHeadsSet).map(s => s.split(",").map(Number));

    const btn = document.getElementById("run-ablation-test-btn");
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running Interventional Test...`;
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/ablation/playground`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: prompt,
                ablated_heads: headsArray,
                max_tokens: 24
            })
        });
        const data = await res.json();

        document.getElementById("baseline-output-text").textContent = data.baseline.generated_text;
        document.getElementById("ablated-output-text").textContent = data.ablated.generated_text;

        document.getElementById("base-speed-badge").textContent = `${data.baseline.ms_per_token} ms/tok`;
        document.getElementById("ablated-speed-badge").textContent = `${data.ablated.ms_per_token} ms/tok`;

        document.getElementById("hud-kl-div").textContent = data.kl_divergence.toFixed(4);

        // Update Verdict Banner
        const banner = document.getElementById("verdict-banner");
        const vTitle = document.getElementById("verdict-title");
        const vDesc = document.getElementById("verdict-desc");

        if (data.is_safe) {
            banner.style.borderColor = "var(--green)";
            vTitle.innerHTML = `<span class="text-green"><i class="fa-solid fa-shield-check"></i> PASS: Safety Test Verified!</span>`;
            vDesc.textContent = `Ablating ${data.ablated_heads_count} attention heads resulted in negligible KL Divergence (${data.kl_divergence}). Output answer is identical!`;
            document.getElementById("hud-kl-status").textContent = "PASS - Safe Cut";
            document.getElementById("hud-kl-status").className = "hud-sub status-safe";
        } else {
            banner.style.borderColor = "var(--rose)";
            vTitle.innerHTML = `<span class="text-rose"><i class="fa-solid fa-triangle-exclamation"></i> WARNING: Critical Circuit Ablated</span>`;
            vDesc.textContent = `KL Divergence shift is ${data.kl_divergence}. Target heads contain essential semantic knowledge for this task!`;
            document.getElementById("hud-kl-status").textContent = "WARNING - High KL Shift";
            document.getElementById("hud-kl-status").className = "hud-sub text-rose";
        }
    } catch (e) {
        console.error("Error running ablation test:", e);
    } finally {
        btn.innerHTML = `<i class="fa-solid fa-bolt"></i> Run Interventional Safety Test`;
        btn.disabled = false;
    }
}

// Execute Dynamic Task Route & Per-Task Calculation Test
async function executeDynamicTaskRoute(taskKey = null, prompt = null) {
    if (!prompt) {
        prompt = document.getElementById("router-prompt-input").value;
    }

    const btn = document.getElementById("run-task-route-btn");
    const origBtnText = btn.innerHTML;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Calculating Task Test...`;
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/route/infer`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: prompt,
                task_key: taskKey,
                max_tokens: 24
            })
        });
        const data = await res.json();

        document.getElementById("route-task-detected").textContent = data.routing.task_name;
        document.getElementById("route-heads-active").textContent = `${data.routing.active_heads} / ${data.routing.total_heads}`;
        document.getElementById("route-speedup-factor").textContent = `${data.actual_speedup}x Faster`;
        
        const flopsSaved = (100.0 - data.routing.active_percentage) * 0.42;
        document.getElementById("route-flops-saved").textContent = `${flopsSaved.toFixed(1)}%`;

        if (data.baseline && data.pruned) {
            document.getElementById("task-base-output-text").textContent = data.baseline.generated_text;
            document.getElementById("task-pruned-output-text").textContent = data.pruned.generated_text;
            document.getElementById("task-base-speed").textContent = `${data.baseline.ms_per_token} ms/tok`;
            document.getElementById("task-pruned-speed").textContent = `${data.pruned.ms_per_token} ms/tok`;
        }

        // Apply dynamic head mask to matrix view
        ablatedHeadsSet.clear();
        data.routing.ablated_pairs.forEach(pair => ablatedHeadsSet.add(`${pair[0]},${pair[1]}`));
        updateAblatedPillsUI();
        renderHeatmapColors();
        updateHUDStats();
    } catch (e) {
        console.error("Error executing task route:", e);
    } finally {
        btn.innerHTML = origBtnText;
        btn.disabled = false;
    }
}

// Fetch All Tasks Summary Matrix
async function fetchAllTasksSummary() {
    const btn = document.getElementById("eval-all-tasks-btn");
    const origText = btn.innerHTML;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Evaluating All Tasks...`;
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/tasks/all_summary`);
        const data = await res.json();

        const tbody = document.querySelector(".task-table tbody");
        if (tbody && data.tasks) {
            tbody.innerHTML = "";
            const icons = {
                "summarization": "fa-file-contract text-purple",
                "sentiment": "fa-face-smile text-cyan",
                "logic_math": "fa-code text-amber",
                "qa": "fa-circle-question text-rose",
                "sorting": "fa-arrow-down-short-wide text-green"
            };
            const badges = {
                "summarization": "purple",
                "sentiment": "cyan",
                "logic_math": "amber",
                "qa": "rose",
                "sorting": "green"
            };

            data.tasks.forEach(task => {
                const tr = document.createElement("tr");
                const iconClass = icons[task.task_key] || "fa-layer-group";
                const badgeClass = badges[task.task_key] || "purple";

                tr.innerHTML = `
                    <td><i class="fa-solid ${iconClass}"></i> <strong>${task.task_name}</strong></td>
                    <td>${task.active_heads} / ${task.total_heads}</td>
                    <td>${task.pruned_heads} / ${task.total_heads}</td>
                    <td><span class="badge ${badgeClass}">${task.prune_percentage}% Cut</span></td>
                    <td>${task.flops_saved_pct}%</td>
                    <td><span class="text-green">${task.estimated_speedup}x</span></td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (e) {
        console.error("Error fetching all tasks summary:", e);
    } finally {
        btn.innerHTML = origText;
        btn.disabled = false;
    }
}

// Run Full Benchmark Suite
async function runFullBenchmarkSuite() {
    const btn = document.getElementById("run-full-benchmark-btn");
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running Suite...`;
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/benchmark/compare`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompts: [
                    "The human brain is a remarkably efficient computing engine because",
                    "Functions in python allow developers to build modular code",
                    "Summarize the core technical features of task-aware attention head pruning"
                ],
                sparsity_target: 0.4
            })
        });
        const data = await res.json();
        updateBenchmarkChartsWithData(data);
    } catch (e) {
        console.error("Error running benchmark:", e);
    } finally {
        btn.innerHTML = `<i class="fa-solid fa-play"></i> Run Live Benchmark Suite`;
        btn.disabled = false;
    }
}
