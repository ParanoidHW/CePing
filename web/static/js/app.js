/**
 * LLM Performance Evaluator - Web Interface
 * Configuration sections: Cluster Topology, Model Config, Parallelism, Workload
 */

const CATEGORY_NAMES = {
    'strategy': '并行策略',
    'model': '模型规格',
    'sequence': '序列切分',
    'memory': '内存容量',
    'special': '特殊场景',
};

const state = {
    devices: {},
    devicePresets: {},
    modelPresets: {},
    topologyPresets: {},
    currentPipeline: null,
    videoParams: null,
    rlCurrentPhase: 'train',
    diffusionCurrentMode: 'T2I',
};

const elements = {
    modelPreset: document.getElementById('model-preset'),
    modelType: document.getElementById('model-type'),
    deviceVendor: document.getElementById('device-vendor'),
    deviceModel: document.getElementById('device-model'),
    totalDevices: document.getElementById('total-devices'),
    topologyType: document.getElementById('topology-type'),
    topologyParams: document.getElementById('topology-params'),
    tpDegree: document.getElementById('tp-degree'),
    ppDegree: document.getElementById('pp-degree'),
    vppDegree: document.getElementById('vpp-degree'),
    pipelineSchedule: document.getElementById('pipeline-schedule'),
    epDegree: document.getElementById('ep-degree'),
    dpDegreeDisplay: document.getElementById('dp-degree-display'),
    dpDegree: document.getElementById('dp-degree'),
    ulyssesDegree: document.getElementById('ulysses-degree'),
    ringDegree: document.getElementById('ring-degree'),
    megatronSpEnabled: document.getElementById('megatron-sp-enabled'),
    dpValidationError: document.getElementById('dp-validation-error'),
    evaluateBtn: document.getElementById('evaluate-btn'),
    results: document.getElementById('results'),
    resultsContent: document.getElementById('results-content'),
    workloadScenario: document.getElementById('workload-scenario'),
};

async function init() {
    await loadData();
    setupEventListeners();
    updateDeviceModels();
    updateTopologyParams();
    loadModelPreset();
    calculateDP();
    switchWorkloadScenario('training');
}

async function loadData() {
    try {
        const deviceRes = await fetch('/api/devices');
        const deviceData = await deviceRes.json();
        state.devices = deviceData.devices;
        state.devicePresets = deviceData.device_info;
        
        const initialWorkload = elements.workloadScenario ? elements.workloadScenario.value : 'training';
        await loadModelPresetsForWorkload(initialWorkload);
        
        const topoRes = await fetch('/api/topology/presets');
        state.topologyPresets = await topoRes.json();
    } catch (error) {
        console.error('Failed to load data:', error);
        showError('Failed to load configuration data. Please refresh.');
    }
}

async function loadModelPresetsForWorkload(workloadType) {
    try {
        const modelRes = await fetch(`/api/models?workload_type=${workloadType}`);
        state.modelPresets = await modelRes.json();
        populateModelPresets();
    } catch (error) {
        console.error('Failed to load model presets for workload:', error);
    }
}

function populateModelPresets() {
    const select = elements.modelPreset;
    select.innerHTML = '';
    
    const presets = state.modelPresets.presets || state.modelPresets;
    Object.keys(presets).forEach(key => {
        const preset = presets[key];
        
        const option = document.createElement('option');
        option.value = key;
        const desc = preset.description || preset.name || key;
        option.textContent = `${key} - ${desc}`;
        select.appendChild(option);
    });
}

function setupEventListeners() {
    elements.modelPreset.addEventListener('change', loadModelPreset);
    elements.modelType.addEventListener('change', updateModelTypeUI);
    elements.deviceVendor.addEventListener('change', updateDeviceModels);
    elements.topologyType.addEventListener('change', updateTopologyParams);
    elements.evaluateBtn.addEventListener('click', evaluate);
    
    elements.totalDevices.addEventListener('input', calculateDP);
    
    if (elements.workloadScenario) {
        elements.workloadScenario.addEventListener('change', async (e) => {
            const workloadType = e.target.value;
            await loadModelPresetsForWorkload(workloadType);
            switchWorkloadScenario(workloadType);
        });
    }
    
    const parallelismInputs = [
        elements.tpDegree, elements.ppDegree, elements.vppDegree,
        elements.epDegree, elements.ulyssesDegree, elements.ringDegree
    ];
    parallelismInputs.forEach(input => {
        if (input) {
            input.addEventListener('input', calculateDP);
        }
    });
}

function calculateDP() {
    const tp = parseInt(elements.tpDegree.value) || 1;
    const pp = parseInt(elements.ppDegree.value) || 1;
    const ep = parseInt(elements.epDegree.value) || 1;
    const ulysses = parseInt(elements.ulyssesDegree.value) || 1;
    const ring = parseInt(elements.ringDegree.value) || 1;
    const totalDevices = parseInt(elements.totalDevices.value) || 64;
    
    const product = tp * pp * ep * ulysses * ring;
    const dp = totalDevices / product;
    
    elements.dpValidationError.style.display = 'none';
    
    if (product > totalDevices) {
        elements.dpValidationError.innerHTML = `
            <span class="error-icon">❌</span>
            <span>并行度乘积 (${product}) > 总设备数 (${totalDevices})，无法计算 DP</span>
        `;
        elements.dpValidationError.style.display = 'block';
        elements.dpDegreeDisplay.textContent = 'N/A';
        elements.dpDegree.value = 0;
        return null;
    }
    
    if (!Number.isInteger(dp)) {
        elements.dpValidationError.innerHTML = `
            <span class="error-icon">⚠️</span>
            <span>DP = ${dp.toFixed(2)}（非整数），建议调整并行参数使 DP 为整数</span>
        `;
        elements.dpValidationError.style.display = 'block';
    }
    
    const dpInt = Math.floor(dp);
    elements.dpDegreeDisplay.textContent = dpInt;
    elements.dpDegree.value = dpInt;
    
    return dpInt;
}

function switchWorkloadScenario(scenario) {
    const allParams = document.querySelectorAll('.workload-params');
    allParams.forEach(el => el.style.display = 'none');
    
    const targetParams = document.getElementById(`workload-params-${scenario}`);
    if (targetParams) {
        targetParams.style.display = 'block';
    }
    
    const allParallelModes = document.querySelectorAll('.parallel-mode');
    allParallelModes.forEach(el => el.style.display = 'none');
    
    const parallelSingle = document.getElementById('parallel-single');
    const parallelPDDisagg = document.getElementById('parallel-pd-disagg');
    const parallelRLTraining = document.getElementById('parallel-rl-training');
    
    if (parallelSingle) parallelSingle.style.display = 'none';
    if (parallelPDDisagg) parallelPDDisagg.style.display = 'none';
    if (parallelRLTraining) parallelRLTraining.style.display = 'none';
    
    if (scenario === 'pd_disagg') {
        if (parallelPDDisagg) parallelPDDisagg.style.display = 'block';
        setupPDDisaggListeners();
        calculatePDDP('prefill');
        calculatePDDP('decode');
    } else if (scenario === 'rl_training') {
        if (parallelRLTraining) parallelRLTraining.style.display = 'block';
        setupRLPhaseTabs();
        calculateRLDP('train');
        calculateRLDP('infer');
    } else {
        if (parallelSingle) parallelSingle.style.display = 'block';
        calculateDP();
    }
    
    if (scenario === 'diffusion') {
        setupDiffusionModeTabs();
        updateDiffusionModeUI(state.diffusionCurrentMode);
    }
    
    updateMemoryOptsVisibility(scenario);
}

function updateMemoryOptsVisibility(scenario) {
    const recomputeEl = document.querySelector('.memory-opt-recompute');
    const zeroEl = document.querySelector('.memory-opt-zero');
    const sectionEl = document.querySelector('.memory-opt-section');
    
    const isTraining = scenario === 'training' || scenario === 'rl_training';
    
    if (recomputeEl) {
        recomputeEl.classList.toggle('memory-opts-hidden', !isTraining);
    }
    if (zeroEl) {
        zeroEl.classList.toggle('memory-opts-hidden', !isTraining);
    }
    if (sectionEl) {
        sectionEl.classList.toggle('memory-opts-hidden', !isTraining);
    }
}

function setupPDDisaggListeners() {
    const prefillInputs = ['pd-tp-prefill', 'pd-pp-prefill', 'pd-ep-prefill', 'pd-ulysses-prefill', 'pd-ring-prefill', 'prefill-devices'];
    const decodeInputs = ['pd-tp-decode', 'pd-pp-decode', 'pd-ep-decode', 'pd-ulysses-decode', 'pd-ring-decode', 'decode-devices'];
    
    prefillInputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', () => calculatePDDP('prefill'));
    });
    
    decodeInputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', () => calculatePDDP('decode'));
    });
}

function calculatePDDP(phase) {
    const tp = parseInt(document.getElementById(`pd-tp-${phase}`)?.value) || 1;
    const pp = parseInt(document.getElementById(`pd-pp-${phase}`)?.value) || 1;
    const ep = parseInt(document.getElementById(`pd-ep-${phase}`)?.value) || 1;
    const ulysses = parseInt(document.getElementById(`pd-ulysses-${phase}`)?.value) || 1;
    const ring = parseInt(document.getElementById(`pd-ring-${phase}`)?.value) || 1;
    const devices = parseInt(document.getElementById(`${phase === 'prefill' ? 'prefill' : 'decode'}-devices`)?.value) || 32;
    const product = tp * pp * ep * ulysses * ring;
    const dp = Math.floor(devices / product);
    const display = document.getElementById(`pd-dp-${phase}-display`);
    if (display) display.textContent = dp > 0 ? dp : 'N/A';
}

function setupRLPhaseTabs() {
    const tabs = document.querySelectorAll('#rl-phase-tabs .tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            tabs.forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            const phase = e.target.dataset.phase;
            state.rlCurrentPhase = phase;
            
            document.getElementById('rl-strategy-train').style.display = phase === 'train' ? 'block' : 'none';
            document.getElementById('rl-strategy-infer').style.display = phase === 'infer' ? 'block' : 'none';
        });
    });
    
    const trainInputs = ['rl-tp-train', 'rl-pp-train', 'rl-ep-train', 'rl-vpp-train', 'rl-ulysses-train', 'rl-ring-train'];
    trainInputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', () => calculateRLDP('train'));
    });
    
    const inferInputs = ['rl-tp-infer', 'rl-pp-infer', 'rl-ep-infer', 'rl-vpp-infer', 'rl-ulysses-infer', 'rl-ring-infer'];
    inferInputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', () => calculateRLDP('infer'));
    });
}

function calculateRLDP(phase) {
    const tp = parseInt(document.getElementById(`rl-tp-${phase}`)?.value) || 1;
    const pp = parseInt(document.getElementById(`rl-pp-${phase}`)?.value) || 1;
    const ep = parseInt(document.getElementById(`rl-ep-${phase}`)?.value) || 1;
    const ulysses = parseInt(document.getElementById(`rl-ulysses-${phase}`)?.value) || 1;
    const ring = parseInt(document.getElementById(`rl-ring-${phase}`)?.value) || 1;
    const totalDevices = parseInt(elements.totalDevices?.value) || 64;
    const product = tp * pp * ep * ulysses * ring;
    const dp = Math.floor(totalDevices / product);
    const display = document.getElementById(`rl-dp-${phase}-display`);
    if (display) display.textContent = dp > 0 ? dp : 'N/A';
}

function setupDiffusionModeTabs() {
    const tabs = document.querySelectorAll('#diffusion-mode-tabs .tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            tabs.forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            const mode = e.target.dataset.mode;
            state.diffusionCurrentMode = mode;
            updateDiffusionModeUI(mode);
        });
    });
}

function updateDiffusionModeUI(mode) {
    const promptGroup = document.getElementById('diffusion-prompt-group');
    const inputImageGroup = document.getElementById('diffusion-input-image-group');
    const inputVideoGroup = document.getElementById('diffusion-input-video-group');
    const outputImageGroup = document.getElementById('diffusion-output-image-group');
    const outputVideoGroup = document.getElementById('diffusion-output-video-group');
    
    if (promptGroup) promptGroup.style.display = (mode === 'T2I' || mode === 'T2V') ? 'block' : 'none';
    if (inputImageGroup) inputImageGroup.style.display = (mode === 'I2I' || mode === 'I2V') ? 'block' : 'none';
    if (inputVideoGroup) inputVideoGroup.style.display = mode === 'V2V' ? 'block' : 'none';
    if (outputImageGroup) outputImageGroup.style.display = (mode === 'T2I' || mode === 'I2I') ? 'block' : 'none';
    if (outputVideoGroup) outputVideoGroup.style.display = (mode === 'T2V' || mode === 'I2V' || mode === 'V2V') ? 'block' : 'none';
}

function loadModelPreset() {
    const presetKey = elements.modelPreset.value;
    const presets = state.modelPresets.presets || state.modelPresets;
    if (!presetKey || !presets[presetKey]) {
        updateModelSpecDisplay(null);
        return;
    }

    const preset = presets[presetKey];

    state.currentPreset = preset;
    const scenario = elements.workloadScenario?.value || 'training';
    state.currentPipeline = preset.architecture === 'wan_pipeline' && scenario !== 'training'
        ? 'diffusion-video' 
        : null;

    elements.modelType.value = preset.sparse_type || 'dense';

    document.getElementById('hidden-size').value = preset.hidden_size || 4096;
    document.getElementById('num-layers').value = preset.num_layers || 32;
    document.getElementById('num-heads').value = preset.num_heads || preset.num_attention_heads || 32;
    document.getElementById('max-seq-len').value = preset.max_seq_len || 4096;
    document.getElementById('dtype').value = preset.dtype || 'fp16';

    if (preset.num_experts) {
        document.getElementById('num-experts').value = preset.num_experts;
        document.getElementById('experts-per-token').value = preset.num_experts_per_token;
    }

    updateModelTypeUI();
    updateModelSpecDisplay(preset, presetKey);
}

function updateModelSpecDisplay(config, presetKey) {
    const displayEl = document.getElementById('model-spec-display');
    if (!displayEl) return;

    if (!config) {
        displayEl.innerHTML = '<p class="hint-text">选择预设后显示规格</p>';
        return;
    }

    const numKVHeads = config.num_kv_heads || config.num_heads || 32;
    const headDim = config.head_dim || (config.hidden_size ? config.hidden_size / config.num_heads : 128);
    const ffnType = config.num_experts ? 'MoE' : 'Dense';

    const compactRows = [
        ['参数', estimateModelParams(config)],
        ['层数', config.num_layers || '-'],
        ['隐藏层', config.hidden_size || '-'],
        ['头数', config.num_heads || '-'],
        ['KV头', numKVHeads],
        ['Head维', Math.round(headDim)],
        ['FFN', ffnType],
    ];

    if (config.num_experts) {
        compactRows.push(['专家', `${config.num_experts}/${config.num_experts_per_token || 8}`]);
    }

    const tableHtml = compactRows.map(([label, value]) => 
        `<tr><td class="spec-label-compact">${label}</td><td class="spec-value-compact">${value}</td></tr>`
    ).join('');

    displayEl.innerHTML = `
        <table class="spec-table-compact">
            ${tableHtml}
        </table>
    `;
}

function estimateModelParams(config) {
    if (!config.hidden_size || !config.num_layers) return 'N/A';
    
    const hs = config.hidden_size;
    const nl = config.num_layers;
    const nh = config.num_heads || 32;
    const nkv = config.num_kv_heads || nh;
    const intermediate = config.intermediate_size || hs * 4;
    const vocab = config.vocab_size || 32000;
    
    const embedParams = hs * vocab;
    const attnParams = hs * hs * (nh / nkv) * nl * 2 + hs * nkv * 2;
    const ffnParams = intermediate * hs * 2 * nl;
    
    const total = embedParams + attnParams + ffnParams;
    const totalB = total / 1e9;
    
    if (config.num_experts) {
        return `${totalB.toFixed(1)}B (${config.num_experts} experts)`;
    }
    
    return `${totalB.toFixed(1)}B`;
}

function updateModelTypeUI() {
    const isMoE = elements.modelType.value.includes('moe');
    const moeFields = document.querySelectorAll('.moe-only');
    moeFields.forEach(el => {
        el.style.display = isMoE ? 'block' : 'none';
    });
}

function updateDeviceModels() {
    const vendor = elements.deviceVendor.value;
    const models = state.devices[vendor] || [];
    
    elements.deviceModel.innerHTML = '';
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        elements.deviceModel.appendChild(option);
    });
}

function updateTopologyParams() {
    const type = elements.topologyType.value;
    const defaults = state.topologyPresets.defaults[type] || {};
    
    let html = '';
    
    switch(type) {
        case '2-Tier Simple':
            html = `
                <div class="form-row">
                    <div class="form-group">
                        <label>Intra-Node BW (GB/s)</label>
                        <input type="number" id="topo-intra" value="${defaults.intra_node_bw_gbps || 900}">
                    </div>
                    <div class="form-group">
                        <label>Inter-Node BW (GB/s)</label>
                        <input type="number" id="topo-inter" value="${defaults.inter_node_bw_gbps || 200}">
                    </div>
                </div>
            `;
            break;
        case '3-Tier Clos':
            html = `
                <div class="form-row">
                    <div class="form-group">
                        <label>Node BW (GB/s)</label>
                        <input type="number" id="topo-node" value="${defaults.node_bw_gbps || 900}">
                    </div>
                    <div class="form-group">
                        <label>Rack BW (Gbps)</label>
                        <input type="number" id="topo-rack" value="${defaults.rack_bw_gbps || 200}">
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Cluster BW (Gbps)</label>
                        <input type="number" id="topo-cluster" value="${defaults.cluster_bw_gbps || 100}">
                    </div>
                </div>
            `;
            break;
        case 'Fat-Tree':
            html = `
                <div class="form-row">
                    <div class="form-group">
                        <label>Edge BW (Gbps)</label>
                        <input type="number" id="topo-edge" value="${defaults.edge_bw_gbps || 800}">
                    </div>
                    <div class="form-group">
                        <label>Aggregation BW (Gbps)</label>
                        <input type="number" id="topo-agg" value="${defaults.agg_bw_gbps || 400}">
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Core BW (Gbps)</label>
                        <input type="number" id="topo-core" value="${defaults.core_bw_gbps || 100}">
                    </div>
                    <div class="form-group">
                        <label>Oversubscription</label>
                        <input type="number" id="topo-oversub" value="${defaults.oversubscription || 4}" step="0.5">
                    </div>
                </div>
            `;
            break;
        case 'CloudMatrix Supernode':
            html = `
                <div class="form-row">
                    <div class="form-group">
                        <label>Num NPUs</label>
                        <input type="number" id="topo-npus" value="${defaults.num_npus || 384}">
                    </div>
                    <div class="form-group">
                        <label>UB BW (Gbps)</label>
                        <input type="number" id="topo-ub" value="${defaults.ub_bw_gbps || 3136}">
                    </div>
                </div>
                <div class="form-group">
                    <label>RDMA BW (Gbps)</label>
                    <input type="number" id="topo-rdma" value="${defaults.rdma_bw_gbps || 400}">
                </div>
                <p class="help-text">CloudMatrix: 全对等超节点架构，统一高带宽</p>
            `;
            break;
    }
    
    elements.topologyParams.innerHTML = html;
}

function renderValidationErrors(validation) {
    if (!validation) return '';
    
    const errors = validation.errors || [];
    const warnings = validation.warnings || [];
    
    if (errors.length === 0 && warnings.length === 0) return '';
    
    const errorHtml = errors.map(e => `
        <div class="validation-error">
            <span class="error-icon">❌</span>
            <span class="error-category">${CATEGORY_NAMES[e.category] || e.category}</span>
            <span class="error-message">${e.message}</span>
            ${e.suggestion ? `<div class="error-suggestion">建议: ${e.suggestion}</div>` : ''}
        </div>
    `).join('');
    
    const warningHtml = warnings.map(e => `
        <div class="validation-warning">
            <span class="warning-icon">⚠️</span>
            <span class="warning-category">${CATEGORY_NAMES[e.category] || e.category}</span>
            <span class="warning-message">${e.message}</span>
            ${e.suggestion ? `<div class="warning-suggestion">建议: ${e.suggestion}</div>` : ''}
        </div>
    `).join('');
    
    return `
        <div class="validation-container">
            ${errorHtml}
            ${warningHtml}
        </div>
    `;
}

function validateConfigBeforeSubmit() {
    const tp = parseInt(elements.tpDegree.value) || 1;
    const pp = parseInt(elements.ppDegree.value) || 1;
    const dp = parseInt(elements.dpDegree.value) || 1;
    const ep = parseInt(elements.epDegree.value) || 1;
    const ulysses = parseInt(elements.ulyssesDegree.value) || 1;
    const ring = parseInt(elements.ringDegree.value) || 1;
    const totalDevices = parseInt(elements.totalDevices.value) || 64;
    
    const errors = [];
    
    const product = tp * pp * dp * ep * ulysses * ring;
    if (product !== totalDevices) {
        errors.push({
            level: 'error',
            category: 'strategy',
            message: `并行度乘积 (${product}) ≠ 总设备数 (${totalDevices})`,
            suggestion: `调整并行度使 TP×PP×DP×EP×Ulysses×Ring = ${totalDevices}`,
        });
    }
    
    if (dp < 1) {
        errors.push({
            level: 'error',
            category: 'strategy',
            message: `DP 必须为 ≥ 1 的整数`,
            suggestion: '调整其他并行参数使 DP 为正整数',
        });
    }
    
    return errors;
}

function showValidationErrors(errors) {
    elements.results.style.display = 'block';
    elements.resultsContent.innerHTML = `
        <div class="validation-container">
            ${errors.map(e => `
                <div class="validation-error">
                    <span class="error-icon">❌</span>
                    <span class="error-category">${CATEGORY_NAMES[e.category] || e.category}</span>
                    <span class="error-message">${e.message}</span>
                    ${e.suggestion ? `<div class="error-suggestion">建议: ${e.suggestion}</div>` : ''}
                </div>
            `).join('')}
        </div>
    `;
}

async function evaluate() {
    const frontendErrors = validateConfigBeforeSubmit();
    if (frontendErrors.length > 0) {
        showValidationErrors(frontendErrors);
        return;
    }
    
    const btn = elements.evaluateBtn;
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.spinner');
    
    btn.disabled = true;
    btnText.textContent = '评估中...';
    spinner.style.display = 'inline-block';
    
    try {
        const config = collectConfig();
        
        let endpoint;
        const scenario = config.workload?.scenario || 'training';
        
        if (state.currentPipeline === 'diffusion-video') {
            endpoint = '/api/evaluate/pipeline/diffusion-video';
        } else {
            endpoint = '/api/evaluate';
        }
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        const data = await response.json();
        
        if (!data.success) {
            if (data.validation) {
                const validationHtml = renderValidationErrors(data.validation);
                elements.results.style.display = 'block';
                elements.resultsContent.innerHTML = `
                    <div class="error-message">
                        <strong>Error:</strong> ${data.error || '评估失败'}
                    </div>
                    ${validationHtml}
                `;
            } else {
                showError(data.error || 'Evaluation failed');
            }
            return;
        }
        
        displayResults(data.result);
        
        if (data.validation) {
            const validationHtml = renderValidationErrors(data.validation);
            const currentContent = elements.resultsContent.innerHTML;
            elements.resultsContent.innerHTML = validationHtml + currentContent;
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        btn.disabled = false;
        btnText.textContent = '开始评估';
        spinner.style.display = 'none';
    }
}

function collectConfig() {
    const topologyType = elements.topologyType.value;
    let topology = { type: topologyType };
    
    switch(topologyType) {
        case '2-Tier Simple':
            topology.intra_node_bw_gbps = parseFloat(document.getElementById('topo-intra').value);
            topology.inter_node_bw_gbps = parseFloat(document.getElementById('topo-inter').value);
            break;
        case '3-Tier Clos':
            topology.node_bw_gbps = parseFloat(document.getElementById('topo-node').value);
            topology.rack_bw_gbps = parseFloat(document.getElementById('topo-rack').value);
            topology.cluster_bw_gbps = parseFloat(document.getElementById('topo-cluster').value);
            break;
        case 'Fat-Tree':
            topology.edge_bw_gbps = parseFloat(document.getElementById('topo-edge').value);
            topology.agg_bw_gbps = parseFloat(document.getElementById('topo-agg').value);
            topology.core_bw_gbps = parseFloat(document.getElementById('topo-core').value);
            topology.oversubscription = parseFloat(document.getElementById('topo-oversub').value);
            break;
        case 'CloudMatrix Supernode':
            topology.num_npus = parseInt(document.getElementById('topo-npus').value);
            topology.ub_bw_gbps = parseFloat(document.getElementById('topo-ub').value);
            topology.rdma_bw_gbps = parseFloat(document.getElementById('topo-rdma').value);
            break;
    }
    
    const presetKey = elements.modelPreset.value;
    const presets = state.modelPresets.presets || state.modelPresets;
    const preset = presets[presetKey] || {};
    
    const scenario = elements.workloadScenario ? elements.workloadScenario.value : 'training';
    
    const baseConfig = {
        cluster: {
            topology: topologyType,
            total_devices: parseInt(elements.totalDevices.value),
        },
        model: {
            preset: presetKey,
            type: preset?.architecture || 'llama',
            ...preset,
            sparse_type: elements.modelType.value,
            hidden_size: parseInt(document.getElementById('hidden-size').value),
            num_layers: parseInt(document.getElementById('num-layers').value),
            num_attention_heads: parseInt(document.getElementById('num-heads').value),
            dtype: document.getElementById('dtype').value,
        },
        device: elements.deviceModel.value,
        num_devices: parseInt(elements.totalDevices.value),
        topology: topology,
        workload: collectWorkloadConfig(scenario),
    };
    
    if (scenario === 'pd_disagg') {
        baseConfig.strategy_prefill = {
            tp: parseInt(document.getElementById('pd-tp-prefill')?.value || 8),
            pp: parseInt(document.getElementById('pd-pp-prefill')?.value || 1),
            ep: parseInt(document.getElementById('pd-ep-prefill')?.value || 1),
            ulysses_degree: parseInt(document.getElementById('pd-ulysses-prefill')?.value || 1),
            ring_degree: parseInt(document.getElementById('pd-ring-prefill')?.value || 1),
        };
        baseConfig.strategy_decode = {
            tp: parseInt(document.getElementById('pd-tp-decode')?.value || 4),
            pp: parseInt(document.getElementById('pd-pp-decode')?.value || 1),
            ep: parseInt(document.getElementById('pd-ep-decode')?.value || 1),
            ulysses_degree: parseInt(document.getElementById('pd-ulysses-decode')?.value || 1),
            ring_degree: parseInt(document.getElementById('pd-ring-decode')?.value || 1),
        };
    } else if (scenario === 'rl_training') {
        baseConfig.strategy_train = {
            tp: parseInt(document.getElementById('rl-tp-train')?.value || 8),
            pp: parseInt(document.getElementById('rl-pp-train')?.value || 1),
            vpp: parseInt(document.getElementById('rl-vpp-train')?.value || 1),
            ep: parseInt(document.getElementById('rl-ep-train')?.value || 1),
            ulysses_degree: parseInt(document.getElementById('rl-ulysses-train')?.value || 1),
            ring_degree: parseInt(document.getElementById('rl-ring-train')?.value || 1),
        };
        baseConfig.strategy_infer = {
            tp: parseInt(document.getElementById('rl-tp-infer')?.value || 4),
            pp: parseInt(document.getElementById('rl-pp-infer')?.value || 1),
            vpp: parseInt(document.getElementById('rl-vpp-infer')?.value || 1),
            ep: parseInt(document.getElementById('rl-ep-infer')?.value || 1),
            ulysses_degree: parseInt(document.getElementById('rl-ulysses-infer')?.value || 1),
            ring_degree: parseInt(document.getElementById('rl-ring-infer')?.value || 1),
        };
    } else {
        baseConfig.strategy = {
            tp: parseInt(elements.tpDegree.value),
            pp: parseInt(elements.ppDegree.value),
            vpp: parseInt(elements.vppDegree?.value || 1),
            pipeline_schedule: elements.pipelineSchedule?.value || '1f1b',
            dp: parseInt(elements.dpDegree.value),
            ep: parseInt(elements.epDegree.value),
            ulysses_degree: parseInt(elements.ulyssesDegree.value),
            ring_degree: parseInt(elements.ringDegree.value),
            megatron_sp_enabled: elements.megatronSpEnabled.checked,
            activation_checkpointing: document.getElementById('activation-checkpointing').checked,
            zero_stage: parseInt(document.getElementById('zero-stage').value)
        };
    }
    
    return baseConfig;
}

function collectWorkloadConfig(scenario) {
    const workload = { scenario };
    
    switch(scenario) {
        case 'training':
            workload.batch_size = parseInt(document.getElementById('train-batch-size')?.value || 32);
            workload.micro_batch_size = parseInt(document.getElementById('micro-batch-size')?.value || 1);
            workload.seq_len = parseInt(document.getElementById('train-seq-len')?.value || 4096);
            workload.num_steps = parseInt(document.getElementById('num-steps')?.value || 1000);
            break;
        case 'inference':
            workload.batch_size = parseInt(document.getElementById('infer-batch-size')?.value || 1);
            workload.input_tokens = parseInt(document.getElementById('infer-input-tokens')?.value || 1000);
            workload.output_tokens = parseInt(document.getElementById('infer-output-tokens')?.value || 100);
            break;
        case 'pd_disagg':
            workload.batch_size = parseInt(document.getElementById('pd-batch-size')?.value || 1);
            workload.input_tokens = parseInt(document.getElementById('pd-input-tokens')?.value || 1000);
            workload.output_tokens = parseInt(document.getElementById('pd-output-tokens')?.value || 100);
            workload.prefill_devices = parseInt(document.getElementById('prefill-devices')?.value || 32);
            workload.decode_devices = parseInt(document.getElementById('decode-devices')?.value || 32);
            break;
        case 'rl_training':
            workload.batch_size = parseInt(document.getElementById('rl-batch-size')?.value || 32);
            workload.seq_len = parseInt(document.getElementById('rl-seq-len')?.value || 4096);
            workload.num_rollouts = parseInt(document.getElementById('num-rollouts')?.value || 100);
            workload.ppo_epochs = parseInt(document.getElementById('ppo-epochs')?.value || 4);
            break;
        case 'diffusion':
            workload.batch_size = parseInt(document.getElementById('diffusion-batch-size')?.value || 1);
            workload.diffusion_steps = parseInt(document.getElementById('diffusion-steps')?.value || 50);
            workload.generation_mode = state.diffusionCurrentMode || 'T2I';
            
            if (workload.generation_mode === 'T2I' || workload.generation_mode === 'T2V') {
                workload.prompt_tokens = parseInt(document.getElementById('diffusion-prompt-tokens')?.value || 512);
            }
            if (workload.generation_mode === 'I2I' || workload.generation_mode === 'I2V') {
                workload.input_image_width = parseInt(document.getElementById('diffusion-input-image-width')?.value || 1024);
                workload.input_image_height = parseInt(document.getElementById('diffusion-input-image-height')?.value || 1024);
            }
            if (workload.generation_mode === 'V2V') {
                workload.input_video_width = parseInt(document.getElementById('diffusion-input-video-width')?.value || 1920);
                workload.input_video_height = parseInt(document.getElementById('diffusion-input-video-height')?.value || 1080);
                workload.input_video_frames = parseInt(document.getElementById('diffusion-input-video-frames')?.value || 16);
            }
            if (workload.generation_mode === 'T2I' || workload.generation_mode === 'I2I') {
                workload.output_image_width = parseInt(document.getElementById('diffusion-output-image-width')?.value || 1024);
                workload.output_image_height = parseInt(document.getElementById('diffusion-output-image-height')?.value || 1024);
            }
            if (workload.generation_mode === 'T2V' || workload.generation_mode === 'I2V' || workload.generation_mode === 'V2V') {
                workload.output_video_width = parseInt(document.getElementById('diffusion-output-video-width')?.value || 1920);
                workload.output_video_height = parseInt(document.getElementById('diffusion-output-video-height')?.value || 1080);
                workload.output_video_frames = parseInt(document.getElementById('diffusion-output-video-frames')?.value || 24);
            }
            break;
    }
    
    return workload;
}

function displayResults(result) {
    elements.results.style.display = 'block';
    
    const scenarioType = result.scenario_type || result.workload_type || 'training';
    
    const RENDER_TEMPLATES = {
        'diffusion': renderDiffusionSummary,
        'training': renderTrainingSummary,
        'rl_training': renderTrainingSummary,
        'inference': renderInferenceSummary,
        'pd_disagg': renderInferenceSummary,
    };
    
    const template = RENDER_TEMPLATES[scenarioType];
    if (template) {
        template(result);
    } else {
        renderGenericSummary(result);
    }
    
    elements.results.scrollIntoView({ behavior: 'smooth' });
}

function renderDiffusionSummary(result) {
    const phases = result.phases || [];
    const totalTime = result.total_time_sec || 0;
    
    const denoisePhase = phases.find(p => p.name === 'denoise') || {};
    const encodePhase = phases.find(p => p.name === 'encode') || {};
    const decodePhase = phases.find(p => p.name === 'decode') || {};
    
    const denoisePct = totalTime > 0 ? (denoisePhase.total_time_sec || 0) / totalTime * 100 : 0;
    const encodePct = totalTime > 0 ? (encodePhase.total_time_sec || 0) / totalTime * 100 : 0;
    const decodePct = totalTime > 0 ? (decodePhase.total_time_sec || 0) / totalTime * 100 : 0;
    
    const params = result.params || {};
    const diffusionSteps = denoisePhase.repeat_count || params.num_steps || 50;
    const stepTime = (denoisePhase.total_time_sec || 0) / diffusionSteps;
    
    const pixelsPerStep = params.height && params.width 
        ? params.height * params.width 
        : (params.num_frames || 81) * (params.height || 720) * (params.width || 1280);
    const throughput = pixelsPerStep > 0 ? pixelsPerStep / totalTime : 0;
        
        elements.resultsContent.innerHTML = `
            <div class="result-grid">
                <div class="result-card highlight">
                    <div class="result-value">${totalTime.toFixed(1)}s</div>
                    <div class="result-label">Total Time</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${(stepTime * 1000).toFixed(0)}ms</div>
                    <div class="result-label">Step Time</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${diffusionSteps}</div>
                    <div class="result-label">Diffusion Steps</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${(result.peak_memory_gb || 0).toFixed(1)}GB</div>
                    <div class="result-label">Peak Memory</div>
                </div>
            </div>
            <div class="result-grid" style="margin-top: 1rem;">
                <div class="result-card">
                    <div class="result-value">${(throughput / 1000000).toFixed(2)}M</div>
                    <div class="result-label">Pixels/sec</div>
                </div>
            </div>
            <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">耗时分解</h3>
            <table class="breakdown-table">
                <tr>
                    <th>阶段</th>
                    <th>时间</th>
                    <th>占比</th>
                </tr>
                ${encodePhase.total_time_sec > 0 ? `
                <tr>
                    <td>Encode</td>
                    <td>${((encodePhase.total_time_sec || 0) * 1000).toFixed(2)} ms</td>
                    <td>${encodePct.toFixed(1)}%</td>
                </tr>
                ` : ''}
                <tr>
                    <td>Denoise (${diffusionSteps} steps)</td>
                    <td>${(denoisePhase.total_time_sec || 0).toFixed(2)} s</td>
                    <td>${denoisePct.toFixed(1)}%</td>
                </tr>
                ${decodePhase.total_time_sec > 0 ? `
                <tr>
                    <td>Decode</td>
                    <td>${((decodePhase.total_time_sec || 0) * 1000).toFixed(2)} ms</td>
                    <td>${decodePct.toFixed(1)}%</td>
                </tr>
                ` : ''}
            </table>
            ${params.height && params.width ? `
            <div style="margin-top: 1rem; padding: 1rem; background: var(--gray-100); border-radius: 8px;">
                <strong>生成配置:</strong> ${params.height}x${params.width} | Steps: ${diffusionSteps} | CFG: ${params.use_cfg ? '启用' : '禁用'}
            </div>
            ` : params.num_frames ? `
            <div style="margin-top: 1rem; padding: 1rem; background: var(--gray-100); border-radius: 8px;">
                <strong>生成配置:</strong> ${params.num_frames}帧 @ ${params.height || 720}x${params.width || 1280} | CFG: ${params.use_cfg ? '启用' : '禁用'}
            </div>
            ` : ''}
            ${renderDetailedBreakdown(result.detailed_breakdown)}
        `;
}

function renderTrainingSummary(result) {
    const detailed = result.detailed_breakdown;
    const detailedHtml = renderDetailedBreakdown(detailed);
    
    elements.resultsContent.innerHTML = `
        <div class="result-grid">
            <div class="result-card highlight">
                <div class="result-value">${(result.throughput?.tokens_per_sec / 1000 || 0).toFixed(2)}K</div>
                <div class="result-label">Tokens/sec</div>
            </div>
            <div class="result-card">
                <div class="result-value">${(result.throughput?.samples_per_sec || 0).toFixed(2)}</div>
                <div class="result-label">Samples/sec</div>
            </div>
            <div class="result-card">
                <div class="result-value">${((result.time?.time_per_step_sec || 0) * 1000).toFixed(0)}ms</div>
                <div class="result-label">Time/Step</div>
            </div>
            <div class="result-card">
                <div class="result-value">${(result.memory?.memory_per_device_gb || 0).toFixed(1)}GB</div>
                <div class="result-label">Memory/device</div>
            </div>
        </div>
        ${renderBreakdown(result.breakdown)}
        ${detailedHtml}
    `;
}

function renderInferenceSummary(result) {
    const detailed = result.detailed_breakdown;
    const detailedHtml = renderDetailedBreakdown(detailed);
    
    elements.resultsContent.innerHTML = `
        <div class="result-grid">
            <div class="result-card highlight">
                <div class="result-value">${(result.decode?.tps || 0).toFixed(0)}</div>
                <div class="result-label">Decode TPS</div>
            </div>
            <div class="result-card">
                <div class="result-value">${((result.prefill?.ttft_sec || 0) * 1000).toFixed(0)}ms</div>
                <div class="result-label">TTFT (Time To First Token)</div>
            </div>
            <div class="result-card">
                <div class="result-value">${((result.decode?.tpot_sec || 0) * 1000).toFixed(1)}ms</div>
                <div class="result-label">TPOT (Time Per Output Token)</div>
            </div>
            <div class="result-card">
                <div class="result-value">${(result.end_to_end?.overall_tps || 0).toFixed(0)}</div>
                <div class="result-label">Overall TPS</div>
            </div>
        </div>
        <div class="result-grid" style="margin-top: 1rem;">
            <div class="result-card">
                <div class="result-value">${(result.end_to_end?.total_time_sec || 0).toFixed(2)}s</div>
                <div class="result-label">Total Time</div>
            </div>
            <div class="result-card">
                <div class="result-value">${(result.memory?.memory_per_device_gb || 0).toFixed(1)}GB</div>
                <div class="result-label">Memory/device</div>
            </div>
            <div class="result-card">
                <div class="result-value">${(result.memory?.kv_cache_gb || 0).toFixed(1)}GB</div>
                <div class="result-label">KV Cache</div>
            </div>
        </div>
        ${renderInferenceBreakdown(result.breakdown)}
        ${detailedHtml}
    `;
}

function renderGenericSummary(result) {
    const detailedHtml = renderDetailedBreakdown(result.detailed_breakdown);
    
    elements.resultsContent.innerHTML = `
        <div class="result-grid">
            <div class="result-card highlight">
                <div class="result-value">${(result.total_time_sec || 0).toFixed(2)}s</div>
                <div class="result-label">Total Time</div>
            </div>
            <div class="result-card">
                <div class="result-value">${(result.peak_memory_gb || 0).toFixed(1)}GB</div>
                <div class="result-label">Peak Memory</div>
            </div>
        </div>
        ${detailedHtml}
    `;
}

function renderDetailedBreakdown(detailed) {
    if (!detailed) return '';

    const memByType = detailed.memory?.by_type || {};
    const { total, ...breakdownItems } = memByType;

    const orderedPriority = ['weight', 'gradient', 'optimizer', 'activation'];
    const allTypes = Object.keys(breakdownItems).filter(k => !k.startsWith('total') && !k.startsWith('params'));
    const orderedTypes = [
        ...orderedPriority.filter(t => breakdownItems[t] !== undefined),
        ...allTypes.filter(t => !orderedPriority.includes(t) && breakdownItems[t] !== undefined)
    ];
    const memRows = orderedTypes
        .map(type => `<tr><td>${type}</td><td>${breakdownItems[type].toFixed(2)} GB</td></tr>`)
        .join('');

    const totalRow = total !== undefined 
        ? `<tr style="font-weight: bold; border-top: 2px solid var(--gray-200);">
             <td>总计</td><td>${total.toFixed(2)} GB</td></tr>` 
        : '';

    const memBySubmodel = detailed.memory?.by_submodel || {};
    const submodelMemRows = Object.entries(memBySubmodel)
        .map(([name, mems]) => {
            const total = mems.activations_gb || 0;
            return `<tr><td>${name}</td><td>${total.toFixed(2)} GB</td></tr>`;
        }).join('');

    const bySubmoduleType = detailed.by_submodule_type || {};
    
    const totalComputeTime = Object.values(bySubmoduleType)
        .reduce((sum, d) => sum + (d.compute?.time_sec || 0), 0);
    const totalCommTime = Object.values(bySubmoduleType)
        .reduce((sum, d) => sum + (d.communication?.time_sec || 0), 0);
    const totalMemoryGb = Object.values(bySubmoduleType)
        .reduce((sum, d) => sum + (d.memory?.activations_gb || 0), 0);
    
    let submoduleBreakdownRows = '';
    for (const [submoduleType, data] of Object.entries(bySubmoduleType)) {
        const memGb = data.memory?.activations_gb || 0;
        const computeSec = data.compute?.time_sec || 0;
        const commTimeSec = data.communication?.time_sec || 0;
        const computePct = totalComputeTime > 0 ? (computeSec / totalComputeTime * 100) : 0;
        const commPct = totalCommTime > 0 ? (commTimeSec / totalCommTime * 100) : 0;
        
        submoduleBreakdownRows += `<tr>
            <td style="font-weight: bold;">${submoduleType}</td>
            <td>${(computeSec * 1000).toFixed(3)} ms</td>
            <td>${computePct.toFixed(1)}%</td>
            <td>${(commTimeSec * 1000).toFixed(3)} ms</td>
            <td>${commPct.toFixed(1)}%</td>
            <td>${memGb.toFixed(2)} GB</td>
        </tr>`;
        
        if (data.nested_breakdown) {
            for (const [nestedType, nestedData] of Object.entries(data.nested_breakdown)) {
                const nestedMemGb = nestedData.memory?.activations_gb || 0;
                const nestedComputeSec = nestedData.compute?.time_sec || 0;
                const nestedCommTimeSec = nestedData.communication?.time_sec || 0;
                const nestedComputePct = totalComputeTime > 0 ? (nestedComputeSec / totalComputeTime * 100) : 0;
                const nestedCommPct = totalCommTime > 0 ? (nestedCommTimeSec / totalCommTime * 100) : 0;
                
                submoduleBreakdownRows += `<tr style="background: var(--gray-50);">
                    <td style="padding-left: 1.5rem;">${nestedType}</td>
                    <td>${(nestedComputeSec * 1000).toFixed(3)} ms</td>
                    <td>${nestedComputePct.toFixed(1)}%</td>
                    <td>${(nestedCommTimeSec * 1000).toFixed(3)} ms</td>
                    <td>${nestedCommPct.toFixed(1)}%</td>
                    <td>${nestedMemGb.toFixed(2)} GB</td>
                </tr>`;
            }
        }
    }
    
    const summaryRow = (totalComputeTime > 0 || totalCommTime > 0) 
        ? `<tr class="summary-row">
            <td><strong>总计</strong></td>
            <td>${(totalComputeTime * 1000).toFixed(3)} ms</td>
            <td>100%</td>
            <td>${(totalCommTime * 1000).toFixed(3)} ms</td>
            <td>100%</td>
            <td>${totalMemoryGb.toFixed(2)} GB</td>
        </tr>` 
        : '';
    
    const totalTime = totalComputeTime + totalCommTime;
    const computeOverallPct = totalTime > 0 ? (totalComputeTime / totalTime * 100) : 0;
    const commOverallPct = totalTime > 0 ? (totalCommTime / totalTime * 100) : 0;
    
    const progressBarHtml = totalTime > 0 ? `
        <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">时间占比</h3>
        <div class="time-summary-row" style="margin-bottom: 0.5rem; font-size: 0.875rem; color: var(--gray-600);">
            <span>总时间: ${(totalTime * 1000).toFixed(1)} ms</span>
            <span style="margin-left: 1rem;">计算: ${(totalComputeTime * 1000).toFixed(1)} ms</span>
            <span style="margin-left: 1rem;">通信: ${(totalCommTime * 1000).toFixed(1)} ms</span>
        </div>
        <div class="time-breakdown-bar">
            <div class="compute-bar" style="width: ${computeOverallPct}%;">
                <span class="bar-label">计算: ${computeOverallPct.toFixed(1)}%</span>
            </div>
            <div class="comm-bar" style="width: ${commOverallPct}%;">
                <span class="bar-label">通信: ${commOverallPct.toFixed(1)}%</span>
            </div>
        </div>
    ` : '';

    const commByPara = detailed.communication?.by_parallelism || {};
    const commRows = Object.entries(commByPara)
        .map(([type, data]) => {
            const totalGb = (data.total_bytes || 0) / 1e9;
            const totalMs = (data.total_time_sec || 0) * 1000;
            return `<tr><td>${type.toUpperCase()}</td><td>${totalGb.toFixed(2)} GB</td><td>${totalMs.toFixed(2)} ms</td></tr>`;
        }).join('');

    const submodelDetails = (detailed.submodels || []).map(sm => {
        const summary = sm.memory?.summary || {};
        const memTypes = Object.entries(summary)
            .filter(([t]) => t.endsWith('_gb'))
            .map(([t, v]) => `${t.replace('_gb', '')}: ${v.toFixed(1)}G`)
            .join(', ');
        return `
            <div style="margin: 0.5rem 0; padding: 0.5rem; background: var(--gray-50); border-radius: 4px;">
                <strong>${sm.model_name}</strong> (${sm.model_type})<br>
                Memory: ${memTypes || '无数据'}
            </div>
        `;
    }).join('');

    return `
        <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">详细内存分解 (按类型)</h3>
        <table class="breakdown-table">
            <tr><th>内存类型</th><th>大小</th></tr>
            ${memRows}${totalRow || '<tr><td colspan="2">无数据</td></tr>'}
        </table>

        ${progressBarHtml}
        
        <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">子模块分解 (按类型)</h3>
        <table class="breakdown-table">
            <tr>
                <th>子模块类型</th>
                <th>计算时间 (ms)</th>
                <th>计算占比 (%)</th>
                <th>通信时间 (ms)</th>
                <th>通信占比 (%)</th>
                <th>内存 (GB)</th>
            </tr>
            ${submoduleBreakdownRows || '<tr><td colspan="6">无数据</td></tr>'}
            ${summaryRow}
        </table>

        <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">内存分解 (按子模型)</h3>
        <table class="breakdown-table">
            <tr><th>子模型</th><th>总内存</th></tr>
            ${submodelMemRows || '<tr><td colspan="2">无数据</td></tr>'}
        </table>

        <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">通信分解 (按并行方式)</h3>
        <table class="breakdown-table">
            <tr><th>并行类型</th><th>通信量</th><th>时间</th></tr>
            ${commRows || '<tr><td colspan="3">无通信数据</td></tr>'}
        </table>

        <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">通信分解 (按通信原语)</h3>
        <table class="breakdown-table">
            <tr><th>原语类型</th><th>通信量 (GB)</th><th>时间 (ms)</th></tr>
            ${Object.entries(detailed.communication?.by_operation || {})
                .map(([type, data]) => {
                    const totalGb = (data.total_bytes || 0) / 1e9;
                    const totalMs = (data.total_time_sec || 0) * 1000;
                    let rows = `<tr><td><b>${type}</b></td><td>${totalGb.toFixed(2)}</td><td>${totalMs.toFixed(2)}</td></tr>`;
                    if (data.by_ptype) {
                        for (const [ptype, pdata] of Object.entries(data.by_ptype)) {
                            const pGb = (pdata.total_bytes || 0) / 1e9;
                            const pMs = (pdata.total_time_sec || 0) * 1000;
                            rows += `<tr style="background-color: #f5f5f5;"><td>${type} (${ptype.toUpperCase()})</td><td>${pGb.toFixed(2)}</td><td>${pMs.toFixed(2)}</td></tr>`;
                        }
                    }
                    return rows;
                }).join('') || '<tr><td colspan="3">无数据</td></tr>'}
        </table>

        <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">子模型详情</h3>
        ${submodelDetails || '<p>无子模型详情</p>'}
    `;
}

function renderBreakdown(breakdown) {
    if (!breakdown || !breakdown.time_breakdown) return '';
    
    const tb = breakdown.time_breakdown;
    const priorityCategories = [
        {key: 'compute_sec', name: 'Compute'},
        {key: 'backward_sec', name: 'Backward'},
        {key: 'optimizer_sec', name: 'Optimizer'},
        {key: 'communication_sec', name: 'Communication'},
        {key: 'memory_sec', name: 'Memory Wait'},
    ];
    
    const allSecKeys = Object.keys(tb).filter(k => k.endsWith('_sec'));
    const additionalCategories = allSecKeys
        .filter(k => !priorityCategories.find(c => c.key === k))
        .map(k => ({key: k, name: k.replace('_sec', '').replace(/_/g, ' ')}));
    
    const categories = [...priorityCategories, ...additionalCategories];
    
    const rows = categories.map(cat => {
        const sec = tb[cat.key] || 0;
        const percentKey = cat.key.replace('_sec', '_percent');
        const percent = tb[percentKey] || 0;
        return `<tr>
            <td>${cat.name}</td>
            <td>${(sec * 1000).toFixed(1)} ms</td>
            <td>${percent.toFixed(1)}%</td>
        </tr>`;
    }).join('');
    
    return `
        <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">性能分解</h3>
        <table class="breakdown-table">
            <tr>
                <th>类别</th>
                <th>时间</th>
                <th>占比</th>
            </tr>
            ${rows}
        </table>
    `;
}

function renderInferenceBreakdown(breakdown) {
    if (!breakdown || !breakdown.inference_breakdown) return '';
    
    const inf = breakdown.inference_breakdown;
    const priorityCategories = [
        {key: 'prefill_sec', name: 'Prefill', showPercent: true},
        {key: 'decode_sec', name: 'Decode', showPerToken: true},
        {key: 'communication_sec', name: 'Communication', showPercent: true},
    ];
    
    const allSecKeys = Object.keys(inf).filter(k => k.endsWith('_sec'));
    const additionalCategories = allSecKeys
        .filter(k => !priorityCategories.find(c => c.key === k) && inf[k] > 0)
        .map(k => ({key: k, name: k.replace('_sec', '').replace(/_/g, ' '), showPercent: true}));
    
    const categories = [...priorityCategories, ...additionalCategories];
    
    const rows = categories.map(cat => {
        const sec = inf[cat.key] || 0;
        const percentKey = cat.key.replace('_sec', '_percent');
        const percent = inf[percentKey] || 0;
        
        let name = cat.name;
        if (cat.showPerToken && inf.decode_per_token_sec) {
            name = `${cat.name} (${inf.decode_per_token_sec.toFixed(3)} ms/token)`;
        }
        
        return `<tr>
            <td>${name}</td>
            <td>${(sec * 1000 || 0).toFixed(1)} ms</td>
            <td>${percent.toFixed(1)}%</td>
        </tr>`;
    }).join('');
    
    return `
        <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">性能分解</h3>
        <table class="breakdown-table">
            <tr>
                <th>类别</th>
                <th>时间</th>
                <th>占比</th>
            </tr>
            ${rows}
        </table>
    `;
}

function showError(message) {
    elements.results.style.display = 'block';
    elements.resultsContent.innerHTML = `
        <div class="error-message">
            <strong>Error:</strong> ${message}
        </div>
    `;
}

document.addEventListener('DOMContentLoaded', init);