/**
 * LLM Performance Evaluator - Web Interface
 * Flat, minimal UI controller
 */

// ==================== State ====================
const state = {
    mode: 'training', // 'training' | 'inference'
    devices: {},
    devicePresets: {},
    modelPresets: {},
    topologyPresets: {},
    currentPipeline: null, // 'diffusion-video' for video generation
    videoParams: null
};

// ==================== DOM Elements ====================
const elements = {
    modeTabs: document.querySelectorAll('.tab'),
    modelPreset: document.getElementById('model-preset'),
    modelType: document.getElementById('model-type'),
    advancedToggle: document.querySelector('.advanced-toggle button'),
    advancedParams: document.querySelector('.advanced-params'),
    deviceVendor: document.getElementById('device-vendor'),
    deviceModel: document.getElementById('device-model'),
    topologyType: document.getElementById('topology-type'),
    topologyParams: document.getElementById('topology-params'),
    evaluateBtn: document.getElementById('evaluate-btn'),
    results: document.getElementById('results'),
    resultsContent: document.getElementById('results-content'),
    trainingParams: document.getElementById('training-params'),
    inferenceParams: document.getElementById('inference-params'),
    runtimeTitle: document.getElementById('runtime-title')
};

// ==================== Initialization ====================
async function init() {
    await loadData();
    setupEventListeners();
    updateDeviceModels();
    updateTopologyParams();
}

async function loadData() {
    try {
        // Load devices
        const deviceRes = await fetch('/api/devices');
        const deviceData = await deviceRes.json();
        state.devices = deviceData.devices;
        state.devicePresets = deviceData.device_info;
        
        // Load model presets
        const modelRes = await fetch('/api/model/presets');
        state.modelPresets = await modelRes.json();
        populateModelPresets();
        
        // Load topology presets
        const topoRes = await fetch('/api/topology/presets');
        state.topologyPresets = await topoRes.json();
    } catch (error) {
        console.error('Failed to load data:', error);
        showError('Failed to load configuration data. Please refresh.');
    }
}

function populateModelPresets() {
    const select = elements.modelPreset;
    select.innerHTML = '';
    
    const presets = state.modelPresets.presets || state.modelPresets;
    Object.keys(presets).forEach(key => {
        const preset = presets[key];
        const presetType = preset.preset_type || 'model';
        
        if (state.mode === 'inference' && presetType === 'component') {
            return;
        }
        
        const option = document.createElement('option');
        option.value = key;
        const desc = preset.description || preset.name || key;
        option.textContent = `${key} - ${desc}`;
        select.appendChild(option);
    });
}

// ==================== Event Listeners ====================
function setupEventListeners() {
    // Mode tabs
    elements.modeTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            elements.modeTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            state.mode = tab.dataset.mode;
            populateModelPresets();
            updateModeUI();
        });
    });
    
    // Model preset selection
    elements.modelPreset.addEventListener('change', loadModelPreset);
    
    // Model type toggle
    elements.modelType.addEventListener('change', updateModelTypeUI);
    
    // Advanced params toggle
    elements.advancedToggle.addEventListener('click', toggleAdvanced);
    
    // Device vendor change
    elements.deviceVendor.addEventListener('change', updateDeviceModels);
    
    // Topology type change
    elements.topologyType.addEventListener('change', updateTopologyParams);
    
    // Evaluate button
    elements.evaluateBtn.addEventListener('click', evaluate);
}

// ==================== UI Updates ====================
function updateModeUI() {
    if (state.mode === 'training') {
        elements.trainingParams.style.display = 'block';
        elements.inferenceParams.style.display = 'none';
        elements.runtimeTitle.textContent = '训练参数';
    } else {
        elements.trainingParams.style.display = 'none';
        elements.inferenceParams.style.display = 'block';
        elements.runtimeTitle.textContent = '推理参数';
    }
    elements.results.style.display = 'none';
    renderParamInputs();
}

function renderParamInputs() {
    const preset = state.currentPreset;
    const schema = preset?.param_schema;
    const defaultLLMSchema = {
        training: [
            {name: 'batch_size', label: 'Batch Size', type: 'number', default: 32},
            {name: 'seq_len', label: 'Sequence Length', type: 'number', default: 4096},
        ],
        inference: [
            {name: 'batch_size', label: 'Batch Size', type: 'number', default: 8},
            {name: 'prompt_len', label: 'Prompt Length', type: 'number', default: 1024},
            {name: 'generation_len', label: 'Generation Length', type: 'number', default: 128},
        ]
    };
    
    const currentSchema = schema || defaultLLMSchema;
    const modeSchema = currentSchema[state.mode] || [];
    
    const container = state.mode === 'training' ? elements.trainingParams : elements.inferenceParams;
    
    let html = '';
    const paramsPerRow = 2;
    
    for (let i = 0; i < modeSchema.length; i += paramsPerRow) {
        html += '<div class="form-row">';
        for (let j = i; j < Math.min(i + paramsPerRow, modeSchema.length); j++) {
            const param = modeSchema[j];
            html += `<div class="form-group">
                <label>${param.label}</label>
                ${renderParamInput(param)}
            </div>`;
        }
        html += '</div>';
    }
    
    container.innerHTML = html;
}

function renderParamInput(param) {
    const inputId = `param-${param.name}`;
    
    if (param.type === 'select') {
        const options = (param.options || []).map(opt => 
            `<option value="${opt}" ${opt === param.default ? 'selected' : ''}>${opt}</option>`
        ).join('');
        return `<select id="${inputId}">${options}</select>`;
    }
    
    const minAttr = param.min ? ` min="${param.min}"` : '';
    const maxAttr = param.max ? ` max="${param.max}"` : '';
    return `<input type="number" id="${inputId}" value="${param.default}"${minAttr}${maxAttr}>`;
}

function loadModelPreset() {
    const presetKey = elements.modelPreset.value;
    const presets = state.modelPresets.presets || state.modelPresets;
    if (!presetKey || !presets[presetKey]) return;

    const preset = presets[presetKey];

    // Store current preset and pipeline info
    state.currentPreset = preset;
    state.currentPipeline = preset.architecture === 'wan_pipeline' && state.mode === 'inference' 
        ? 'diffusion-video' 
        : null;

    // Set model type based on sparse_type
    elements.modelType.value = preset.sparse_type || 'dense';

    // Set model config params
    document.getElementById('hidden-size').value = preset.hidden_size || 4096;
    document.getElementById('num-layers').value = preset.num_layers || 32;
    document.getElementById('num-heads').value = preset.num_heads || preset.num_attention_heads || 32;
    document.getElementById('max-seq-len').value = preset.max_seq_len || 4096;
    document.getElementById('dtype').value = preset.dtype || 'fp16';

    // MoE params
    if (preset.num_experts) {
        document.getElementById('num-experts').value = preset.num_experts;
        document.getElementById('experts-per-token').value = preset.num_experts_per_token;
    }

    updateModelTypeUI();
    renderParamInputs();
}

function updateModelTypeUI() {
    const isMoE = elements.modelType.value.includes('moe');
    const moeFields = document.querySelectorAll('.moe-only');
    moeFields.forEach(el => {
        el.style.display = isMoE ? 'block' : 'none';
    });
}

function toggleAdvanced() {
    elements.advancedParams.classList.toggle('collapsed');
    const btn = elements.advancedToggle;
    btn.textContent = elements.advancedParams.classList.contains('collapsed') 
        ? '展开高级参数 ▼' 
        : '收起高级参数 ▲';
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

// ==================== Evaluation ====================
async function evaluate() {
    const btn = elements.evaluateBtn;
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.spinner');
    
    // Show loading state
    btn.disabled = true;
    btnText.textContent = '评估中...';
    spinner.style.display = 'inline-block';
    
    try {
        const config = collectConfig();
        
        // Use pipeline endpoint for video generation models
        let endpoint;
        if (state.currentPipeline === 'diffusion-video') {
            if (state.mode === 'training') {
                endpoint = '/api/evaluate/training';
            } else {
                endpoint = '/api/evaluate/pipeline/diffusion-video';
            }
        } else {
            endpoint = state.mode === 'training' 
                ? '/api/evaluate/training' 
                : '/api/evaluate/inference';
        }
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data.result);
        } else {
            showError(data.error || 'Evaluation failed');
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
    
    // Collect topology params
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
    
    const config = {
        model: {
            ...preset,
            sparse_type: elements.modelType.value,
            hidden_size: parseInt(document.getElementById('hidden-size').value),
            num_layers: parseInt(document.getElementById('num-layers').value),
            num_attention_heads: parseInt(document.getElementById('num-heads').value),
            dtype: document.getElementById('dtype').value,
        },
        device: elements.deviceModel.value,
        num_devices: parseInt(document.getElementById('num-devices').value),
        devices_per_node: parseInt(document.getElementById('devices-per-node').value),
        topology: topology,
        strategy: {
            tp: parseInt(document.getElementById('tp-degree').value),
            pp: parseInt(document.getElementById('pp-degree').value),
            dp: parseInt(document.getElementById('dp-degree').value),
            ep: parseInt(document.getElementById('ep-degree').value),
            activation_checkpointing: document.getElementById('activation-checkpointing').checked,
            zero_stage: parseInt(document.getElementById('zero-stage').value)
        }
    };
    
    // Collect mode-specific params from param_schema
    const schema = preset.param_schema;
    const modeParams = schema?.[state.mode] || [];
    
    if (state.mode === 'training') {
        config.training = { num_steps: 1000 };
        modeParams.forEach(param => {
            const inputEl = document.getElementById(`param-${param.name}`);
            if (inputEl) {
                const value = param.type === 'select' ? inputEl.value : parseInt(inputEl.value);
                config.training[param.name] = value;
            }
        });
    } else {
        config.inference = {};
        modeParams.forEach(param => {
            const inputEl = document.getElementById(`param-${param.name}`);
            if (inputEl) {
                const value = param.type === 'select' ? inputEl.value === 'true' : parseInt(inputEl.value);
                config.inference[param.name] = value;
            }
        });
    }
    
    return config;
}

function displayResults(result) {
    elements.results.style.display = 'block';
    
    // DEBUG: Log raw data for troubleshooting
    console.log('=== DEBUG: Raw API Result ===');
    console.log('result:', JSON.stringify(result, null, 2).substring(0, 2000));
    
    const detailed = result.detailed_breakdown;
    if (detailed && detailed.memory) {
        console.log('=== DEBUG: Memory by_type ===');
        console.log('by_type:', detailed.memory.by_type);
        console.log('weight:', detailed.memory.by_type?.weight);
        console.log('by_submodule_type keys:', Object.keys(detailed.memory.by_submodule_type || {}));
        console.log('by_submodule_type weights:', 
            Object.entries(detailed.memory.by_submodule_type || {})
                .map(([k, v]) => `${k}: ${v.memory?.weight_gb?.toFixed(2)} GB`)
                .join(', ')
        );
    }
    
    // Handle pipeline results (e.g., video generation)
    if (state.currentPipeline === 'diffusion-video') {
        const phases = result.phases || [];
        const totalTime = result.total_time_sec || 0;
        
        // 从phases提取各组件时间
        const encodePhase = phases.find(p => p.name === 'encode') || {};
        const denoisePhase = phases.find(p => p.name === 'denoise') || {};
        const decodePhase = phases.find(p => p.name === 'decode') || {};
        
        // 计算占比
        const encodePct = (encodePhase.total_time_sec || 0) / totalTime * 100;
        const denoisePct = (denoisePhase.total_time_sec || 0) / totalTime * 100;
        const decodePct = (decodePhase.total_time_sec || 0) / totalTime * 100;
        
        // 从params获取配置信息
        const params = result.params || {};
        
        elements.resultsContent.innerHTML = `
            <div class="result-grid">
                <div class="result-card highlight">
                    <div class="result-value">${(totalTime).toFixed(1)}s</div>
                    <div class="result-label">Total Time</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${(result.peak_memory_gb || 0).toFixed(1)}GB</div>
                    <div class="result-label">Peak Memory</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${((result.throughput?.pixels_per_sec || 0) / 1000000).toFixed(2)}M</div>
                    <div class="result-label">Pixels/sec</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${denoisePhase.repeat_count || params.num_inference_steps || 50}</div>
                    <div class="result-label">Inference Steps</div>
                </div>
            </div>
            <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">组件耗时分解</h3>
            <table class="breakdown-table">
                <tr>
                    <th>组件</th>
                    <th>时间</th>
                    <th>占比</th>
                </tr>
                <tr>
                    <td>Text Encoder (encode)</td>
                    <td>${((encodePhase.total_time_sec || 0) * 1000).toFixed(2)} ms</td>
                    <td>${encodePct.toFixed(3)}%</td>
                </tr>
                <tr>
                    <td>DiT Denoising (${denoisePhase.repeat_count || params.num_inference_steps || 50} steps)</td>
                    <td>${(denoisePhase.total_time_sec || 0).toFixed(2)} s</td>
                    <td>${denoisePct.toFixed(1)}%</td>
                </tr>
                <tr>
                    <td>VAE Decoder (decode)</td>
                    <td>${((decodePhase.total_time_sec || 0) * 1000).toFixed(2)} ms</td>
                    <td>${decodePct.toFixed(3)}%</td>
                </tr>
            </table>
            <div style="margin-top: 1rem; padding: 1rem; background: var(--gray-100); border-radius: 8px;">
                <strong>生成配置:</strong> ${params.num_frames || 81}帧 @ ${params.height || 720}x${params.width || 1280} | CFG: ${params.use_cfg ? '启用' : '禁用'}
            </div>
        `;
        elements.results.scrollIntoView({ behavior: 'smooth' });
        return;
    }
    
    if (state.mode === 'training') {
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
                    <div class="result-value">${(result.memory?.memory_per_gpu_gb || 0).toFixed(1)}GB</div>
                    <div class="result-label">Memory/GPU</div>
                </div>
            </div>
            ${renderBreakdown(result.breakdown)}
            ${detailedHtml}
        `;
    } else {
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
                    <div class="result-value">${(result.memory?.memory_per_gpu_gb || 0).toFixed(1)}GB</div>
                    <div class="result-label">Memory/GPU</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${(result.memory?.kv_cache_gb || 0).toFixed(1)}GB</div>
                    <div class="result-label">KV Cache</div>
                </div>
            </div>
            ${detailedHtml}
        `;
    }
    
    elements.results.scrollIntoView({ behavior: 'smooth' });
}

function renderDetailedBreakdown(detailed) {
    if (!detailed) return '';

    // DEBUG: Log memory data for troubleshooting
    console.log('=== DEBUG: renderDetailedBreakdown ===');
    console.log('detailed.memory.by_type:', detailed.memory?.by_type);
    console.log('detailed.memory.by_submodule_type:', detailed.memory?.by_submodule_type);
    console.log('detailed.by_submodule_type:', detailed.by_submodule_type);

    // Memory breakdown by type - separate total from breakdown items
    const memByType = detailed.memory?.by_type || {};
    const { total, ...breakdownItems } = memByType;
    
    console.log('=== DEBUG: memByType extracted ===');
    console.log('total:', total);
    console.log('breakdownItems:', breakdownItems);
    console.log('breakdownItems.weight:', breakdownItems.weight);

    // Render breakdown items in fixed order
    const orderedTypes = ['weight', 'gradient', 'optimizer', 'activation'];
    const memRows = orderedTypes
        .filter(type => breakdownItems[type] !== undefined)
        .map(type => `<tr><td>${type}</td><td>${breakdownItems[type].toFixed(2)} GB</td></tr>`)
        .join('');
    
    console.log('=== DEBUG: memRows HTML ===');
    console.log(memRows);

    // Total row with visual distinction
    const totalRow = total !== undefined 
        ? `<tr style="font-weight: bold; border-top: 2px solid var(--gray-200);">
             <td>总计</td><td>${total.toFixed(2)} GB</td></tr>` 
        : '';

    // Memory breakdown by submodel
    const memBySubmodel = detailed.memory?.by_submodel || {};
    const submodelMemRows = Object.entries(memBySubmodel)
        .map(([name, mems]) => {
            const total = mems.activations_gb || 0;
            return `<tr><td>${name}</td><td>${total.toFixed(2)} GB</td></tr>`;
        }).join('');

    // Unified breakdown by submodule type with nested breakdown
    const bySubmoduleType = detailed.by_submodule_type || {};
    
    let submoduleBreakdownRows = '';
    for (const [submoduleType, data] of Object.entries(bySubmoduleType)) {
        const memGb = data.memory?.activations_gb || 0;
        const computeTflops = (data.compute?.flops || 0) / 1e12;
        const computeSec = data.compute?.time_sec || 0;
        const commGb = data.communication?.gb || 0;
        
        // Main row for submodule type
        submoduleBreakdownRows += `<tr>
            <td style="font-weight: bold;">${submoduleType}</td>
            <td>${memGb.toFixed(2)} GB</td>
            <td>${computeTflops.toFixed(2)} T</td>
            <td>${computeSec.toFixed(3)} s</td>
            <td>${commGb.toFixed(2)} GB</td>
        </tr>`;
        
        // Nested rows (attention, ffn) for transformer_block
        if (data.nested_breakdown) {
            for (const [nestedType, nestedData] of Object.entries(data.nested_breakdown)) {
                const nestedMemGb = nestedData.memory?.activations_gb || 0;
                const nestedComputeTflops = (nestedData.compute?.flops || 0) / 1e12;
                const nestedComputeSec = nestedData.compute?.time_sec || 0;
                const nestedCommGb = nestedData.communication?.gb || 0;
                
                submoduleBreakdownRows += `<tr style="background: var(--gray-50);">
                    <td style="padding-left: 1.5rem;">${nestedType}</td>
                    <td>${nestedMemGb.toFixed(2)} GB</td>
                    <td>${nestedComputeTflops.toFixed(2)} T</td>
                    <td>${nestedComputeSec.toFixed(3)} s</td>
                    <td>${nestedCommGb.toFixed(2)} GB</td>
                </tr>`;
            }
        }
    }

    // Communication breakdown by parallelism type
    const commByPara = detailed.communication?.by_parallelism || {};
    const commRows = Object.entries(commByPara)
        .map(([type, data]) => {
            const totalGb = (data.total_bytes || 0) / 1e9;
            const totalMs = (data.total_time_sec || 0) * 1000;
            return `<tr><td>${type.toUpperCase()}</td><td>${totalGb.toFixed(2)} GB</td><td>${totalMs.toFixed(2)} ms</td></tr>`;
        }).join('');

    // Submodel details - read from summary, not by_type
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

    // DEBUG panel - show raw memory data for troubleshooting
    const debugPanel = `
        <details style="margin: 1.5rem 0; background: var(--gray-100); padding: 0.5rem; border-radius: 4px;">
            <summary style="cursor: pointer; font-weight: bold; color: #666;">
                🔍 Debug: 原始内存数据 (点击展开)
            </summary>
            <pre style="margin: 0.5rem 0; font-size: 12px; overflow: auto; max-height: 300px;">
memory.by_type:
${JSON.stringify(detailed.memory?.by_type, null, 2)}

memory.by_submodule_type (weight_gb):
${JSON.stringify(
    Object.fromEntries(
        Object.entries(detailed.memory?.by_submodule_type || {})
            .map(([k, v]) => [k, {weight_gb: v.memory?.weight_gb}])
    ), null, 2
)}
            </pre>
        </details>
    `;

    return `
        ${debugPanel}
        <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">详细内存分解 (按类型)</h3>
        <table class="breakdown-table">
            <tr><th>内存类型</th><th>大小</th></tr>
            ${memRows}${totalRow || '<tr><td colspan="2">无数据</td></tr>'}
        </table>

        <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">子模块分解 (按类型)</h3>
        <table class="breakdown-table">
            <tr><th>子模块类型</th><th>内存</th><th>计算量</th><th>计算时间</th><th>通信量</th></tr>
            ${submoduleBreakdownRows || '<tr><td colspan="5">无数据</td></tr>'}
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

        <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">子模型详情</h3>
        ${submodelDetails || '<p>无子模型详情</p>'}
    `;
}

function renderBreakdown(breakdown) {
    if (!breakdown) return '';
    
    return `
        <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">性能分解</h3>
        <table class="breakdown-table">
            <tr>
                <th>类别</th>
                <th>时间</th>
                <th>占比</th>
            </tr>
            <tr>
                <td>Compute</td>
                <td>${(breakdown.time_breakdown?.compute_sec * 1000 || 0).toFixed(1)} ms</td>
                <td>${(breakdown.time_breakdown?.compute_percent || 0).toFixed(1)}%</td>
            </tr>
            <tr>
                <td>Communication</td>
                <td>${(breakdown.time_breakdown?.communication_sec * 1000 || 0).toFixed(1)} ms</td>
                <td>${(breakdown.time_breakdown?.communication_percent || 0).toFixed(1)}%</td>
            </tr>
            <tr>
                <td>Memory Wait</td>
                <td>${(breakdown.time_breakdown?.memory_sec * 1000 || 0).toFixed(1)} ms</td>
                <td>${(breakdown.time_breakdown?.memory_percent || 0).toFixed(1)}%</td>
            </tr>
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

// ==================== Start ====================
document.addEventListener('DOMContentLoaded', init);
