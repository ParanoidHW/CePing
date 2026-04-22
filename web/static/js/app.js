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
    mode: 'training',
    devices: {},
    devicePresets: {},
    modelPresets: {},
    topologyPresets: {},
    currentPipeline: null,
    videoParams: null
};

const elements = {
    modeTabs: document.querySelectorAll('.tab'),
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
        
        const modelRes = await fetch('/api/model/presets');
        state.modelPresets = await modelRes.json();
        populateModelPresets();
        
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

function setupEventListeners() {
    elements.modeTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            elements.modeTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            state.mode = tab.dataset.mode;
            populateModelPresets();
            updateModeUI();
        });
    });
    
    elements.modelPreset.addEventListener('change', loadModelPreset);
    elements.modelType.addEventListener('change', updateModelTypeUI);
    elements.deviceVendor.addEventListener('change', updateDeviceModels);
    elements.topologyType.addEventListener('change', updateTopologyParams);
    elements.evaluateBtn.addEventListener('click', evaluate);
    
    elements.totalDevices.addEventListener('input', calculateDP);
    
    if (elements.workloadScenario) {
        elements.workloadScenario.addEventListener('change', (e) => {
            switchWorkloadScenario(e.target.value);
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
}

function updateModeUI() {
    elements.results.style.display = 'none';
    renderParamInputs();
}

function renderParamInputs() {
}

function loadModelPreset() {
    const presetKey = elements.modelPreset.value;
    const presets = state.modelPresets.presets || state.modelPresets;
    if (!presetKey || !presets[presetKey]) return;

    const preset = presets[presetKey];

    state.currentPreset = preset;
    state.currentPipeline = preset.architecture === 'wan_pipeline' && state.mode === 'inference' 
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
    renderParamInputs();
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
            const endpointMap = {
                'training': '/api/evaluate/training',
                'inference_prefill': '/api/evaluate/inference',
                'inference_decode': '/api/evaluate/inference',
                'pd_disagg': '/api/evaluate/inference',
                'rl_training': '/api/evaluate/training',
                'diffusion': '/api/evaluate/inference',
            };
            endpoint = endpointMap[scenario] || '/api/evaluate/training';
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
    
    const config = {
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
        strategy: {
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
        },
        workload: collectWorkloadConfig(scenario),
    };
    
    return config;
}

function collectWorkloadConfig(scenario) {
    const workload = { scenario };
    
    switch(scenario) {
        case 'training':
            workload.global_batch_size = parseInt(document.getElementById('global-batch-size')?.value || 32);
            workload.micro_batch_size = parseInt(document.getElementById('micro-batch-size')?.value || 1);
            workload.seq_len = parseInt(document.getElementById('train-seq-len')?.value || 4096);
            workload.num_steps = parseInt(document.getElementById('num-steps')?.value || 1000);
            break;
        case 'inference_prefill':
            workload.input_tokens = parseInt(document.getElementById('prefill-input-tokens')?.value || 1000);
            workload.output_tokens = parseInt(document.getElementById('prefill-output-tokens')?.value || 100);
            break;
        case 'inference_decode':
            workload.input_tokens = parseInt(document.getElementById('decode-input-tokens')?.value || 1000);
            workload.output_tokens = parseInt(document.getElementById('decode-output-tokens')?.value || 100);
            break;
        case 'pd_disagg':
            workload.prefill_devices = parseInt(document.getElementById('prefill-devices')?.value || 32);
            workload.decode_devices = parseInt(document.getElementById('decode-devices')?.value || 32);
            workload.input_tokens = parseInt(document.getElementById('pd-input-tokens')?.value || 1000);
            workload.output_tokens = parseInt(document.getElementById('pd-output-tokens')?.value || 100);
            break;
        case 'rl_training':
            workload.batch_size = parseInt(document.getElementById('rl-batch-size')?.value || 32);
            workload.seq_len = parseInt(document.getElementById('rl-seq-len')?.value || 4096);
            workload.num_rollouts = parseInt(document.getElementById('num-rollouts')?.value || 100);
            workload.ppo_epochs = parseInt(document.getElementById('ppo-epochs')?.value || 4);
            break;
        case 'diffusion':
            workload.image_size = parseInt(document.getElementById('image-size')?.value || 1024);
            workload.diffusion_steps = parseInt(document.getElementById('diffusion-steps')?.value || 50);
            workload.batch_size = parseInt(document.getElementById('diffusion-batch-size')?.value || 1);
            break;
    }
    
    return workload;
}

function displayResults(result) {
    elements.results.style.display = 'block';
    
    if (state.currentPipeline === 'diffusion-video') {
        const phases = result.phases || [];
        const totalTime = result.total_time_sec || 0;
        
        const encodePhase = phases.find(p => p.name === 'encode') || {};
        const denoisePhase = phases.find(p => p.name === 'denoise') || {};
        const decodePhase = phases.find(p => p.name === 'decode') || {};
        
        const encodePct = (encodePhase.total_time_sec || 0) / totalTime * 100;
        const denoisePct = (denoisePhase.total_time_sec || 0) / totalTime * 100;
        const decodePct = (decodePhase.total_time_sec || 0) / totalTime * 100;
        
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
                    <div class="result-value">${(result.memory?.memory_per_device_gb || 0).toFixed(1)}GB</div>
                    <div class="result-label">Memory/device</div>
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
                    <div class="result-value">${(result.memory?.memory_per_device_gb || 0).toFixed(1)}GB</div>
                    <div class="result-label">Memory/device</div>
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

    const memByType = detailed.memory?.by_type || {};
    const { total, ...breakdownItems } = memByType;

    const orderedTypes = ['weight', 'gradient', 'optimizer', 'activation'];
    const memRows = orderedTypes
        .filter(type => breakdownItems[type] !== undefined)
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
    
    let submoduleBreakdownRows = '';
    for (const [submoduleType, data] of Object.entries(bySubmoduleType)) {
        const memGb = data.memory?.activations_gb || 0;
        const computeTflops = (data.compute?.flops || 0) / 1e12;
        const computeSec = data.compute?.time_sec || 0;
        const commGb = data.communication?.gb || 0;
        const commTimeMs = (data.communication?.time_sec || 0) * 1000;
        
        submoduleBreakdownRows += `<tr>
            <td style="font-weight: bold;">${submoduleType}</td>
            <td>${memGb.toFixed(2)} GB</td>
            <td>${computeTflops.toFixed(2)} T</td>
            <td>${computeSec.toFixed(3)} s</td>
            <td>${commGb.toFixed(2)} GB</td>
            <td>${commTimeMs.toFixed(2)} ms</td>
        </tr>`;
        
        if (data.nested_breakdown) {
            for (const [nestedType, nestedData] of Object.entries(data.nested_breakdown)) {
                const nestedMemGb = nestedData.memory?.activations_gb || 0;
                const nestedComputeTflops = (nestedData.compute?.flops || 0) / 1e12;
                const nestedComputeSec = nestedData.compute?.time_sec || 0;
                const nestedCommGb = nestedData.communication?.gb || 0;
                const nestedCommTimeMs = (nestedData.communication?.time_sec || 0) * 1000;
                
                submoduleBreakdownRows += `<tr style="background: var(--gray-50);">
                    <td style="padding-left: 1.5rem;">${nestedType}</td>
                    <td>${nestedMemGb.toFixed(2)} GB</td>
                    <td>${nestedComputeTflops.toFixed(2)} T</td>
                    <td>${nestedComputeSec.toFixed(3)} s</td>
                    <td>${nestedCommGb.toFixed(2)} GB</td>
                    <td>${nestedCommTimeMs.toFixed(2)} ms</td>
                </tr>`;
            }
        }
    }

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

        <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem; color: var(--gray-700);">子模块分解 (按类型)</h3>
        <table class="breakdown-table">
            <tr><th>子模块类型</th><th>内存</th><th>计算量</th><th>计算时间</th><th>通信量</th><th>通信时间</th></tr>
            ${submoduleBreakdownRows || '<tr><td colspan="6">无数据</td></tr>'}
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
                <td>Backward</td>
                <td>${(breakdown.time_breakdown?.backward_sec * 1000 || 0).toFixed(1)} ms</td>
                <td>${(breakdown.time_breakdown?.backward_percent || 0).toFixed(1)}%</td>
            </tr>
            <tr>
                <td>Optimizer</td>
                <td>${(breakdown.time_breakdown?.optimizer_sec * 1000 || 0).toFixed(1)} ms</td>
                <td>${(breakdown.time_breakdown?.optimizer_percent || 0).toFixed(1)}%</td>
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

document.addEventListener('DOMContentLoaded', init);