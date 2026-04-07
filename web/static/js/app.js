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
    Object.keys(state.modelPresets).forEach(key => {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = key;
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
}

function loadModelPreset() {
    const presetKey = elements.modelPreset.value;
    if (!presetKey || !state.modelPresets[presetKey]) return;
    
    const preset = state.modelPresets[presetKey];
    
    // Update form fields
    elements.modelType.value = preset.type;
    
    // Handle video generation pipeline presets (e.g., wan-t2v-14b)
    if (preset.type === 'wan-pipeline') {
        // For video generation, set default values for model params
        document.getElementById('hidden-size').value = 4096;
        document.getElementById('num-layers').value = 32;
        document.getElementById('num-heads').value = 32;
        document.getElementById('max-seq-len').value = 512;
        document.getElementById('dtype').value = 'bf16';
        
        // Store pipeline info for evaluate function
        state.currentPipeline = 'diffusion-video';
        state.videoParams = {
            num_frames: 81,
            height: 720,
            width: 1280,
            num_inference_steps: 50,
            use_cfg: true
        };
    } else {
        // Standard LLM/MoE models
        document.getElementById('hidden-size').value = preset.hidden_size;
        document.getElementById('num-layers').value = preset.num_layers;
        document.getElementById('num-heads').value = preset.num_attention_heads;
        document.getElementById('max-seq-len').value = preset.max_seq_len;
        document.getElementById('dtype').value = preset.dtype;
        
        // Clear pipeline info
        state.currentPipeline = null;
        
        // MoE params
        if (preset.num_experts) {
            document.getElementById('num-experts').value = preset.num_experts;
            document.getElementById('experts-per-token').value = preset.num_experts_per_token;
        }
    }
    
    updateModelTypeUI();
}

function updateModelTypeUI() {
    const isMoE = elements.modelType.value === 'moe';
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
            endpoint = '/api/evaluate/pipeline/diffusion-video';
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
    
    const config = {
        model: {
            type: elements.modelType.value,
            vocab_size: 32000,
            hidden_size: parseInt(document.getElementById('hidden-size').value),
            num_layers: parseInt(document.getElementById('num-layers').value),
            num_attention_heads: parseInt(document.getElementById('num-heads').value),
            max_seq_len: parseInt(document.getElementById('max-seq-len').value),
            dtype: document.getElementById('dtype').value,
            ...(elements.modelType.value === 'moe' && {
                num_experts: parseInt(document.getElementById('num-experts').value),
                num_experts_per_token: parseInt(document.getElementById('experts-per-token').value)
            })
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
    
    // Add video generation params for diffusion-video pipeline
    if (state.currentPipeline === 'diffusion-video' && state.videoParams) {
        config.num_frames = state.videoParams.num_frames;
        config.height = state.videoParams.height;
        config.width = state.videoParams.width;
        config.num_inference_steps = state.videoParams.num_inference_steps;
        config.use_cfg = state.videoParams.use_cfg;
    }
    
    // Add mode-specific params
    if (state.mode === 'training') {
        config.training = {
            batch_size: parseInt(document.getElementById('batch-size').value),
            seq_len: parseInt(document.getElementById('seq-len').value),
            num_steps: 1000
        };
    } else {
        config.inference = {
            batch_size: parseInt(document.getElementById('inf-batch-size').value),
            prompt_len: parseInt(document.getElementById('prompt-len').value),
            generation_len: parseInt(document.getElementById('generation-len').value)
        };
    }
    
    return config;
}

function displayResults(result) {
    elements.results.style.display = 'block';
    
    // Handle pipeline results (e.g., video generation)
    if (state.currentPipeline === 'diffusion-video') {
        const metadata = result.metadata || {};
        const breakdown = metadata.component_breakdown || {};
        elements.resultsContent.innerHTML = `
            <div class="result-grid">
                <div class="result-card highlight">
                    <div class="result-value">${((result.total_time_sec || 0) / 60).toFixed(1)}m</div>
                    <div class="result-label">Total Time</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${(result.memory_peak_gb || 0).toFixed(1)}GB</div>
                    <div class="result-label">Peak Memory</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${((result.throughput || 0) / 1000000).toFixed(2)}M</div>
                    <div class="result-label">Pixels/sec</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${metadata.num_inference_steps || 50}</div>
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
                    <td>Text Encoder</td>
                    <td>${((breakdown.text_encoder_time || 0) * 1000).toFixed(2)} ms</td>
                    <td>${(breakdown.text_encoder_pct || 0).toFixed(3)}%</td>
                </tr>
                <tr>
                    <td>DiT Denoising (${metadata.num_inference_steps || 50} steps)</td>
                    <td>${(breakdown.dit_total_time || 0).toFixed(2)} s</td>
                    <td>${(breakdown.dit_pct || 0).toFixed(1)}%</td>
                </tr>
                <tr>
                    <td>VAE Decoder</td>
                    <td>${(breakdown.vae_decoder_time || 0).toFixed(2)} s</td>
                    <td>${(breakdown.vae_decoder_pct || 0).toFixed(2)}%</td>
                </tr>
            </table>
            <div style="margin-top: 1rem; padding: 1rem; background: var(--gray-100); border-radius: 8px;">
                <strong>生成配置:</strong> ${metadata.num_frames || 81}帧 @ ${metadata.height || 720}x${metadata.width || 1280} | CFG: ${metadata.use_cfg ? '启用' : '禁用'}
            </div>
        `;
        elements.results.scrollIntoView({ behavior: 'smooth' });
        return;
    }
    
    if (state.mode === 'training') {
        elements.resultsContent.innerHTML = `
            <div class="result-grid">
                <div class="result-card highlight">
                    <div class="result-value">${(result.tokens_per_sec / 1000).toFixed(2)}K</div>
                    <div class="result-label">Tokens/sec</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${result.samples_per_sec.toFixed(2)}</div>
                    <div class="result-label">Samples/sec</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${(result.time_per_step_sec * 1000).toFixed(0)}ms</div>
                    <div class="result-label">Time/Step</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${result.memory_per_gpu_gb.toFixed(1)}GB</div>
                    <div class="result-label">Memory/GPU</div>
                </div>
            </div>
            ${renderBreakdown(result.breakdown)}
        `;
    } else {
        elements.resultsContent.innerHTML = `
            <div class="result-grid">
                <div class="result-card highlight">
                    <div class="result-value">${result.decode_tokens_per_sec.toFixed(0)}</div>
                    <div class="result-label">Decode TPS</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${(result.prefill_time_sec * 1000).toFixed(0)}ms</div>
                    <div class="result-label">TTFT (Time To First Token)</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${(result.decode_time_per_step_sec * 1000).toFixed(1)}ms</div>
                    <div class="result-label">TPOT (Time Per Output Token)</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${result.overall_tps.toFixed(0)}</div>
                    <div class="result-label">Overall TPS</div>
                </div>
            </div>
            <div class="result-grid" style="margin-top: 1rem;">
                <div class="result-card">
                    <div class="result-value">${result.total_time_sec.toFixed(2)}s</div>
                    <div class="result-label">Total Time</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${result.memory_per_gpu_gb.toFixed(1)}GB</div>
                    <div class="result-label">Memory/GPU</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${result.kv_cache_memory_gb.toFixed(1)}GB</div>
                    <div class="result-label">KV Cache</div>
                </div>
            </div>
        `;
    }
    
    elements.results.scrollIntoView({ behavior: 'smooth' });
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
