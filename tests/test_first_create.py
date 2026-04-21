import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_perf.modeling import DeepSeekModel, create_model_from_config
from llm_perf.hardware import Device, Cluster, NetworkTopology
from llm_perf.strategy import StrategyConfig, ParallelContext
from llm_perf.analyzer import UnifiedAnalyzer, get_workload
from llm_perf.modeling.tensor import ShardedTensor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

def test_deepseek_v3_first_create():
    """测试第一次创建 DeepSeekModel 时 weight 是否正确."""
    
    model_config = {"preset": "deepseek-v3"}
    
    print("=" * 60)
    print("[TEST] First model creation")
    print("=" * 60)
    
    model = create_model_from_config(model_config)
    
    print(f"Model type: {type(model).__name__}")
    print(f"Model._submodules keys: {list(model._submodules.keys())[:10]}...")
    print(f"Model._submodules count: {len(model._submodules)}")
    print(f"Expected layers: 61 (1 dense + 60 MoE)")
    print(f"Model.layers count: {len(model.layers)}")
    
    weights = model.get_weights()
    print(f"Model.get_weights() count: {len(weights)}")
    
    device = Device.from_preset("H100-SXM-80GB")
    topology = NetworkTopology.create_2tier_simple(900, 200)
    cluster = Cluster.create_homogeneous(device.config, 8, topology)
    strategy = StrategyConfig(tp_degree=8, pp_degree=1, dp_degree=1)
    
    ctx = ParallelContext(
        tp_degree=8,
        pp_degree=1,
        dp_degree=1,
        device=device.config,
    )
    
    input_tensor = ShardedTensor(shape=(32, 4096), dtype="fp16", name="input")
    model(input_tensor)
    
    instance = model.bind(ctx, mode="forward_backward")
    
    print(f"ModuleInstance._submodule_instances count: {len(instance._submodule_instances)}")
    print(f"ModuleInstance._weight_instances count: {len(instance._weight_instances)}")
    print(f"ModuleInstance.params_count_physical: {instance.params_count_physical / 1e9:.2f}B")
    print(f"ModuleInstance.weight_memory_physical: {instance.weight_memory_physical / 1e9:.2f}GB")
    
    expected_weight_gb = 168.0
    actual_weight_gb = instance.weight_memory_physical / 1e9
    
    print(f"\nExpected: ~{expected_weight_gb}GB (DeepSeek-V3, TP=8)")
    print(f"Actual: {actual_weight_gb:.2f}GB")
    
    if abs(actual_weight_gb - expected_weight_gb) < 10:
        print("[PASS] Weight memory is correct!")
    else:
        print(f"[FAIL] Weight memory mismatch! Difference: {abs(actual_weight_gb - expected_weight_gb):.2f}GB")
    
    return actual_weight_gb

def test_deepseek_v3_second_create():
    """测试第二次创建."""
    print("=" * 60)
    print("[TEST] Second model creation")
    print("=" * 60)
    return test_deepseek_v3_first_create()

def test_deepseek_v3_layers_registration():
    """测试 layers 注册的详细情况."""
    print("=" * 60)
    print("[TEST] Layers registration detail")
    print("=" * 60)
    
    model_config = {"preset": "deepseek-v3"}
    model = create_model_from_config(model_config)
    
    print(f"\n[ANALYSIS] DeepSeekModel._submodules:")
    for name in sorted(model._submodules.keys()):
        submodule = model._submodules[name]
        print(f"  {name}: {submodule.__class__.__name__}, weights={len(submodule._weights)}, submodules={len(submodule._submodules)}")
    
    print(f"\n[ANALYSIS] DeepSeekModel._weights:")
    for name in sorted(model._weights.keys()):
        weight = model._weights[name]
        print(f"  {name}: shape={weight.shape}, dtype={weight.dtype}")
    
    print(f"\n[ANALYSIS] Model.layers list:")
    for i, layer in enumerate(model.layers):
        print(f"  layers[{i}]: {layer.__class__.__name__}, _name={layer._name}")
    
    print(f"\n[ANALYSIS] Cross-check: Is layers list equal to layers.* in _submodules?")
    layers_from_submodules = [model._submodules.get(f"layers.{i}") for i in range(len(model.layers))]
    for i, (from_list, from_dict) in enumerate(zip(model.layers, layers_from_submodules)):
        if from_list is not from_dict:
            print(f"  [MISMATCH] layers[{i}] != _submodules['layers.{i}']")
        else:
            print(f"  [MATCH] layers[{i}] == _submodules['layers.{i}']")
    
    return model

if __name__ == "__main__":
    first = test_deepseek_v3_first_create()
    second = test_deepseek_v3_second_create()
    
    print("\n" + "=" * 60)
    print(f"First:  {first:.2f}GB")
    print(f"Second: {second:.2f}GB")
    print(f"Diff:   {abs(first - second):.2f}GB")
    print("=" * 60)
    
    test_deepseek_v3_layers_registration()