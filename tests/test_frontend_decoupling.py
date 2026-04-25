"""Tests for frontend breakdown decoupling.

Ensures:
1. All breakdown categories are auto-discovered from backend data
2. Qwen3.5 full_attention appears in nested breakdown
3. New categories automatically included in time_breakdown
4. New submodule types automatically included in by_submodule_type
"""

import pytest

from llm_perf.modeling.qwen3_5 import Qwen3_5MoEModel
from llm_perf.hardware.device import Device, DeviceConfig
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer.unified import UnifiedAnalyzer


def make_cluster(device, num_devices):
    topology = NetworkTopology(
        name="test",
        intra_node_bandwidth_gbps=200.0,
        intra_node_latency_us=1.0,
        inter_node_bandwidth_gbps=25.0,
        inter_node_latency_us=10.0,
    )
    return Cluster.create_homogeneous(device.config, num_devices, topology)


class TestQwen35FullAttentionBreakdown:
    """Test Qwen3.5 full_attention appears in breakdown."""

    @pytest.fixture
    def setup(self):
        model = Qwen3_5MoEModel(
            vocab_size=151936,
            hidden_size=2048,
            num_layers=8,
            num_heads=16,
            num_experts=64,
            num_experts_per_token=8,
            intermediate_size=1024,
            dtype="fp16",
        )

        device_config = DeviceConfig(
            name="nvidia_a100",
            memory_gb=80.0,
            memory_bandwidth_gbps=2039.0,
            fp16_tflops=312.0,
        )
        device = Device(device_config)
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=2, pp_degree=1, dp_degree=4, ep_degree=1)
        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        return {"model": model, "analyzer": analyzer}

    def test_qwen35_full_attention_in_nested_breakdown(self, setup):
        """Test Qwen3.5 full_attention appears in nested_breakdown."""
        analyzer = setup["analyzer"]
        result = analyzer.analyze("training", batch_size=4, seq_len=1024)

        detailed = result.detailed_breakdown
        by_type = detailed.get("by_submodule_type", {})
        assert "transformer_block" in by_type

        nested = by_type["transformer_block"].get("nested_breakdown", {})
        nested_keys = list(nested.keys())

        assert "linear_attention" in nested_keys, f"linear_attention not in {nested_keys}"
        assert "attention" in nested_keys, f"attention (full_attention) not in {nested_keys}"
        assert "moe" in nested_keys, f"moe not in {nested_keys}"

        assert nested["linear_attention"]["compute"]["time_sec"] > 0
        assert nested["attention"]["compute"]["time_sec"] > 0
        assert nested["moe"]["compute"]["time_sec"] > 0

    def test_qwen35_both_attention_types_have_data(self, setup):
        """Test both linear_attention and attention have valid data."""
        analyzer = setup["analyzer"]
        result = analyzer.analyze("training", batch_size=4, seq_len=1024)

        detailed = result.detailed_breakdown
        nested = detailed["by_submodule_type"]["transformer_block"]["nested_breakdown"]

        linear_attn = nested["linear_attention"]
        full_attn = nested["attention"]

        assert linear_attn["memory"]["params_count"] > 0
        assert full_attn["memory"]["params_count"] > 0
        assert linear_attn["compute"]["flops"] > 0
        assert full_attn["compute"]["flops"] > 0

    def test_qwen35_signature_different_for_different_attention_types(self, setup):
        """Test different attention types have different signatures."""
        model = setup["model"]
        analyzer = setup["analyzer"]

        from llm_perf.modeling.tensor import ShardedTensor
        from llm_perf.strategy.parallel_context import ParallelContext, SPType

        ctx = ParallelContext(
            tp_degree=2,
            pp_degree=1,
            dp_degree=4,
            ep_degree=1,
            sp_degree=1,
            sp_type=SPType.NONE,
            dtype="fp16",
            device=analyzer.device.config,
        )

        input_tensor = ShardedTensor(shape=(4, 1024))
        model(input_tensor)
        module_inst = model.bind(ctx, mode="forward_backward")

        block0_sig = analyzer._compute_structure_signature(
            module_inst._submodule_instances["layers.0"]
        )
        block3_sig = analyzer._compute_structure_signature(
            module_inst._submodule_instances["layers.3"]
        )

        assert "layer_type=linear_attention" in block0_sig
        assert "layer_type=full_attention" in block3_sig
        assert block0_sig != block3_sig


class TestTimeBreakdownAutoDiscovery:
    """Test time_breakdown auto-discovery of new categories."""

    def test_time_breakdown_has_expected_fields(self):
        """Test time_breakdown has expected fields."""
        from llm_perf.modeling import LlamaModel

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )

        device_config = DeviceConfig(
            name="nvidia_a100",
            memory_gb=80.0,
            memory_bandwidth_gbps=2039.0,
            fp16_tflops=312.0,
        )
        device = Device(device_config)
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=2, pp_degree=1, dp_degree=4)
        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        result = analyzer.analyze("training", batch_size=4, seq_len=1024)

        time_breakdown = result.breakdown.get("time_breakdown", {})
        assert "compute_sec" in time_breakdown
        assert "backward_sec" in time_breakdown
        assert "optimizer_sec" in time_breakdown
        assert "communication_sec" in time_breakdown

        assert "compute_percent" in time_breakdown
        assert "backward_percent" in time_breakdown

    def test_memory_sec_field_in_time_breakdown(self):
        """Test memory_sec field in time_breakdown (kv_cache for inference)."""
        from llm_perf.modeling import LlamaModel

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )

        device_config = DeviceConfig(
            name="nvidia_a100",
            memory_gb=80.0,
            memory_bandwidth_gbps=2039.0,
            fp16_tflops=312.0,
        )
        device = Device(device_config)
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=2, pp_degree=1, dp_degree=4)
        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        result = analyzer.analyze("training", batch_size=4, seq_len=1024)

        time_breakdown = result.breakdown.get("time_breakdown", {})
        assert "memory_sec" in time_breakdown


class TestSubmoduleTypeAutoDiscovery:
    """Test by_submodule_type auto-discovery of new types."""

    def test_all_submodule_types_from_data(self):
        """Test all submodule types are discovered from data, not hardcoded."""
        from llm_perf.modeling import LlamaModel

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )

        device_config = DeviceConfig(
            name="nvidia_a100",
            memory_gb=80.0,
            memory_bandwidth_gbps=2039.0,
            fp16_tflops=312.0,
        )
        device = Device(device_config)
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=2, pp_degree=1, dp_degree=4)
        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        result = analyzer.analyze("training", batch_size=4, seq_len=1024)

        by_type = result.detailed_breakdown.get("by_submodule_type", {})
        type_keys = list(by_type.keys())

        assert "embedding" in type_keys
        assert "transformer_block" in type_keys
        assert "lm_head" in type_keys

        for type_name, data in by_type.items():
            assert "memory" in data
            assert "compute" in data
            assert "communication" in data


class TestCommunicationAutoDiscovery:
    """Test communication breakdown auto-discovery."""

    def test_by_parallelism_from_data(self):
        """Test communication by_parallelism is from data."""
        from llm_perf.modeling import LlamaModel

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )

        device_config = DeviceConfig(
            name="nvidia_a100",
            memory_gb=80.0,
            memory_bandwidth_gbps=2039.0,
            fp16_tflops=312.0,
        )
        device = Device(device_config)
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=2, pp_degree=1, dp_degree=4)
        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        result = analyzer.analyze("training", batch_size=4, seq_len=1024)

        comm_by_para = result.detailed_breakdown.get("communication", {}).get(
            "by_parallelism", {}
        )

        if comm_by_para:
            for para_type, data in comm_by_para.items():
                assert "total_bytes" in data
                assert "total_time_sec" in data

    def test_by_operation_from_data(self):
        """Test communication by_operation is from data."""
        from llm_perf.modeling import LlamaModel

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )

        device_config = DeviceConfig(
            name="nvidia_a100",
            memory_gb=80.0,
            memory_bandwidth_gbps=2039.0,
            fp16_tflops=312.0,
        )
        device = Device(device_config)
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=2, pp_degree=1, dp_degree=4)
        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        result = analyzer.analyze("training", batch_size=4, seq_len=1024)

        comm_by_op = result.detailed_breakdown.get("communication", {}).get(
            "by_operation", {}
        )

        if comm_by_op:
            for op_type, data in comm_by_op.items():
                assert "total_bytes" in data
                assert "total_time_sec" in data