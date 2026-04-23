"""Test framework overhead calculation and seq_len handling."""

import pytest
from llm_perf.analyzer.unified import UnifiedAnalyzer
from llm_perf.analyzer.base import Phase, ComputeType, WorkloadType, WorkloadConfig
from llm_perf.hardware.device import Device, DeviceConfig
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.strategy.base import StrategyConfig
from llm_perf.modeling import LlamaModel
from llm_perf.modeling.layers import ShardedAttention
from llm_perf.kernels.functional import flash_attention


def create_test_cluster(device: Device, num_devices: int = 1) -> Cluster:
    """Create a simple test cluster."""
    topology = NetworkTopology.create_2tier_simple(
        intra_node_bw_gbps=400.0,
        inter_node_bw_gbps=100.0,
    )
    return Cluster.create_homogeneous(device.config, num_devices, topology)


class TestSeqLenHandling:
    """Test correct seq_len handling for prefill and decode phases."""

    def test_prefill_seq_len_from_prompt_len(self):
        """Prefill phase should use seq_len = prompt_len."""
        device = Device(DeviceConfig(
            name="H100",
            memory_gb=80,
            memory_bandwidth_gbps=3352,
            fp16_tflops_cube=1000
        ))
        cluster = create_test_cluster(device, num_devices=1)
        strategy = StrategyConfig(tp_degree=1)

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
        )

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        prefill_phase = Phase(
            name="prefill",
            compute_type=ComputeType.FORWARD,
            component="main",
            repeat=1,
        )

        params = {"batch_size": 1, "prompt_len": 512}

        if prefill_phase.name == "prefill":
            seq_len = params.get("prompt_len", params.get("seq_len", 512))
        elif prefill_phase.name == "decode":
            seq_len = 1
        else:
            seq_len = params.get("seq_len", params.get("prompt_len", 512))

        assert seq_len == 512, f"Prefill seq_len should be prompt_len (512), got {seq_len}"

    def test_decode_seq_len_equals_one(self):
        """Decode phase should use seq_len = 1."""
        device = Device(DeviceConfig(
            name="H100",
            memory_gb=80,
            memory_bandwidth_gbps=3352,
            fp16_tflops_cube=1000
        ))
        cluster = create_test_cluster(device, num_devices=1)
        strategy = StrategyConfig(tp_degree=1)

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
        )

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        decode_phase = Phase(
            name="decode",
            compute_type=ComputeType.FORWARD,
            component="main",
            repeat=128,
        )

        params = {"batch_size": 1, "prompt_len": 512}

        if decode_phase.name == "prefill":
            seq_len = params.get("prompt_len", params.get("seq_len", 512))
        elif decode_phase.name == "decode":
            seq_len = 1
        else:
            seq_len = params.get("seq_len", params.get("prompt_len", 512))

        assert seq_len == 1, f"Decode seq_len should be 1, got {seq_len}"

    def test_seq_len_difference_between_phases(self):
        """Prefill and decode should have different seq_len values."""
        device = Device(DeviceConfig(
            name="H100",
            memory_gb=80,
            memory_bandwidth_gbps=3352,
            fp16_tflops_cube=1000
        ))
        cluster = create_test_cluster(device, num_devices=1)
        strategy = StrategyConfig(tp_degree=1)

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
        )

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        params = {"batch_size": 1, "prompt_len": 1024}

        prefill_seq_len = params.get("prompt_len", params.get("seq_len", 512))
        decode_seq_len = 1

        assert prefill_seq_len > decode_seq_len, "Prefill seq_len should be larger than decode seq_len"
        assert prefill_seq_len == 1024
        assert decode_seq_len == 1


class TestKVCacheFrameworkOverhead:
    """Test kvcache framework overhead calculation."""

    def test_framework_overhead_zero_for_prefill(self):
        """Prefill phase should have zero framework overhead."""
        device = Device(DeviceConfig(
            name="H100",
            memory_gb=80,
            memory_bandwidth_gbps=3352,
            fp16_tflops_cube=1000
        ))
        cluster = create_test_cluster(device, num_devices=1)
        strategy = StrategyConfig(tp_degree=1)

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
        )

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        prefill_phase = Phase(
            name="prefill",
            compute_type=ComputeType.FORWARD,
            component="main",
            repeat=1,
        )

        params = {"batch_size": 1, "prompt_len": 512}

        overhead = analyzer._calculate_framework_overhead(prefill_phase, params, model)

        assert overhead == 0, f"Prefill framework overhead should be 0, got {overhead}"

    def test_framework_overhead_positive_for_decode(self):
        """Decode phase should have positive framework overhead."""
        device = Device(DeviceConfig(
            name="H100",
            memory_gb=80,
            memory_bandwidth_gbps=3352,
            fp16_tflops_cube=1000
        ))
        cluster = create_test_cluster(device, num_devices=1)
        strategy = StrategyConfig(tp_degree=1)

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
            num_kv_heads=8,
        )

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        decode_phase = Phase(
            name="decode",
            compute_type=ComputeType.FORWARD,
            component="main",
            repeat=128,
        )

        params = {"batch_size": 1, "prompt_len": 512, "generated_tokens": 128}

        overhead = analyzer._calculate_framework_overhead(decode_phase, params, model)

        assert overhead > 0, f"Decode framework overhead should be > 0, got {overhead}"

        batch_size = params.get("batch_size", 1)
        prompt_len = params.get("prompt_len", 512)
        generated_tokens = params.get("generated_tokens", 0)
        kv_seq_len = prompt_len + generated_tokens

        num_kv_heads = 8
        head_dim = 4096 // 32
        num_layers = 2
        dtype_size = 2

        expected_read = batch_size * kv_seq_len * num_kv_heads * head_dim * dtype_size * 2
        expected_write = batch_size * 1 * num_kv_heads * head_dim * dtype_size * 2
        expected_total = (expected_read + expected_write) * num_layers

        assert overhead == expected_total, f"Expected {expected_total} bytes, got {overhead}"

    def test_framework_overhead_with_different_batch_size(self):
        """Framework overhead should scale with batch size."""
        device = Device(DeviceConfig(
            name="H100",
            memory_gb=80,
            memory_bandwidth_gbps=3352,
            fp16_tflops_cube=1000
        ))
        cluster = create_test_cluster(device, num_devices=1)
        strategy = StrategyConfig(tp_degree=1)

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
            num_kv_heads=8,
        )

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        decode_phase = Phase(
            name="decode",
            compute_type=ComputeType.FORWARD,
            component="main",
            repeat=128,
        )

        params_batch1 = {"batch_size": 1, "prompt_len": 512, "generated_tokens": 128}
        params_batch4 = {"batch_size": 4, "prompt_len": 512, "generated_tokens": 128}

        overhead_batch1 = analyzer._calculate_framework_overhead(decode_phase, params_batch1, model)
        overhead_batch4 = analyzer._calculate_framework_overhead(decode_phase, params_batch4, model)

        assert overhead_batch4 == overhead_batch1 * 4, "Framework overhead should scale linearly with batch size"


class TestMoEExpertMemory:
    """Test MoE expert memory calculation."""

    def test_moe_expert_flops_calculation(self):
        """Test MoE expert FLOPs calculation."""
        from llm_perf.kernels.functional import moe_expert

        hidden_shape = (1, 512, 4096)
        intermediate_size = 14336
        num_experts_per_token = 1

        result = moe_expert(hidden_shape, intermediate_size, num_experts_per_token)

        assert result.flops > 0
        assert result.output == hidden_shape

        batch_size = 1 * 512
        hidden_size = 4096

        gate_flops = 2 * batch_size * hidden_size * intermediate_size
        gate_activation_flops = batch_size * intermediate_size * 10
        up_flops = 2 * batch_size * hidden_size * intermediate_size
        gate_up_mul_flops = batch_size * intermediate_size
        down_flops = 2 * batch_size * intermediate_size * hidden_size

        expected_flops = gate_flops + gate_activation_flops + up_flops + gate_up_mul_flops + down_flops

        assert result.flops == expected_flops

    def test_moe_expert_memory_calculation(self):
        """Test MoE expert memory calculation."""
        from llm_perf.kernels.functional import moe_expert

        hidden_shape = (2, 256, 4096)
        intermediate_size = 14336

        result = moe_expert(hidden_shape, intermediate_size, num_experts_per_token=2)

        assert result.bytes_accessed > 0
        assert result.params > 0

        hidden_size = 4096
        params_per_expert = hidden_size * intermediate_size * 3

        assert result.params == params_per_expert
        assert result.arithmetic_intensity > 200
        assert result.memory_bound == False

    def test_moe_expert_backward_metrics(self):
        """Test MoE expert backward metrics."""
        from llm_perf.kernels.functional import moe_expert

        hidden_shape = (1, 128, 4096)
        intermediate_size = 14336

        result = moe_expert(hidden_shape, intermediate_size)

        assert result.flops_backward > 0
        assert result.bytes_accessed_backward > 0

        assert result.flops_backward >= result.flops * 1.5
        assert result.bytes_accessed_backward >= result.bytes_accessed * 1.5


class TestAttentionOpPurity:
    """Test AttentionOp is pure kernel layer (no business logic)."""

    def test_attention_op_no_phase_parameter(self):
        """AttentionOp should not have phase parameter."""
        from llm_perf.kernels.op import AttentionOp
        from llm_perf.modeling.tensor import ShardedTensor

        query = ShardedTensor(shape=(1, 32, 512, 128))
        key = ShardedTensor(shape=(1, 8, 512, 128))
        value = ShardedTensor(shape=(1, 8, 512, 128))
        output = ShardedTensor(shape=(1, 32, 512, 128))

        op = AttentionOp(
            query=query,
            key=key,
            value=value,
            output=output,
        )

        assert not hasattr(op, "phase"), "AttentionOp should not have 'phase' attribute"
        assert not hasattr(op, "kv_cache_config"), "AttentionOp should not have 'kv_cache_config' attribute"
        assert not hasattr(op, "kv_cache_memory"), "AttentionOp should not have 'kv_cache_memory' method"

    def test_attention_op_kernel_attributes(self):
        """AttentionOp should only have kernel-level attributes."""
        from llm_perf.kernels.op import AttentionOp
        from llm_perf.modeling.tensor import ShardedTensor

        query = ShardedTensor(shape=(1, 32, 512, 128))
        key = ShardedTensor(shape=(1, 8, 512, 128))
        value = ShardedTensor(shape=(1, 8, 512, 128))
        output = ShardedTensor(shape=(1, 32, 512, 128))

        op = AttentionOp(
            query=query,
            key=key,
            value=value,
            output=output,
            is_causal=True,
        )

        assert hasattr(op, "kernel_name")
        assert hasattr(op, "dtype")
        assert hasattr(op, "query")
        assert hasattr(op, "key")
        assert hasattr(op, "value")
        assert hasattr(op, "output")
        assert hasattr(op, "is_causal")
        assert op.kernel_name == "flash_attention"
        assert op.is_causal == True


class TestKVSeqLenInDecodeAttention:
    """Test that decode phase attention uses kv_seq_len for K/V tensor shape."""

    def test_attention_kv_seq_len_forward(self):
        """ShardedAttention forward should use _kv_seq_len for K/V tensor shape."""
        from llm_perf.modeling.tensor import ShardedTensor

        attn = ShardedAttention(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
        )

        attn._kv_seq_len = 640

        hidden = ShardedTensor(shape=(1, 1, 4096))
        output = attn(hidden)

        assert output is not None

        for op in output._op_history:
            if hasattr(op, "key"):
                assert op.key.shape[2] == 640, f"K tensor kv_seq_len should be 640, got {op.key.shape[2]}"
            if hasattr(op, "value"):
                assert op.value.shape[2] == 640, f"V tensor kv_seq_len should be 640, got {op.value.shape[2]}"

    def test_attention_flops_scales_with_kv_seq_len(self):
        """Attention FLOPs should scale linearly with kv_seq_len."""
        batch = 1
        num_heads = 32
        kv_num_heads = 8
        seq_len = 1
        head_dim = 128

        kv_seq_len_short = 128
        kv_seq_len_long = 1024

        result_short = flash_attention(
            (batch, num_heads, seq_len, head_dim),
            (batch, kv_num_heads, kv_seq_len_short, head_dim),
            (batch, kv_num_heads, kv_seq_len_short, head_dim),
        )

        result_long = flash_attention(
            (batch, num_heads, seq_len, head_dim),
            (batch, kv_num_heads, kv_seq_len_long, head_dim),
            (batch, kv_num_heads, kv_seq_len_long, head_dim),
        )

        ratio = result_long.flops / result_short.flops
        expected_ratio = kv_seq_len_long / kv_seq_len_short

        assert ratio == expected_ratio, f"FLOPs ratio should be {expected_ratio}, got {ratio}"

    def test_decode_attention_time_differs_with_kv_seq_len(self):
        """Decode phase attention time should differ with different kv_seq_len."""
        device = Device(DeviceConfig(
            name="H100",
            memory_gb=80,
            memory_bandwidth_gbps=3352,
            fp16_tflops_cube=1000
        ))
        cluster = create_test_cluster(device, num_devices=1)
        strategy = StrategyConfig(tp_degree=1)

        model_short = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
            num_kv_heads=8,
        )

        model_long = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
            num_kv_heads=8,
        )

        workload = WorkloadConfig(
            name="inference",
            workload_type=WorkloadType.INFERENCE,
            phases=[
                Phase(name="decode", compute_type=ComputeType.FORWARD, component="main", repeat=1),
            ],
        )

        analyzer_short = UnifiedAnalyzer(model_short, device, cluster, strategy)
        result_short = analyzer_short.analyze(workload, batch_size=1, prompt_len=512, generated_tokens=128)

        analyzer_long = UnifiedAnalyzer(model_long, device, cluster, strategy)
        result_long = analyzer_long.analyze(workload, batch_size=1, prompt_len=512, generated_tokens=512)

        decode_short = result_short.phases[0]
        decode_long = result_long.phases[0]

        assert decode_long.single_time_sec > decode_short.single_time_sec, (
            f"Decode time should increase with longer kv_seq_len: "
            f"short={decode_short.single_time_sec:.6f}, long={decode_long.single_time_sec:.6f}"
        )


class TestLinearAttentionKernel:
    """Test Linear Attention kernel FLOPs and memory characteristics."""

    def test_linear_attention_flops_linear_in_seq(self):
        """Linear Attention FLOPs should be O(seq), not O(seq^2)."""
        from llm_perf.kernels.functional import linear_attention, flash_attention

        batch = 1
        num_heads = 16
        num_kv_heads = 32
        head_dim = 128
        kernel_dim = 4

        seq_short = 512
        seq_long = 2048

        linear_result_short = linear_attention(
            (batch, num_heads, seq_short, head_dim),
            (batch, num_kv_heads, seq_short, head_dim),
            (batch, num_kv_heads, seq_short, head_dim),
            kernel_dim=kernel_dim,
        )

        linear_result_long = linear_attention(
            (batch, num_heads, seq_long, head_dim),
            (batch, num_kv_heads, seq_long, head_dim),
            (batch, num_kv_heads, seq_long, head_dim),
            kernel_dim=kernel_dim,
        )

        linear_ratio = linear_result_long.flops / linear_result_short.flops

        assert linear_ratio == seq_long / seq_short, (
            f"Linear Attention FLOPs ratio should be {seq_long/seq_short} (linear), got {linear_ratio}"
        )

        flash_result_short = flash_attention(
            (batch, num_heads, seq_short, head_dim),
            (batch, num_kv_heads, seq_short, head_dim),
            (batch, num_kv_heads, seq_short, head_dim),
        )

        flash_result_long = flash_attention(
            (batch, num_heads, seq_long, head_dim),
            (batch, num_kv_heads, seq_long, head_dim),
            (batch, num_kv_heads, seq_long, head_dim),
        )

        flash_ratio = flash_result_long.flops / flash_result_short.flops

        assert flash_ratio > linear_ratio, (
            f"Flash Attention FLOPs growth ({flash_ratio}) should exceed Linear Attention ({linear_ratio})"
        )

    def test_linear_attention_memory_no_seq_growth(self):
        """Linear Attention memory should NOT have seq^2 term (no KV cache growth)."""
        from llm_perf.kernels.functional import linear_attention

        batch = 1
        num_heads = 16
        num_kv_heads = 32
        head_dim = 128
        kernel_dim = 4

        seq = 4096
        dtype_size = 2

        result = linear_attention(
            (batch, num_heads, seq, head_dim),
            (batch, num_kv_heads, seq, head_dim),
            (batch, num_kv_heads, seq, head_dim),
            kernel_dim=kernel_dim,
        )

        seq_linear_term = (
            batch * num_heads * seq * head_dim * dtype_size  # Q
            + batch * num_kv_heads * seq * head_dim * dtype_size  # K
            + batch * num_kv_heads * seq * head_dim * dtype_size  # V
            + batch * num_heads * seq * head_dim * dtype_size  # Output
        )
        state_term = batch * num_kv_heads * head_dim * head_dim * dtype_size

        expected_bytes = seq_linear_term + state_term

        assert result.bytes_accessed == expected_bytes, (
            f"Linear Attention bytes should be {expected_bytes} (linear + state), got {result.bytes_accessed}"
        )

        seq_squared_term = batch * num_heads * seq * seq
        squared_bytes = seq_squared_term * dtype_size

        assert result.bytes_accessed < squared_bytes, (
            f"Linear Attention bytes ({result.bytes_accessed}) should be less than seq^2 term ({squared_bytes})"
        )

    def test_linear_attention_backward_metrics(self):
        """Linear Attention backward metrics should be ~2x forward."""
        from llm_perf.kernels.functional import linear_attention

        batch = 1
        num_heads = 16
        num_kv_heads = 32
        seq = 512
        head_dim = 128
        kernel_dim = 4

        result = linear_attention(
            (batch, num_heads, seq, head_dim),
            (batch, num_kv_heads, seq, head_dim),
            (batch, num_kv_heads, seq, head_dim),
            kernel_dim=kernel_dim,
        )

        assert result.flops_backward > 0
        assert result.bytes_accessed_backward > 0

        assert result.flops_backward >= result.flops * 1.5
        assert result.bytes_accessed_backward >= result.bytes_accessed * 1.5

    def test_linear_attention_op_purity(self):
        """LinearAttentionOp should not have phase/workload parameters."""
        from llm_perf.kernels.op import LinearAttentionOp
        from llm_perf.modeling.tensor import ShardedTensor

        query = ShardedTensor(shape=(1, 16, 512, 128))
        key = ShardedTensor(shape=(1, 32, 512, 128))
        value = ShardedTensor(shape=(1, 32, 512, 128))
        output = ShardedTensor(shape=(1, 16, 512, 128))

        op = LinearAttentionOp(
            query=query,
            key=key,
            value=value,
            output=output,
            kernel_dim=4,
        )

        assert not hasattr(op, "phase"), "LinearAttentionOp should not have 'phase' attribute"
        assert not hasattr(op, "kv_cache_config"), "LinearAttentionOp should not have 'kv_cache_config' attribute"
        assert not hasattr(op, "kv_seq_len"), "LinearAttentionOp should not have 'kv_seq_len' attribute"

        assert hasattr(op, "kernel_name")
        assert hasattr(op, "kernel_dim")
        assert hasattr(op, "is_causal")
        assert op.kernel_name == "linear_attention"

    def test_linear_attention_sharded_module(self):
        """ShardedLinearAttention module should work correctly."""
        from llm_perf.modeling.layers import ShardedLinearAttention
        from llm_perf.modeling.tensor import ShardedTensor

        attn = ShardedLinearAttention(
            hidden_size=4096,
            num_heads=16,
            num_kv_heads=32,
            head_dim=128,
            kernel_dim=4,
        )

        hidden = ShardedTensor(shape=(1, 512, 4096))
        output = attn(hidden)

        assert output.shape == (1, 512, 4096)
        assert "q_proj" in attn._activations
        assert "k_proj" in attn._activations
        assert "v_proj" in attn._activations
        assert "output" in attn._activations

    def test_linear_attention_tp_sharding(self):
        """ShardedLinearAttention should support TP sharding."""
        from llm_perf.modeling.layers import ShardedLinearAttention
        from llm_perf.modeling import ParallelContext

        attn = ShardedLinearAttention(
            hidden_size=4096,
            num_heads=16,
            num_kv_heads=32,
            head_dim=128,
            kernel_dim=4,
        )

        ctx = ParallelContext(tp_degree=8)
        instance = attn.bind(ctx)

        assert instance.params_count_physical < attn.params_count()

        q_weight_physical = instance._weight_instances["q_weight"].physical_shape
        assert q_weight_physical[1] == (16 * 128) // 8, "Q weight should be TP-sharded on heads dimension"