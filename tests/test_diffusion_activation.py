"""Tests for Diffusion activation memory behavior.

Diffusion inference differs from training in activation memory handling:
- Training: activation memory accumulates (forward + backward states)
- Inference: activation memory is peak only (no backward states)

Key tests:
1. Activation should be peak value, not sum across steps
2. Forward pass must succeed for diffusion model
3. Activation memory should be in reasonable range
"""

import pytest

from llm_perf.modeling import (
    ShardedTensor,
    create_model_from_config,
    get_model_presets,
)
from llm_perf.modeling.hunyuan_image import HunyuanImage3DiffusionModel
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer import UnifiedAnalyzer


def make_cluster(device, num_devices):
    topology = NetworkTopology(
        name="test",
        intra_node_bandwidth_gbps=200.0,
        intra_node_latency_us=1.0,
        inter_node_bandwidth_gbps=25.0,
        inter_node_latency_us=10.0,
    )
    return Cluster.create_homogeneous(device.config, num_devices, topology)


class TestDiffusionActivationMemory:
    """Diffusion scenario activation memory tests."""

    def test_hunyuan_image3_activation_peak_not_sum(self):
        """Inference activation memory should be peak, not sum across steps."""
        presets = get_model_presets()
        if "hunyuan-image-3" not in presets:
            pytest.skip("hunyuan-image-3 preset not available")

        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        models = {
            "encoder": create_model_from_config({"type": "hunyuan-t5-encoder"}),
            "backbone": create_model_from_config(
                {"preset": "hunyuan-image-3"}, workload_type="diffusion"
            ),
            "decoder": create_model_from_config({"type": "hunyuan-vae-decoder"}),
        }

        analyzer = UnifiedAnalyzer(models, device, cluster, strategy)
        result = analyzer.analyze(
            "diffusion-pipeline",
            height=1024,
            width=1024,
            num_steps=50,
        )

        result_dict = result.to_dict()

        if result_dict["detailed_breakdown"]:
            memory = result_dict["detailed_breakdown"]["memory"]
            if "by_type" in memory:
                activation = memory["by_type"].get("activation", 0)
                assert activation < 1.0, f"Activation should be peak (<1GB), got {activation}GB"
            elif "summary" in memory:
                activation = memory["summary"].get("activation_gb", 0)
                assert activation < 1.0, f"Activation should be peak (<1GB), got {activation}GB"

        denoise_phase = result.get_phase("denoise")
        if denoise_phase:
            assert denoise_phase.repeat_count == 50
            assert denoise_phase.memory_breakdown.get("activation_gb", 0) < 1.0

    def test_forward_pass_success(self):
        """Forward pass must succeed for diffusion model."""
        model = HunyuanImage3DiffusionModel(
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
            num_kv_heads=8,
            moe_intermediate_size=3072,
            num_experts=64,
            num_shared_experts=1,
            image_height=32,
            image_width=32,
            latent_channels=16,
        )

        batch = 1
        seq = 32 * 32

        latent = ShardedTensor(shape=(batch, seq, 16))
        timestep = ShardedTensor(shape=(batch, 256))

        output = model(latent, timestep)

        assert output is not None
        assert output.shape == (batch, seq, 16)

    def test_activation_memory_reasonable(self):
        """Activation memory should be in reasonable range.

        For diffusion inference with batch=1:
        - Forward mode peak should be ~0.1-0.5GB per device (with TP)
        - Should not exceed 10GB (would indicate incorrect calculation)
        """
        presets = get_model_presets()
        if "hunyuan-image-3" not in presets:
            pytest.skip("hunyuan-image-3 preset not available")

        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        model = create_model_from_config(
            {"preset": "hunyuan-image-3"}, workload_type="diffusion"
        )

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        result = analyzer.analyze(
            "diffusion-pipeline",
            height=1024,
            width=1024,
            num_steps=50,
        )

        result_dict = result.to_dict()

        if result_dict["detailed_breakdown"]:
            memory = result_dict["detailed_breakdown"]["memory"]
            if "by_type" in memory:
                activation = memory["by_type"].get("activation", 0)
            elif "summary" in memory:
                activation = memory["summary"].get("activation_gb", 0)
            else:
                activation = 0

            assert 0.01 < activation < 10.0, f"Activation unreasonable: {activation}GB"

        peak_memory = result_dict["peak_memory_gb"]
        assert peak_memory > 0
        assert peak_memory < 100, f"Peak memory should be reasonable: {peak_memory}GB"


class TestDiffusionForwardOnly:
    """Test that diffusion uses forward-only mode, not forward_backward."""

    def test_diffusion_inference_compute_type(self):
        """Diffusion inference phases should have compute_type=forward."""
        presets = get_model_presets()
        if "hunyuan-image-3" not in presets:
            pytest.skip("hunyuan-image-3 preset not available")

        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        models = {
            "encoder": create_model_from_config({"type": "hunyuan-t5-encoder"}),
            "backbone": create_model_from_config(
                {"preset": "hunyuan-image-3"}, workload_type="diffusion"
            ),
            "decoder": create_model_from_config({"type": "hunyuan-vae-decoder"}),
        }

        analyzer = UnifiedAnalyzer(models, device, cluster, strategy)
        result = analyzer.analyze("diffusion-pipeline", height=512, width=512, num_steps=10)

        for phase in result.phases:
            assert phase.compute_type.value == "forward"

    def test_no_backward_phase_in_diffusion(self):
        """Diffusion pipeline should not have backward phase."""
        presets = get_model_presets()
        if "hunyuan-image-3" not in presets:
            pytest.skip("hunyuan-image-3 preset not available")

        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        models = {
            "encoder": create_model_from_config({"type": "hunyuan-t5-encoder"}),
            "backbone": create_model_from_config(
                {"preset": "hunyuan-image-3"}, workload_type="diffusion"
            ),
            "decoder": create_model_from_config({"type": "hunyuan-vae-decoder"}),
        }

        analyzer = UnifiedAnalyzer(models, device, cluster, strategy)
        result = analyzer.analyze("diffusion-pipeline", height=512, width=512, num_steps=10)

        phase_names = [p.name for p in result.phases]
        assert "backward" not in phase_names

    def test_inference_memory_no_gradient_optimizer(self):
        """Diffusion inference should not have gradient or optimizer memory."""
        presets = get_model_presets()
        if "hunyuan-image-3" not in presets:
            pytest.skip("hunyuan-image-3 preset not available")

        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        models = {
            "encoder": create_model_from_config({"type": "hunyuan-t5-encoder"}),
            "backbone": create_model_from_config(
                {"preset": "hunyuan-image-3"}, workload_type="diffusion"
            ),
            "decoder": create_model_from_config({"type": "hunyuan-vae-decoder"}),
        }

        analyzer = UnifiedAnalyzer(models, device, cluster, strategy)
        result = analyzer.analyze("diffusion-pipeline", height=512, width=512, num_steps=10)

        result_dict = result.to_dict()

        if result_dict["detailed_breakdown"]:
            memory = result_dict["detailed_breakdown"]["memory"]
            if "summary" in memory:
                gradient = memory["summary"].get("gradient_gb", 0)
                optimizer = memory["summary"].get("optimizer_gb", 0)
                assert gradient == 0, f"Inference should not have gradient memory, got {gradient}GB"
                assert optimizer == 0, f"Inference should not have optimizer memory, got {optimizer}GB"