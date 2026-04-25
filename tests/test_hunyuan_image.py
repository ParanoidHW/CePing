"""Tests for HunyuanImage 3.0 Models.

HunyuanImage 3.0 family includes:
- Text model: Autoregressive text generation (~80B total, ~13B active)
- Diffusion model: Image generation (same backbone)

Key features:
- MoE FFN: 64 experts + 1 shared expert, top-8 routing
- QK Norm: Attention normalization
- SwiGLU: Expert activation
"""

import pytest

from llm_perf.modeling import (
    ShardedTensor,
    ParallelContext,
    create_model_from_config,
    get_model_presets,
)
from llm_perf.modeling.hunyuan_image import (
    HunyuanImage3TextModel,
    HunyuanImage3DiffusionModel,
    ShardedHunyuanMoEBlock,
    ShardedHunyuanAttention,
)


class TestHunyuanConfig:
    """Test HunyuanImage config validation."""

    def test_hunyuan_config_validation(self):
        """Test creating HunyuanImage model with config."""
        model = HunyuanImage3TextModel(
            vocab_size=133120,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            moe_intermediate_size=3072,
            num_experts=64,
            num_experts_per_token=8,
            num_shared_experts=1,
        )

        assert model.vocab_size == 133120
        assert model.hidden_size == 4096
        assert model.num_layers == 32
        assert model.num_experts == 64
        assert model.num_experts_per_token == 8
        assert model.num_shared_experts == 1

    def test_hunyuan_default_config(self):
        """Test HunyuanImage with default config."""
        model = HunyuanImage3TextModel()

        assert model.vocab_size == 133120
        assert model.hidden_size == 4096
        assert model.num_layers == 32
        assert model.num_heads == 32
        assert model.num_kv_heads == 8
        assert model.head_dim == 128
        assert model.moe_intermediate_size == 3072
        assert model.num_experts == 64
        assert model.num_experts_per_token == 8
        assert model.num_shared_experts == 1
        assert model.use_qk_norm == True


class TestShardedHunyuanMoEBlock:
    """Test ShardedHunyuanMoEBlock."""

    def test_hunyuan_moe_block_structure(self):
        """Test MoE Block has attention + moe submodules."""
        block = ShardedHunyuanMoEBlock(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            moe_intermediate_size=3072,
            num_experts=64,
            num_experts_per_token=8,
            num_shared_experts=1,
        )

        assert "attention" in block._submodules
        assert "moe" in block._submodules
        assert isinstance(block.attention, ShardedHunyuanAttention)
        assert block.num_experts == 64
        assert block.moe.num_experts == 64

    def test_hunyuan_moe_block_forward(self):
        """Test MoE block forward."""
        block = ShardedHunyuanMoEBlock(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            moe_intermediate_size=3072,
            num_experts=64,
            num_shared_experts=1,
        )

        hidden = ShardedTensor(shape=(1, 512, 4096))
        output = block(hidden)

        assert output.shape == (1, 512, 4096)
        assert "attn_out" in block._activations
        assert "moe_out" in block._activations

    def test_hunyuan_moe_block_shared_expert(self):
        """Test MoE block with shared expert."""
        block = ShardedHunyuanMoEBlock(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            moe_intermediate_size=3072,
            num_experts=64,
            num_shared_experts=1,
        )

        assert block.moe.shared_expert_intermediate == 3072


class TestShardedHunyuanAttention:
    """Test ShardedHunyuanAttention with QK Norm."""

    def test_hunyuan_attention_with_qk_norm(self):
        """Test Attention has QK Norm."""
        attn = ShardedHunyuanAttention(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            use_qk_norm=True,
        )

        assert attn.use_qk_norm == True
        assert "query_norm_weight" in attn._weights
        assert "key_norm_weight" in attn._weights

    def test_hunyuan_attention_without_qk_norm(self):
        """Test Attention without QK Norm."""
        attn = ShardedHunyuanAttention(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            use_qk_norm=False,
        )

        assert attn.use_qk_norm == False
        assert "query_norm_weight" not in attn._weights
        assert "key_norm_weight" not in attn._weights

    def test_hunyuan_attention_qkv_fused(self):
        """Test Attention uses fused QKV projection."""
        attn = ShardedHunyuanAttention(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
        )

        assert "qkv_weight" in attn._weights

        q_dim = 32 * 128
        kv_dim = 8 * 128
        expected_qkv_dim = q_dim + 2 * kv_dim

        assert attn.qkv_weight.shape == (4096, expected_qkv_dim)

    def test_hunyuan_attention_forward(self):
        """Test Attention forward."""
        attn = ShardedHunyuanAttention(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
        )

        hidden = ShardedTensor(shape=(1, 512, 4096))
        output = attn(hidden)

        assert output.shape == (1, 512, 4096)


class TestHunyuanTextModel:
    """Test HunyuanImage 3.0 Text Model."""

    def test_hunyuan_text_model_params(self):
        """Test Text model params ~80B total, ~13B active.

        Params breakdown:
        - Embedding: 133120 * 4096 = 546M
        - Each layer:
          - Input norm: 4096
          - Attention: 4096 * (32*128 + 2*8*128) + 32*128 * 4096 = 4096 * 6144 + 4096 * 4096 = 25M + 16M = 41M
            - Plus QK norm: 128 + 128 = 256 (negligible)
          - Post attn norm: 4096
          - MoE routed: 64 * (4096 * 3072 * 2 + 3072 * 2 * 4096) = 64 * 25M = 1.6B
          - MoE shared: 4096 * 3072 * 2 + 3072 * 2 * 4096 = 25M
          - Router: 4096 * 64 = 0.26M
        - Final norm: 4096
        - LM head: 133120 * 4096 = 546M
        """
        model = HunyuanImage3TextModel(
            vocab_size=133120,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            moe_intermediate_size=3072,
            num_experts=64,
            num_experts_per_token=8,
            num_shared_experts=1,
        )

        params = model.params_count()

        assert params > 50e9, f"HunyuanImage should have >50B params, got {params / 1e9:.2f}B"
        assert params < 100e9, f"HunyuanImage should have <100B params, got {params / 1e9:.2f}B"

    def test_hunyuan_text_inference_eval(self):
        """Test Text inference evaluation."""
        model = HunyuanImage3TextModel(
            vocab_size=133120,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
            num_kv_heads=8,
            moe_intermediate_size=3072,
            num_experts=64,
            num_shared_experts=1,
        )

        input_ids = ShardedTensor(shape=(1, 512))
        logits = model(input_ids)

        assert logits.shape == (1, 512, 133120)
        assert "embedding_output" in model._activations
        assert "layer_0_output" in model._activations
        assert "final_norm_output" in model._activations

    def test_hunyuan_text_module_instance(self):
        """Test Text model instance with TP."""
        model = HunyuanImage3TextModel(
            vocab_size=133120,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
            num_kv_heads=8,
            moe_intermediate_size=3072,
            num_experts=64,
        )

        ctx = ParallelContext(tp_degree=8, ep_degree=2)
        instance = model.bind(ctx)

        assert instance.params_count_logical == model.params_count()
        assert instance.params_count_physical < instance.params_count_logical

    def test_hunyuan_text_forward_with_op_history(self):
        """Test forward produces op history."""
        model = HunyuanImage3TextModel(
            vocab_size=133120,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
            num_kv_heads=8,
            moe_intermediate_size=3072,
            num_experts=64,
        )

        input_ids = ShardedTensor(shape=(1, 128))
        logits = model(input_ids)

        assert len(logits._op_history) > 0


class TestHunyuanDiffusionModel:
    """Test HunyuanImage 3.0 Diffusion Model."""

    def test_hunyuan_diffusion_model_structure(self):
        """Test Diffusion model structure."""
        model = HunyuanImage3DiffusionModel(
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            moe_intermediate_size=3072,
            num_experts=64,
            num_shared_experts=1,
            image_height=64,
            image_width=64,
            latent_channels=16,
        )

        assert model.hidden_size == 4096
        assert model.num_layers == 32
        assert model.num_experts == 64
        assert model.image_height == 64
        assert model.image_width == 64
        assert model.seq_len == 64 * 64

    def test_hunyuan_diffusion_eval(self):
        """Test Diffusion evaluation (50 steps)."""
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

        assert output.shape == (batch, seq, 16)
        assert "input_proj_output" in model._activations
        assert "timestep_embedding" in model._activations
        assert "layer_0_output" in model._activations

    def test_hunyuan_diffusion_params(self):
        """Test Diffusion model params (same backbone as Text)."""
        model = HunyuanImage3DiffusionModel(
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            moe_intermediate_size=3072,
            num_experts=64,
            num_shared_experts=1,
        )

        params = model.params_count()

        assert params > 50e9, f"HunyuanImage Diffusion should have >50B params, got {params / 1e9:.2f}B"
        assert params < 100e9, f"HunyuanImage Diffusion should have <100B params, got {params / 1e9:.2f}B"

    def test_hunyuan_diffusion_module_instance(self):
        """Test Diffusion model instance with TP."""
        model = HunyuanImage3DiffusionModel(
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
            num_kv_heads=8,
            moe_intermediate_size=3072,
            num_experts=64,
        )

        ctx = ParallelContext(tp_degree=8, ep_degree=2)
        instance = model.bind(ctx)

        assert instance.params_count_logical == model.params_count()
        assert instance.params_count_physical < instance.params_count_logical


class TestHunyuanParamsBreakdown:
    """Test HunyuanImage params breakdown."""

    def test_hunyuan_attention_breakdown(self):
        """Test attention params breakdown."""
        model = HunyuanImage3TextModel(
            vocab_size=133120,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
            num_kv_heads=8,
            moe_intermediate_size=3072,
            num_experts=64,
        )

        breakdown = model.params_count_breakdown()

        attention_params = sum(
            v for k, v in breakdown.items() if "attention" in k and "moe" not in k
        )
        assert attention_params > 0

    def test_hunyuan_moe_breakdown(self):
        """Test MoE params breakdown."""
        model = HunyuanImage3TextModel(
            vocab_size=133120,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
            num_kv_heads=8,
            moe_intermediate_size=3072,
            num_experts=64,
            num_shared_experts=1,
        )

        breakdown = model.params_count_breakdown()

        moe_params = sum(v for k, v in breakdown.items() if "moe" in k)
        assert moe_params > 0

    def test_hunyuan_qk_norm_params(self):
        """Test QK Norm params (head_dim dimension)."""
        attn = ShardedHunyuanAttention(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            use_qk_norm=True,
        )

        breakdown = attn.params_count_breakdown()

        qk_norm_params = breakdown.get("query_norm_weight", 0) + breakdown.get("key_norm_weight", 0)
        assert qk_norm_params == 128 + 128  # head_dim * 2


class TestHunyuanMoEActiveParams:
    """Test HunyuanImage MoE active params calculation."""

    def test_hunyuan_moe_active_params(self):
        """Test MoE active params calculation.

        Total MoE params include:
        - Router: hidden_size * num_experts
        - Routed experts: num_experts * (gate + up + down) params
        - Shared expert: (gate + up + down) params
        """
        block = ShardedHunyuanMoEBlock(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            moe_intermediate_size=3072,
            num_experts=64,
            num_experts_per_token=8,
            num_shared_experts=1,
        )

        total_params = block.params_count()

        assert total_params > 1e9

        moe_total_params = block.moe.params_count()
        routed_expert_params = (
            4096 * 3072 * 2 + 3072 * 2 * 4096
        )

        assert moe_total_params > routed_expert_params


class TestHunyuanFromConfig:
    """Test creating HunyuanImage from config."""

    def test_create_from_preset_with_architecture_text(self):
        """Test creating HunyuanImage Text from preset with architecture."""
        presets = get_model_presets()
        if "hunyuan-image-3" not in presets:
            pytest.skip("hunyuan-image-3 preset not available")

        model = create_model_from_config({"preset": "hunyuan-image-3", "architecture": "hunyuan_image_3_text"})
        assert isinstance(model, HunyuanImage3TextModel)
        assert model.hidden_size == 4096
        assert model.num_layers == 32

    def test_create_from_preset_with_architecture_diffusion(self):
        """Test creating HunyuanImage Diffusion from preset with architecture."""
        presets = get_model_presets()
        if "hunyuan-image-3" not in presets:
            pytest.skip("hunyuan-image-3 preset not available")

        model = create_model_from_config({"preset": "hunyuan-image-3", "architecture": "hunyuan_image_3_diffusion"})
        assert isinstance(model, HunyuanImage3DiffusionModel)
        assert model.hidden_size == 4096
        assert model.num_layers == 32

    def test_create_from_type_text(self):
        """Test creating HunyuanImage Text from type field."""
        model = create_model_from_config({"type": "hunyuan_image_3_text"})
        assert isinstance(model, HunyuanImage3TextModel)

    def test_create_from_type_diffusion(self):
        """Test creating HunyuanImage Diffusion from type field."""
        model = create_model_from_config({"type": "hunyuan_image_3_diffusion"})
        assert isinstance(model, HunyuanImage3DiffusionModel)


class TestPresetWorkloadMapping:
    """Test preset to model class mapping based on workload type."""

    def test_hunyuan_image_3_inference_mapping(self):
        """Test hunyuan-image-3 + inference → hunyuan_image_3_text."""
        presets = get_model_presets()
        if "hunyuan-image-3" not in presets:
            pytest.skip("hunyuan-image-3 preset not available")

        model = create_model_from_config(
            {"preset": "hunyuan-image-3"},
            workload_type="inference"
        )
        assert isinstance(model, HunyuanImage3TextModel)
        assert model.hidden_size == 4096
        assert model.num_layers == 32

    def test_hunyuan_image_3_diffusion_mapping(self):
        """Test hunyuan-image-3 + diffusion → hunyuan_image_3_diffusion."""
        presets = get_model_presets()
        if "hunyuan-image-3" not in presets:
            pytest.skip("hunyuan-image-3 preset not available")

        model = create_model_from_config(
            {"preset": "hunyuan-image-3"},
            workload_type="diffusion"
        )
        assert isinstance(model, HunyuanImage3DiffusionModel)
        assert model.hidden_size == 4096
        assert model.num_layers == 32

    def test_hunyuan_image_3_training_mapping(self):
        """Test hunyuan-image-3 + training → inference model (no training map)."""
        presets = get_model_presets()
        if "hunyuan-image-3" not in presets:
            pytest.skip("hunyuan-image-3 preset not available")

        model = create_model_from_config(
            {"preset": "hunyuan-image-3", "architecture": "hunyuan_image_3_text"},
            workload_type="training"
        )
        assert isinstance(model, HunyuanImage3TextModel)
        assert model.hidden_size == 4096
        assert model.num_layers == 32

    def test_llama_inference_mapping(self):
        """Test llama-7b + inference → llama (no model_class_map)."""
        model = create_model_from_config(
            {"preset": "llama-7b"},
            workload_type="inference"
        )
        from llm_perf.modeling.models import LlamaModel
        assert isinstance(model, LlamaModel)

    def test_model_class_map_in_preset(self):
        """Test that model_class_map is present in hunyuan-image-3 preset."""
        presets = get_model_presets()
        if "hunyuan-image-3" not in presets:
            pytest.skip("hunyuan-image-3 preset not available")

        preset = presets["hunyuan-image-3"]
        assert "model_class_map" in preset
        assert preset["model_class_map"]["inference"] == "hunyuan_image_3_text"
        assert preset["model_class_map"]["diffusion"] == "hunyuan_image_3_diffusion"

    def test_no_model_class_map_uses_architecture(self):
        """Test that presets without model_class_map use architecture."""
        model = create_model_from_config(
            {"preset": "deepseek-v3"},
            workload_type="training"
        )
        from llm_perf.modeling.models import DeepSeekModel
        assert isinstance(model, DeepSeekModel)