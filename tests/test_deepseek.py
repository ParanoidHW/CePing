"""Tests for DeepSeek model classes with MLA support."""

import unittest

from llm_perf.models.deepseek import (
    DeepSeekConfig,
    DeepSeekV3Config,
    DeepSeekModel,
    DeepSeekV2Model,
    DeepSeekV3Model,
)


class TestDeepSeekConfig(unittest.TestCase):
    """Test DeepSeekConfig creation and validation."""

    def test_basic_creation(self):
        """Test creating a basic DeepSeekConfig."""
        config = DeepSeekConfig(
            name="test-deepseek",
            vocab_size=102400,
            hidden_size=5120,
            num_layers=60,
            num_attention_heads=128,
        )
        self.assertEqual(config.name, "test-deepseek")
        self.assertEqual(config.vocab_size, 102400)
        self.assertEqual(config.hidden_size, 5120)
        self.assertEqual(config.num_layers, 60)
        self.assertEqual(config.num_attention_heads, 128)

    def test_mla_default_params(self):
        """Test MLA default parameter values."""
        config = DeepSeekConfig(
            name="test",
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
        )
        # Default MLA params
        self.assertEqual(config.kv_lora_rank, 512)
        self.assertEqual(config.q_lora_rank, 1536)
        self.assertEqual(config.qk_nope_head_dim, 128)
        self.assertEqual(config.qk_rope_head_dim, 64)
        self.assertEqual(config.v_head_dim, 128)

    def test_moe_default_params(self):
        """Test MoE default parameter values."""
        config = DeepSeekConfig(
            name="test",
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
        )
        # Default MoE params
        self.assertEqual(config.n_routed_experts, 160)
        self.assertEqual(config.n_shared_experts, 2)
        self.assertEqual(config.num_experts_per_tok, 6)
        self.assertEqual(config.first_k_dense_replace, 1)

    def test_custom_mla_params(self):
        """Test custom MLA parameter values."""
        config = DeepSeekConfig(
            name="test",
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
            kv_lora_rank=256,
            q_lora_rank=512,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )
        self.assertEqual(config.kv_lora_rank, 256)
        self.assertEqual(config.q_lora_rank, 512)
        self.assertEqual(config.qk_nope_head_dim, 64)
        self.assertEqual(config.qk_rope_head_dim, 32)
        self.assertEqual(config.v_head_dim, 64)

    def test_custom_moe_params(self):
        """Test custom MoE parameter values."""
        config = DeepSeekConfig(
            name="test",
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
            n_routed_experts=256,
            n_shared_experts=1,
            num_experts_per_tok=8,
            first_k_dense_replace=2,
        )
        self.assertEqual(config.n_routed_experts, 256)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.first_k_dense_replace, 2)

    def test_intermediate_size_default(self):
        """Test intermediate size default calculation."""
        config = DeepSeekConfig(
            name="test",
            vocab_size=1000,
            hidden_size=1000,  # Easy to calculate
            num_layers=2,
            num_attention_heads=4,
        )
        # Default is hidden_size * 12 / 5 = 2400
        self.assertEqual(config.intermediate_size, 2400)

    def test_post_init_key_value_heads(self):
        """Test that num_key_value_heads defaults correctly."""
        config = DeepSeekConfig(
            name="test",
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=8,
        )
        self.assertEqual(config.num_key_value_heads, 8)


class TestDeepSeekV3Config(unittest.TestCase):
    """Test DeepSeekV3Config with V3-specific parameters."""

    def test_v3_default_params(self):
        """Test V3 default parameters match official config."""
        config = DeepSeekV3Config(name="deepseek-v3")
        
        # V3-specific values from official config.json
        self.assertEqual(config.hidden_size, 7168)
        self.assertEqual(config.num_layers, 61)
        self.assertEqual(config.vocab_size, 129280)
        self.assertEqual(config.intermediate_size, 18432)
        
        # V3 MoE configuration
        self.assertEqual(config.n_routed_experts, 256)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.num_experts_per_tok, 8)

    def test_v3_inherits_v2_mla(self):
        """Test V3 inherits MLA structure from V2."""
        config = DeepSeekV3Config(name="deepseek-v3")
        
        # MLA params should be consistent
        self.assertEqual(config.kv_lora_rank, 512)
        self.assertEqual(config.q_lora_rank, 1536)
        self.assertEqual(config.qk_nope_head_dim, 128)
        self.assertEqual(config.qk_rope_head_dim, 64)
        self.assertEqual(config.v_head_dim, 128)


class TestDeepSeekModel(unittest.TestCase):
    """Test DeepSeekModel layer building."""

    def setUp(self):
        """Set up test configuration."""
        self.config = DeepSeekConfig(
            name="tiny-deepseek",
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            max_seq_len=128,
            dtype="fp16",
            kv_lora_rank=64,
            q_lora_rank=128,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            n_routed_experts=8,
            n_shared_experts=2,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
        )

    def test_model_creation(self):
        """Test creating a DeepSeekModel."""
        model = DeepSeekModel(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.config.name, "tiny-deepseek")

    def test_total_params_positive(self):
        """Test that total parameters is positive."""
        model = DeepSeekModel(self.config)
        self.assertGreater(model.total_params, 0)

    def test_layers_exist(self):
        """Test that expected layers are created."""
        model = DeepSeekModel(self.config)
        names = [layer.name for layer in model.layers]
        
        # Check embedding and output layers
        self.assertIn("embedding", names)
        self.assertIn("final_norm", names)
        self.assertIn("lm_head", names)

    def test_mla_layers_exist(self):
        """Test that MLA-specific layers are created."""
        model = DeepSeekModel(self.config)
        names = [layer.name for layer in model.layers]
        
        # Check MLA-specific layers
        self.assertIn("layer_0_q_down_proj", names)
        self.assertIn("layer_0_q_up_proj", names)
        self.assertIn("layer_0_kv_down_proj", names)
        self.assertIn("layer_0_k_up_proj", names)
        self.assertIn("layer_0_v_up_proj", names)
        self.assertIn("layer_0_q_rope_proj", names)
        self.assertIn("layer_0_k_rope_proj", names)
        self.assertIn("layer_0_mla_attention", names)

    def test_moe_layers_exist(self):
        """Test that MoE-specific layers are created."""
        model = DeepSeekModel(self.config)
        names = [layer.name for layer in model.layers]
        
        # Layer 0 should have dense FFN (first_k_dense_replace=1)
        self.assertIn("layer_0_up_proj", names)
        
        # Layer 1+ should have MoE
        self.assertIn("layer_1_router", names)
        self.assertIn("layer_1_routed_up", names)
        self.assertIn("layer_1_shared_up", names)

    def test_get_layer_by_name(self):
        """Test retrieving a layer by name."""
        model = DeepSeekModel(self.config)
        layer = model.get_layer_by_name("embedding")
        self.assertIsNotNone(layer)
        self.assertEqual(layer.name, "embedding")
        self.assertIsNone(model.get_layer_by_name("nonexistent"))

    def test_to_dict_structure(self):
        """Test that to_dict produces expected structure."""
        model = DeepSeekModel(self.config)
        data = model.to_dict()
        
        self.assertEqual(data["name"], "tiny-deepseek")
        self.assertIn("config", data)
        self.assertIn("total_params", data)
        self.assertIn("layers", data)
        self.assertEqual(len(data["layers"]), len(model.layers))

    def test_flops_forward_backward(self):
        """Test forward and backward FLOPs relationship."""
        model = DeepSeekModel(self.config)
        self.assertEqual(model.total_flops_backward, model.total_flops_forward * 2)

    def test_activation_memory(self):
        """Test activation memory is positive."""
        model = DeepSeekModel(self.config)
        self.assertGreater(model.activation_memory, 0)


class TestMLAFeatures(unittest.TestCase):
    """Test MLA-specific features."""

    def setUp(self):
        """Set up test configuration."""
        self.config = DeepSeekConfig(
            name="test-mla",
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            kv_lora_rank=64,
            q_lora_rank=128,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
        )
        self.model = DeepSeekModel(self.config)

    def test_mla_kv_cache_size(self):
        """Test MLA KV cache size calculation."""
        batch_size = 2
        seq_len = 1024
        
        mla_size = self.model.get_mla_kv_cache_size(batch_size, seq_len)
        # 2 * 1024 * 64 * 2 bytes (fp16)
        expected = batch_size * seq_len * self.config.kv_lora_rank * 2
        self.assertEqual(mla_size, expected)

    def test_standard_kv_cache_size(self):
        """Test standard KV cache size calculation."""
        batch_size = 2
        seq_len = 1024
        
        standard_size = self.model.get_standard_kv_cache_size(batch_size, seq_len)
        # 2 * 1024 * 4 * (32+16) * 2 (K+V) * 2 bytes
        kv_heads = 4
        head_dim = 32 + 16
        expected = batch_size * seq_len * kv_heads * head_dim * 2 * 2
        self.assertEqual(standard_size, expected)

    def test_kv_cache_compression_ratio(self):
        """Test KV cache compression ratio calculation."""
        ratio = self.model.get_kv_cache_compression_ratio()
        
        # Standard: kv_heads * head_dim * 2 (K+V)
        # MLA: kv_lora_rank
        kv_heads = 4
        head_dim = 32 + 16
        standard_size = kv_heads * head_dim * 2
        mla_size = 64
        expected_ratio = standard_size / mla_size
        
        self.assertEqual(ratio, expected_ratio)
        self.assertGreater(ratio, 1.0)  # MLA should compress

    def test_mla_saves_memory(self):
        """Test that MLA actually saves memory."""
        batch_size = 4
        seq_len = 2048
        
        mla_size = self.model.get_mla_kv_cache_size(batch_size, seq_len)
        standard_size = self.model.get_standard_kv_cache_size(batch_size, seq_len)
        
        self.assertLess(mla_size, standard_size)
        # Should be significant compression (at least 2x)
        self.assertGreater(standard_size / mla_size, 2.0)


class TestDeepSeekV2Model(unittest.TestCase):
    """Test DeepSeekV2Model with official parameters."""

    def test_v2_official_params(self):
        """Test V2 model uses official HuggingFace parameters."""
        model = DeepSeekV2Model()
        config = model.config
        
        # Verify key parameters from official config.json
        self.assertEqual(config.name, "deepseek-v2")
        self.assertEqual(config.vocab_size, 102400)
        self.assertEqual(config.hidden_size, 5120)
        self.assertEqual(config.num_layers, 60)
        self.assertEqual(config.num_attention_heads, 128)
        self.assertEqual(config.intermediate_size, 12288)
        
        # MLA params
        self.assertEqual(config.kv_lora_rank, 512)
        self.assertEqual(config.q_lora_rank, 1536)
        
        # MoE params
        self.assertEqual(config.n_routed_experts, 160)
        self.assertEqual(config.n_shared_experts, 2)

    def test_v2_model_builds(self):
        """Test V2 model builds successfully."""
        model = DeepSeekV2Model()
        self.assertGreater(len(model.layers), 0)
        self.assertGreater(model.total_params, 0)

    def test_v2_kv_compression(self):
        """Test V2 MLA compression ratio."""
        model = DeepSeekV2Model()
        ratio = model.get_kv_cache_compression_ratio()
        
        # V2: 128 heads * (128+64) dims * 2 (K+V) / 512 latent
        # = 128 * 192 * 2 / 512 = 96
        self.assertGreater(ratio, 50.0)  # Should be ~96x compression


class TestDeepSeekV3Model(unittest.TestCase):
    """Test DeepSeekV3Model with official parameters."""

    def test_v3_official_params(self):
        """Test V3 model uses official HuggingFace parameters."""
        model = DeepSeekV3Model()
        config = model.config
        
        # Verify key parameters from official config.json
        self.assertEqual(config.name, "deepseek-v3")
        self.assertEqual(config.vocab_size, 129280)
        self.assertEqual(config.hidden_size, 7168)
        self.assertEqual(config.num_layers, 61)
        self.assertEqual(config.num_attention_heads, 128)
        self.assertEqual(config.intermediate_size, 18432)
        
        # MLA params (same as V2)
        self.assertEqual(config.kv_lora_rank, 512)
        self.assertEqual(config.q_lora_rank, 1536)
        
        # V3 MoE params (scaled up)
        self.assertEqual(config.n_routed_experts, 256)
        self.assertEqual(config.n_shared_experts, 1)

    def test_v3_model_builds(self):
        """Test V3 model builds successfully."""
        model = DeepSeekV3Model()
        self.assertGreater(len(model.layers), 0)
        self.assertGreater(model.total_params, 0)

    def test_v3_larger_than_v2(self):
        """Test V3 is larger than V2."""
        v2_model = DeepSeekV2Model()
        v3_model = DeepSeekV3Model()
        
        self.assertGreater(v3_model.total_params, v2_model.total_params)


class TestDeepSeekMoELayers(unittest.TestCase):
    """Test DeepSeek MoE layer structure."""

    def setUp(self):
        """Set up test configuration with MoE."""
        self.config = DeepSeekConfig(
            name="test-moe",
            vocab_size=1000,
            hidden_size=256,
            num_layers=8,
            num_attention_heads=4,
            n_routed_experts=16,
            n_shared_experts=2,
            num_experts_per_tok=4,
            first_k_dense_replace=2,  # First 2 layers are dense
        )
        self.model = DeepSeekModel(self.config)

    def test_dense_layers_first(self):
        """Test first k layers use dense FFN."""
        names = [layer.name for layer in self.model.layers]
        
        # Layer 0 should be dense
        self.assertIn("layer_0_up_proj", names)
        self.assertNotIn("layer_0_router", names)
        
        # Layer 1 should be dense
        self.assertIn("layer_1_up_proj", names)
        self.assertNotIn("layer_1_router", names)

    def test_moe_layers_after(self):
        """Test layers after k use MoE."""
        names = [layer.name for layer in self.model.layers]
        
        # Layer 2+ should be MoE
        self.assertIn("layer_2_router", names)
        self.assertIn("layer_3_router", names)

    def test_moe_has_routed_and_shared(self):
        """Test MoE layers have both routed and shared experts."""
        names = [layer.name for layer in self.model.layers]
        
        # Should have both routed and shared components
        self.assertIn("layer_2_routed_up", names)
        self.assertIn("layer_2_routed_gate", names)
        self.assertIn("layer_2_shared_up", names)
        self.assertIn("layer_2_shared_gate", names)

    def test_moe_layers_marked(self):
        """Test MoE layers are marked with is_moe=True."""
        moe_layers = [layer for layer in self.model.layers if layer.is_moe]
        
        # Should have MoE layers
        self.assertGreater(len(moe_layers), 0)
        
        # All MoE layers should have is_moe=True
        for layer in moe_layers:
            self.assertTrue(layer.is_moe)


if __name__ == "__main__":
    unittest.main()
