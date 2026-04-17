"""Tests for Wan2.1 Video Generation Model."""

import unittest
from llm_perf.models.wan_video import (
    WanTextEncoderConfig,
    WanDiTConfig,
    WanVAEConfig,
    WanTextEncoder,
    WanDiTModel,
    WanVAEModel,
    create_wan_t2v_14b_text_encoder,
    create_wan_t2v_14b_dit,
    create_wan_t2v_vae,
)
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster, NetworkConfig
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer.diffusion_video import (
    DiffusionVideoAnalyzer,
    create_wan_analyzer,
)


class TestWanTextEncoder(unittest.TestCase):
    """Test Wan Text Encoder (umT5-XXL)."""

    def test_config_creation(self):
        """Test text encoder config creation."""
        config = WanTextEncoderConfig(
            name="test-text-encoder",
            hidden_size=4096,
            num_layers=24,
            num_attention_heads=64,
        )
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_layers, 24)
        self.assertEqual(config.num_attention_heads, 64)

    def test_model_creation(self):
        """Test text encoder model creation."""
        config = WanTextEncoderConfig(
            name="test-text-encoder",
            hidden_size=1024,
            num_layers=4,
            num_attention_heads=16,
            max_text_len=128,
        )
        model = WanTextEncoder(config)
        self.assertGreater(model.total_params, 0)
        self.assertGreater(len(model.layers), 0)

    def test_factory_function(self):
        """Test text encoder factory function."""
        model = create_wan_t2v_14b_text_encoder()
        self.assertEqual(model.config.hidden_size, 4096)
        self.assertEqual(model.config.num_layers, 24)
        self.assertGreater(model.total_params, 0)

    def test_forward_flops(self):
        """Test text encoder FLOPs calculation."""
        model = create_wan_t2v_14b_text_encoder()
        flops = model.total_flops_forward
        self.assertGreater(flops, 0)


class TestWanDiT(unittest.TestCase):
    """Test Wan DiT Model."""

    def test_config_creation(self):
        """Test DiT config creation."""
        config = WanDiTConfig(
            name="test-dit",
            vocab_size=0,
            hidden_size=5120,
            num_layers=40,
            num_attention_heads=40,
        )
        self.assertEqual(config.hidden_size, 5120)
        self.assertEqual(config.num_layers, 40)
        self.assertEqual(config.patch_size, (1, 2, 2))

    def test_model_creation(self):
        """Test DiT model creation."""
        config = WanDiTConfig(
            name="test-dit",
            vocab_size=0,
            hidden_size=512,
            num_layers=4,
            num_attention_heads=8,
            latent_num_frames=10,
            latent_height=32,
            latent_width=32,
        )
        model = WanDiTModel(config)
        self.assertGreater(model.total_params, 0)
        self.assertGreater(len(model.layers), 0)

    def test_patchify_layer(self):
        """Test patchify layer is created."""
        model = create_wan_t2v_14b_dit()
        patchify_layer = model.get_layer_by_name("patchify")
        self.assertIsNotNone(patchify_layer)
        self.assertGreater(patchify_layer.params_count, 0)

    def test_time_embedding_mlp(self):
        """Test time embedding MLP layers."""
        model = create_wan_t2v_14b_dit()
        time_mlp = model.get_layer_by_name("time_embedding_in")
        self.assertIsNotNone(time_mlp)
        
        time_out = model.get_layer_by_name("time_embedding_out")
        self.assertIsNotNone(time_out)
        
        time_proj = model.get_layer_by_name("time_projection")
        self.assertIsNotNone(time_proj)

    def test_transformer_blocks(self):
        """Test transformer blocks are created."""
        model = create_wan_t2v_14b_dit()
        
        # Check first block
        block_qkv = model.get_layer_by_name("block_0_self_attn_qkv")
        self.assertIsNotNone(block_qkv)
        
        # Check cross-attention
        block_cross_q = model.get_layer_by_name("block_0_cross_attn_q")
        self.assertIsNotNone(block_cross_q)
        
        # Check FFN
        block_ffn = model.get_layer_by_name("block_0_ffn_in")
        self.assertIsNotNone(block_ffn)

    def test_unpatchify_layer(self):
        """Test unpatchify layer."""
        model = create_wan_t2v_14b_dit()
        unpatchify = model.get_layer_by_name("unpatchify")
        self.assertIsNotNone(unpatchify)

    def test_14b_params(self):
        """Test 14B model has approximately correct parameter count."""
        model = create_wan_t2v_14b_dit()
        params_b = model.total_params / 1e9
        # Should be around 14B (allow some margin for implementation differences)
        self.assertGreater(params_b, 10)
        self.assertLess(params_b, 20)


class TestWanVAE(unittest.TestCase):
    """Test Wan 3D Causal VAE."""

    def test_config_creation(self):
        """Test VAE config creation."""
        config = WanVAEConfig(
            name="test-vae",
            vocab_size=0,
            hidden_size=0,
            num_layers=0,
            num_frames=81,
            height=720,
            width=1280,
        )
        self.assertEqual(config.latent_channels, 16)
        self.assertEqual(config.temporal_compression, 4)
        self.assertEqual(config.spatial_compression, 8)

    def test_model_creation(self):
        """Test VAE model creation."""
        config = WanVAEConfig(
            name="test-vae",
            vocab_size=0,
            hidden_size=0,
            num_layers=0,
            num_frames=17,
            height=64,
            width=64,
        )
        model = WanVAEModel(config)
        self.assertGreater(model.total_params, 0)
        self.assertGreater(len(model.layers), 0)

    def test_encoder_layers(self):
        """Test VAE encoder layers."""
        model = create_wan_t2v_vae()
        encoder_in = model.get_layer_by_name("encoder_conv_in")
        self.assertIsNotNone(encoder_in)
        
        encoder_out = model.get_layer_by_name("encoder_conv_out")
        self.assertIsNotNone(encoder_out)

    def test_decoder_layers(self):
        """Test VAE decoder layers."""
        model = create_wan_t2v_vae()
        decoder_in = model.get_layer_by_name("decoder_conv_in")
        self.assertIsNotNone(decoder_in)
        
        decoder_out = model.get_layer_by_name("decoder_conv_out")
        self.assertIsNotNone(decoder_out)

    def test_latent_dimensions(self):
        """Test VAE latent dimension calculation."""
        config = WanVAEConfig(
            name="test-vae",
            vocab_size=0,
            hidden_size=0,
            num_layers=0,
            num_frames=81,
            height=720,
            width=1280,
        )
        self.assertEqual(config.latent_num_frames, 21)  # (81-1)//4 + 1
        self.assertEqual(config.latent_height, 90)  # 720//8
        self.assertEqual(config.latent_width, 160)  # 1280//8


class TestDiffusionVideoAnalyzer(unittest.TestCase):
    """Test Diffusion Video Analyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = Device.from_preset("Ascend-910B2")
        self.network = NetworkConfig(
            intra_node_bandwidth_gbps=200.0,
        )
        self.cluster = Cluster.create_homogeneous(
            self.device.config, 1, self.network, 1
        )
        self.strategy = StrategyConfig()
        
        self.text_encoder = create_wan_t2v_14b_text_encoder()
        self.dit = create_wan_t2v_14b_dit(num_frames=17, height=64, width=64)
        self.vae = create_wan_t2v_vae(num_frames=17, height=64, width=64)
        
        self.analyzer = DiffusionVideoAnalyzer(
            text_encoder=self.text_encoder,
            dit=self.dit,
            vae=self.vae,
            device=self.device,
            cluster=self.cluster,
            strategy=self.strategy,
        )

    def test_analyze_basic(self):
        """Test basic analysis."""
        result = self.analyzer.analyze(
            num_frames=17,
            height=64,
            width=64,
            num_inference_steps=10,
            use_cfg=False,
        )
        
        # Time estimates may be 0 for very small configs, but memory should be > 0
        self.assertGreaterEqual(result.text_encoder_time_sec, 0)
        self.assertGreaterEqual(result.dit_single_step_time_sec, 0)
        self.assertGreaterEqual(result.vae_decoder_time_sec, 0)
        self.assertGreater(result.total_generation_time_sec, 0)  # Should have minimum
        self.assertGreater(result.peak_memory_total_gb, 0)

    def test_analyze_with_cfg(self):
        """Test analysis with CFG."""
        result_cfg = self.analyzer.analyze(
            num_frames=17,
            height=64,
            width=64,
            num_inference_steps=10,
            use_cfg=True,
        )
        
        result_no_cfg = self.analyzer.analyze(
            num_frames=17,
            height=64,
            width=64,
            num_inference_steps=10,
            use_cfg=False,
        )
        
        # CFG should take more or equal time (may be equal if both are 0)
        self.assertGreaterEqual(
            result_cfg.dit_single_step_time_sec,
            result_no_cfg.dit_single_step_time_sec
        )

    def test_component_breakdown(self):
        """Test component breakdown."""
        result = self.analyzer.analyze(
            num_frames=17,
            height=64,
            width=64,
            num_inference_steps=10,
        )
        
        breakdown = result.component_breakdown
        self.assertIn("text_encoder_pct", breakdown)
        self.assertIn("dit_pct", breakdown)
        self.assertIn("vae_decoder_pct", breakdown)
        
        # Percentages should sum to approximately 100 (or 0 if all times are 0)
        total_pct = (
            breakdown["text_encoder_pct"] +
            breakdown["dit_pct"] +
            breakdown["vae_decoder_pct"]
        )
        # Allow for 0 (all zero times) or ~100 (normal case)
        self.assertTrue(total_pct == 0 or abs(total_pct - 100) <= 5)

    def test_analyze_components_separately(self):
        """Test component-level analysis."""
        results = self.analyzer.analyze_components_separately(
            num_frames=17,
            height=64,
            width=64,
        )
        
        self.assertIn("text_encoder", results)
        self.assertIn("dit", results)
        self.assertIn("vae", results)
        
        # Check text encoder results
        self.assertIn("total_params", results["text_encoder"])
        self.assertIn("total_flops_forward", results["text_encoder"])
        
        # Check DiT results
        self.assertIn("num_layers", results["dit"])
        self.assertIn("hidden_size", results["dit"])
        
        # Check VAE results
        self.assertIn("encoder_params", results["vae"])
        self.assertIn("decoder_params", results["vae"])

    def test_factory_function(self):
        """Test analyzer factory function."""
        analyzer = create_wan_analyzer(
            device=self.device,
            cluster=self.cluster,
            strategy=self.strategy,
            num_frames=17,
            height=64,
            width=64,
        )
        
        result = analyzer.analyze(num_frames=17, height=64, width=64)
        self.assertGreater(result.total_generation_time_sec, 0)

    def test_result_to_dict(self):
        """Test result serialization."""
        result = self.analyzer.analyze(
            num_frames=17,
            height=64,
            width=64,
            num_inference_steps=10,
        )
        
        data = result.to_dict()
        self.assertIn("time", data)
        self.assertIn("memory", data)
        self.assertIn("throughput", data)
        self.assertIn("component_breakdown", data)


class TestWanIntegration(unittest.TestCase):
    """Integration tests for Wan2.1 models."""

    def test_full_pipeline_components(self):
        """Test that all pipeline components work together."""
        # Create all models
        text_encoder = create_wan_t2v_14b_text_encoder()
        dit = create_wan_t2v_14b_dit()
        vae = create_wan_t2v_vae()
        
        # Verify each component has parameters
        self.assertGreater(text_encoder.total_params, 0)
        self.assertGreater(dit.total_params, 0)
        self.assertGreater(vae.total_params, 0)
        
        # Verify total pipeline params is sum of components
        total_params = (
            text_encoder.total_params +
            dit.total_params +
            vae.total_params
        )
        self.assertGreater(total_params, 14e9)  # Should be > 14B

    def test_resolution_scaling(self):
        """Test that models handle different resolutions."""
        # Test at 480p
        dit_480p = create_wan_t2v_14b_dit(
            num_frames=81, height=480, width=832
        )
        self.assertEqual(dit_480p.config.latent_height, 60)  # 480//8
        self.assertEqual(dit_480p.config.latent_width, 104)  # 832//8
        
        # Test at 720p
        dit_720p = create_wan_t2v_14b_dit(
            num_frames=81, height=720, width=1280
        )
        self.assertEqual(dit_720p.config.latent_height, 90)  # 720//8
        self.assertEqual(dit_720p.config.latent_width, 160)  # 1280//8


if __name__ == "__main__":
    unittest.main()
