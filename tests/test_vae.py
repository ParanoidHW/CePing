"""Tests for Video VAE model implementation."""

import unittest
from llm_perf.models.vae import VAEConfig, VAEModel
from llm_perf.kernels.compute import ComputeKernelRegistry
from llm_perf.hardware.device import Device


class TestVAEConfig(unittest.TestCase):
    """Test VAEConfig dataclass."""

    def test_basic_creation(self):
        """Test creating a VAEConfig."""
        config = VAEConfig(
            name="video_vae",
            vocab_size=0,
            hidden_size=512,
            num_layers=0,
            num_attention_heads=0,
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            num_frames=16,
            height=256,
            width=256,
        )
        self.assertEqual(config.in_channels, 3)
        self.assertEqual(config.latent_channels, 4)
        self.assertEqual(config.num_frames, 16)
        self.assertEqual(config.height, 256)
        self.assertEqual(config.width, 256)
        self.assertTrue(config.use_3d_conv)

    def test_video_vs_image_vae(self):
        """Test video vs image VAE configuration."""
        video_config = VAEConfig(
            name="video_vae",
            vocab_size=0,
            hidden_size=512,
            num_layers=0,
            num_attention_heads=0,
            use_3d_conv=True,
            num_frames=16,
        )
        image_config = VAEConfig(
            name="image_vae",
            vocab_size=0,
            hidden_size=512,
            num_layers=0,
            num_attention_heads=0,
            use_3d_conv=False,
            num_frames=1,
        )
        self.assertTrue(video_config.use_3d_conv)
        self.assertFalse(image_config.use_3d_conv)


class TestVAEModel(unittest.TestCase):
    """Test VAEModel layer building."""

    def test_video_vae_creation(self):
        """Test Video VAE model creation."""
        model = VAEModel.from_config(
            model_type="video_vae",
            num_frames=8,
            height=128,
            width=128,
        )
        self.assertGreater(model.total_params, 0)
        self.assertEqual(model.config.name, "video_vae")
        self.assertTrue(model.config.use_3d_conv)

    def test_image_vae_creation(self):
        """Test Image VAE model creation."""
        model = VAEModel.from_config(
            model_type="image_vae",
            height=256,
            width=256,
        )
        self.assertGreater(model.total_params, 0)
        self.assertEqual(model.config.name, "image_vae")
        self.assertFalse(model.config.use_3d_conv)

    def test_encoder_layers_exist(self):
        """Test that encoder layers exist."""
        model = VAEModel.from_config("video_vae", num_frames=8, height=64, width=64)
        names = [layer.name for layer in model.layers]
        self.assertIn("encoder_conv_in", names)
        self.assertIn("encoder_conv_out", names)
        # Check for downsample blocks
        self.assertIn("encoder_down_0_resnet_0_conv1", names)
        self.assertIn("encoder_mid_resnet_0_conv1", names)

    def test_decoder_layers_exist(self):
        """Test that decoder layers exist."""
        model = VAEModel.from_config("video_vae", num_frames=8, height=64, width=64)
        names = [layer.name for layer in model.layers]
        self.assertIn("decoder_conv_in", names)
        self.assertIn("decoder_conv_out", names)
        # Check for upsample blocks
        self.assertIn("decoder_up_0_resnet_0_conv1", names)
        self.assertIn("decoder_mid_resnet_0_conv1", names)

    def test_attention_layers_exist(self):
        """Test that attention layers exist when enabled."""
        model = VAEModel.from_config("video_vae", num_frames=8, height=64, width=64)
        names = [layer.name for layer in model.layers]
        # Attention in encoder down blocks (later blocks)
        self.assertIn("encoder_down_2_attn_qkv", names)
        self.assertIn("encoder_mid_attn_qkv", names)
        # Attention in decoder
        self.assertIn("decoder_up_0_attn_qkv", names)
        self.assertIn("decoder_mid_attn_qkv", names)

    def test_3d_conv_shapes(self):
        """Test 3D convolution layer shapes."""
        model = VAEModel.from_config("video_vae", num_frames=16, height=256, width=256)
        conv_in = model.get_layer_by_name("encoder_conv_in")
        self.assertIsNotNone(conv_in)
        # Input: (B, 3, 16, 256, 256)
        self.assertEqual(conv_in.input_shape, (1, 3, 16, 256, 256))
        # Output: (B, 128, 16, 256, 256) with stride 1
        self.assertEqual(conv_in.output_shape[0], 1)
        self.assertEqual(conv_in.output_shape[1], 128)
        self.assertEqual(conv_in.output_shape[2], 16)

    def test_2d_conv_shapes(self):
        """Test 2D convolution layer shapes for image VAE."""
        model = VAEModel.from_config("image_vae", height=256, width=256)
        conv_in = model.get_layer_by_name("encoder_conv_in")
        self.assertIsNotNone(conv_in)
        # Input: (B, 3, 256, 256)
        self.assertEqual(conv_in.input_shape, (1, 3, 256, 256))
        # Output: (B, 128, 256, 256)
        self.assertEqual(conv_in.output_shape[0], 1)
        self.assertEqual(conv_in.output_shape[1], 128)

    def test_flops_calculation(self):
        """Test FLOPs calculation."""
        model = VAEModel.from_config("video_vae", num_frames=8, height=64, width=64)
        self.assertGreater(model.total_flops_forward, 0)
        self.assertEqual(model.total_flops_backward, model.total_flops_forward * 2)

    def test_to_dict(self):
        """Test serialization."""
        model = VAEModel.from_config("video_vae", num_frames=8, height=64, width=64)
        data = model.to_dict()
        self.assertEqual(data["name"], "video_vae")
        self.assertIn("config", data)
        self.assertIn("total_params", data)
        self.assertIn("layers", data)
        self.assertGreater(len(data["layers"]), 0)

    def test_activation_memory(self):
        """Test activation memory calculation."""
        model = VAEModel.from_config("video_vae", num_frames=8, height=64, width=64)
        self.assertGreater(model.activation_memory, 0)

    def test_resnet_block_structure(self):
        """Test ResNet block has correct structure."""
        model = VAEModel.from_config("video_vae", num_frames=8, height=64, width=64)
        names = [layer.name for layer in model.layers]
        # ResNet block components
        self.assertIn("encoder_down_0_resnet_0_norm1", names)
        self.assertIn("encoder_down_0_resnet_0_conv1", names)
        self.assertIn("encoder_down_0_resnet_0_norm2", names)
        self.assertIn("encoder_down_0_resnet_0_conv2", names)


class TestConv3dKernelWithVAE(unittest.TestCase):
    """Test Conv3d kernels with VAE models."""

    def setUp(self):
        self.device = Device.from_preset("H100-SXM-80GB")
        self.registry = ComputeKernelRegistry(self.device)

    def test_conv3d_kernel_registration(self):
        """Test that conv3d kernels are registered."""
        kernels = self.registry.list_kernels()
        conv3d_kernels = [k for k in kernels if k.startswith("conv3d")]
        self.assertGreater(len(conv3d_kernels), 0)

    def test_get_conv3d_kernel(self):
        """Test getting a specific conv3d kernel."""
        kernel = self.registry.get_or_create_conv3d(
            batch=1,
            in_channels=128,
            out_channels=256,
            kernel_size_t=3,
            kernel_size_h=3,
            kernel_size_w=3,
            input_t=16,
            input_h=64,
            input_w=64,
            stride_t=1,
            stride_h=2,
            stride_w=2,
            padding_t=1,
            padding_h=1,
            padding_w=1,
            dtype="fp16",
        )
        self.assertIsNotNone(kernel)
        self.assertIn("conv3d", kernel.name)
        self.assertGreater(kernel.flops, 0)
        self.assertGreater(kernel.bytes_accessed, 0)

    def test_conv3d_flops_calculation(self):
        """Test FLOPs calculation for conv3d kernel."""
        kernel = self.registry.get_or_create_conv3d(
            batch=1,
            in_channels=64,
            out_channels=128,
            kernel_size_t=3,
            kernel_size_h=3,
            kernel_size_w=3,
            input_t=16,
            input_h=32,
            input_w=32,
            stride_t=1,
            stride_h=1,
            stride_w=1,
            padding_t=1,
            padding_h=1,
            padding_w=1,
            dtype="fp16",
        )
        # FLOPs = 2 * batch * out_t * out_h * out_w * out_c * kt * kh * kw * in_c
        # out_t = 16, out_h = 32, out_w = 32 (same padding)
        expected_flops = 2 * 1 * 16 * 32 * 32 * 128 * 3 * 3 * 3 * 64
        self.assertEqual(kernel.flops, expected_flops)

    def test_conv3d_memory_calculation(self):
        """Test memory calculation for conv3d kernel."""
        kernel = self.registry.get_or_create_conv3d(
            batch=1,
            in_channels=3,
            out_channels=128,
            kernel_size_t=3,
            kernel_size_h=3,
            kernel_size_w=3,
            input_t=16,
            input_h=256,
            input_w=256,
            stride_t=1,
            stride_h=1,
            stride_w=1,
            padding_t=1,
            padding_h=1,
            padding_w=1,
            dtype="fp16",
        )
        # Check that memory includes input, weights, output, workspace
        # bytes_accessed should be significantly larger than just input+output
        dtype_size = 2
        input_bytes = 1 * 3 * 16 * 256 * 256 * dtype_size
        weight_bytes = 128 * 3 * 3 * 3 * 3 * dtype_size
        output_bytes = 1 * 128 * 16 * 256 * 256 * dtype_size

        # Total should be larger than sum of these due to workspace
        expected_min = input_bytes + weight_bytes + output_bytes
        self.assertGreater(kernel.bytes_accessed, expected_min)

    def test_conv3d_temporal_stride(self):
        """Test conv3d with temporal stride."""
        kernel = self.registry.get_or_create_conv3d(
            batch=1,
            in_channels=512,
            out_channels=512,
            kernel_size_t=3,
            kernel_size_h=3,
            kernel_size_w=3,
            input_t=16,
            input_h=32,
            input_w=32,
            stride_t=2,  # Temporal downsample
            stride_h=1,
            stride_w=1,
            padding_t=1,
            padding_h=1,
            padding_w=1,
            dtype="fp16",
        )
        # out_t = (16 + 2 - 3) // 2 + 1 = 8
        self.assertIn("st2", kernel.name)

    def test_conv3d_estimate_time(self):
        """Test that conv3d kernel can estimate execution time."""
        kernel = self.registry.get_or_create_conv3d(
            batch=1,
            in_channels=128,
            out_channels=256,
            kernel_size_t=3,
            kernel_size_h=3,
            kernel_size_w=3,
            input_t=8,
            input_h=32,
            input_w=32,
            stride_t=1,
            stride_h=1,
            stride_w=1,
            padding_t=1,
            padding_h=1,
            padding_w=1,
            dtype="fp16",
        )
        time = kernel.estimate_time(
            input_shape=(1, 128, 8, 32, 32),
            output_shape=(1, 256, 8, 32, 32),
            dtype="fp16",
        )
        self.assertGreater(time, 0)


class TestVAEMemoryAccess(unittest.TestCase):
    """Test VAE memory access patterns."""

    def test_encoder_memory_reduction(self):
        """Test that encoder reduces spatial dimensions."""
        model = VAEModel.from_config("video_vae", num_frames=16, height=256, width=256)

        # Input conv
        conv_in = model.get_layer_by_name("encoder_conv_in")
        input_size = conv_in.input_shape[2] * conv_in.input_shape[3] * conv_in.input_shape[4]

        # Output conv (latent)
        conv_out = model.get_layer_by_name("encoder_conv_out")
        # Latent should be smaller
        self.assertLess(conv_out.output_shape[3], 256)
        self.assertLess(conv_out.output_shape[4], 256)

    def test_decoder_memory_expansion(self):
        """Test that decoder expands spatial dimensions."""
        model = VAEModel.from_config("video_vae", num_frames=16, height=256, width=256)

        # Decoder input (latent)
        conv_in = model.get_layer_by_name("decoder_conv_in")
        # Decoder output
        conv_out = model.get_layer_by_name("decoder_conv_out")

        # Output should match input resolution
        self.assertEqual(conv_out.output_shape[3], 256)
        self.assertEqual(conv_out.output_shape[4], 256)


if __name__ == "__main__":
    unittest.main()
