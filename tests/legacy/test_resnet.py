"""Tests for ResNet model implementation."""

import unittest
from llm_perf.models.resnet import ResNetConfig, ResNetModel
from llm_perf.kernels.compute import ComputeKernelRegistry
from llm_perf.hardware.device import Device


class TestResNetConfig(unittest.TestCase):
    """Test ResNetConfig dataclass."""

    def test_basic_creation(self):
        """Test creating a ResNetConfig."""
        config = ResNetConfig(
            name="resnet50",
            vocab_size=1000,
            hidden_size=64,
            num_layers=0,
            num_attention_heads=0,
            variant="resnet50",
        )
        self.assertEqual(config.variant, "resnet50")
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.input_channels, 3)
        self.assertEqual(config.input_height, 224)
        self.assertEqual(config.input_width, 224)

    def test_variant_post_init(self):
        """Test that variant is set correctly in post_init."""
        config = ResNetConfig(
            name="test",
            vocab_size=1000,
            hidden_size=64,
            num_layers=0,
            num_attention_heads=0,
            variant="resnet18",
        )
        # BasicBlock type for resnet18
        self.assertEqual(config._block_type, "basic")

        config2 = ResNetConfig(
            name="test",
            vocab_size=1000,
            hidden_size=64,
            num_layers=0,
            num_attention_heads=0,
            variant="resnet50",
        )
        # Bottleneck type for resnet50
        self.assertEqual(config2._block_type, "bottleneck")


class TestResNetModel(unittest.TestCase):
    """Test ResNetModel layer building."""

    def test_resnet18_creation(self):
        """Test ResNet-18 model creation."""
        model = ResNetModel.from_variant("resnet18", num_classes=1000)
        self.assertGreater(model.total_params, 0)
        self.assertEqual(model.config.variant, "resnet18")

    def test_resnet34_creation(self):
        """Test ResNet-34 model creation."""
        model = ResNetModel.from_variant("resnet34", num_classes=1000)
        self.assertGreater(model.total_params, 0)
        self.assertEqual(model.config.variant, "resnet34")

    def test_resnet50_creation(self):
        """Test ResNet-50 model creation."""
        model = ResNetModel.from_variant("resnet50", num_classes=1000)
        self.assertGreater(model.total_params, 0)
        self.assertEqual(model.config.variant, "resnet50")

    def test_resnet101_creation(self):
        """Test ResNet-101 model creation."""
        model = ResNetModel.from_variant("resnet101", num_classes=1000)
        self.assertGreater(model.total_params, 0)
        self.assertEqual(model.config.variant, "resnet101")

    def test_resnet152_creation(self):
        """Test ResNet-152 model creation."""
        model = ResNetModel.from_variant("resnet152", num_classes=1000)
        self.assertGreater(model.total_params, 0)
        self.assertEqual(model.config.variant, "resnet152")

    def test_invalid_variant(self):
        """Test that invalid variant raises error."""
        config = ResNetConfig(
            name="invalid",
            vocab_size=1000,
            hidden_size=64,
            num_layers=0,
            num_attention_heads=0,
            variant="invalid_variant",
        )
        with self.assertRaises(ValueError):
            ResNetModel(config)

    def test_layer_names_resnet18(self):
        """Test that expected layer names exist in ResNet-18."""
        model = ResNetModel.from_variant("resnet18")
        names = [layer.name for layer in model.layers]
        self.assertIn("conv1", names)
        self.assertIn("avgpool", names)
        self.assertIn("fc", names)
        # Check for BasicBlock layers
        self.assertIn("layer2_0_conv1", names)
        self.assertIn("layer2_0_conv2", names)

    def test_layer_names_resnet50(self):
        """Test that expected layer names exist in ResNet-50."""
        model = ResNetModel.from_variant("resnet50")
        names = [layer.name for layer in model.layers]
        self.assertIn("conv1", names)
        self.assertIn("avgpool", names)
        self.assertIn("fc", names)
        # Check for Bottleneck layers
        self.assertIn("layer2_0_conv1", names)
        self.assertIn("layer2_0_conv2", names)
        self.assertIn("layer2_0_conv3", names)

    def test_total_params_increases_with_depth(self):
        """Test that parameter count increases with model depth."""
        model18 = ResNetModel.from_variant("resnet18")
        model34 = ResNetModel.from_variant("resnet34")
        model50 = ResNetModel.from_variant("resnet50")
        model101 = ResNetModel.from_variant("resnet101")
        model152 = ResNetModel.from_variant("resnet152")

        self.assertLess(model18.total_params, model34.total_params)
        self.assertLess(model34.total_params, model50.total_params)
        self.assertLess(model50.total_params, model101.total_params)
        self.assertLess(model101.total_params, model152.total_params)

    def test_conv1_layer_shape(self):
        """Test that initial conv layer has correct shape."""
        model = ResNetModel.from_variant("resnet50")
        conv1 = model.get_layer_by_name("conv1")
        self.assertIsNotNone(conv1)
        # Input: 3x224x224, Output: 64x112x112 (stride 2, padding 3)
        self.assertEqual(conv1.input_shape, (1, 3, 224, 224))
        self.assertEqual(conv1.output_shape, (1, 64, 112, 112))

    def test_fc_layer_shape(self):
        """Test that final fc layer has correct shape."""
        model = ResNetModel.from_variant("resnet50", num_classes=1000)
        fc = model.get_layer_by_name("fc")
        self.assertIsNotNone(fc)
        self.assertEqual(fc.input_shape, (1, 2048))
        self.assertEqual(fc.output_shape, (1, 1000))

    def test_to_dict(self):
        """Test that to_dict produces expected structure."""
        model = ResNetModel.from_variant("resnet50")
        data = model.to_dict()
        self.assertEqual(data["name"], "resnet50")
        self.assertIn("config", data)
        self.assertIn("total_params", data)
        self.assertIn("layers", data)
        self.assertGreater(len(data["layers"]), 0)

    def test_flops_calculation(self):
        """Test that FLOPs are calculated correctly."""
        model = ResNetModel.from_variant("resnet18")
        self.assertGreater(model.total_flops_forward, 0)
        self.assertEqual(model.total_flops_backward, model.total_flops_forward * 2)

    def test_activation_memory(self):
        """Test that activation memory is calculated."""
        model = ResNetModel.from_variant("resnet50")
        self.assertGreater(model.activation_memory, 0)


class TestConvKernelWithResNet(unittest.TestCase):
    """Test Conv2d kernels with ResNet models."""

    def setUp(self):
        self.device = Device.from_preset("H100-SXM-80GB")
        self.registry = ComputeKernelRegistry(self.device)

    def test_conv_kernel_registration(self):
        """Test that conv kernels are registered."""
        kernels = self.registry.list_kernels()
        conv_kernels = [k for k in kernels if k.startswith("conv2d")]
        self.assertGreater(len(conv_kernels), 0)

    def test_get_conv_kernel(self):
        """Test getting a specific conv kernel."""
        kernel = self.registry.get_or_create_conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            batch=1,
            input_size=(56, 56),
            dtype="fp16",
        )
        self.assertIsNotNone(kernel)
        self.assertEqual(kernel.name, "conv2d_b1_ic64_oc64_k3_h56_w56_s1_g1_fp16")
        self.assertGreater(kernel.flops, 0)
        self.assertGreater(kernel.bytes_accessed, 0)

    def test_conv_kernel_flops_calculation(self):
        """Test FLOPs calculation for conv kernel."""
        kernel = self.registry.get_or_create_conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            batch=1,
            input_size=(56, 56),
            dtype="fp16",
        )
        # FLOPs = 2 * batch * out_h * out_w * out_c * k^2 * in_c
        # out_h = out_w = 56 (same padding)
        expected_flops = 2 * 1 * 56 * 56 * 64 * 3 * 3 * 64
        self.assertEqual(kernel.flops, expected_flops)

    def test_conv_kernel_memory_calculation(self):
        """Test memory calculation for conv kernel."""
        kernel = self.registry.get_or_create_conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            batch=1,
            input_size=(224, 224),
            dtype="fp16",
        )
        # out_h = out_w = (224 + 6 - 7) // 2 + 1 = 112
        # input_bytes = 1 * 3 * 224 * 224 * 2
        # weight_bytes = 64 * 3 * 7 * 7 * 2
        # output_bytes = 1 * 64 * 112 * 112 * 2
        dtype_size = 2
        expected_input = 1 * 3 * 224 * 224 * dtype_size
        expected_weight = 64 * 3 * 7 * 7 * dtype_size
        expected_output = 1 * 64 * 112 * 112 * dtype_size
        # Memory includes input, weights, output, and workspace
        workspace_bytes = 1 * 112 * 112 * 7 * 7 * 3 * dtype_size
        expected_bytes = expected_input + expected_weight + expected_output + int(workspace_bytes * 0.5)
        self.assertEqual(kernel.bytes_accessed, expected_bytes)

    def test_conv_kernel_estimate_time(self):
        """Test that conv kernel can estimate execution time."""
        kernel = self.registry.get_or_create_conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            batch=1,
            input_size=(56, 56),
            dtype="fp16",
        )
        time = kernel.estimate_time(
            input_shape=(1, 64, 56, 56),
            output_shape=(1, 64, 56, 56),
            dtype="fp16",
        )
        self.assertGreater(time, 0)


class TestResNetVariantsStructure(unittest.TestCase):
    """Test structure differences between ResNet variants."""

    def test_resnet18_basic_block(self):
        """Test ResNet-18 uses BasicBlock structure."""
        model = ResNetModel.from_variant("resnet18")
        names = [layer.name for layer in model.layers]
        # BasicBlock has conv1, conv2 per block
        self.assertIn("layer2_0_conv1", names)
        self.assertIn("layer2_0_conv2", names)
        # No conv3 in BasicBlock
        self.assertNotIn("layer2_0_conv3", names)

    def test_resnet50_bottleneck_block(self):
        """Test ResNet-50 uses Bottleneck structure."""
        model = ResNetModel.from_variant("resnet50")
        names = [layer.name for layer in model.layers]
        # Bottleneck has conv1, conv2, conv3 per block
        self.assertIn("layer2_0_conv1", names)
        self.assertIn("layer2_0_conv2", names)
        self.assertIn("layer2_0_conv3", names)

    def test_downsample_layers(self):
        """Test that downsample (stride=2) layers exist."""
        model = ResNetModel.from_variant("resnet50")
        names = [layer.name for layer in model.layers]
        # Layer 3 and 4 have stride=2 at first block
        self.assertIn("layer3_0_conv2", names)
        self.assertIn("layer4_0_conv2", names)


if __name__ == "__main__":
    unittest.main()
