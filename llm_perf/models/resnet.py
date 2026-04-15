"""ResNet model implementation for computer vision workloads using kernel API."""

from dataclasses import dataclass
from typing import List

from .base import BaseModel, ModelConfig, LayerConfig
from ..utils.constants import DTYPE_SIZES
from ..kernels import conv2d
from ..kernels.utils import kernel_result_to_layer


@dataclass
class ResNetConfig(ModelConfig):
    """ResNet-specific configuration.

    For ResNet models, we use a simplified config where:
    - hidden_size = base channel width (e.g., 64 for ResNet-18/34/50)
    - num_layers = total number of layers including conv and fc
    - vocab_size = number of classes (typically 1000 for ImageNet)
    """

    # ResNet architecture variant
    variant: str = "resnet50"  # resnet18, resnet34, resnet50, resnet101, resnet152

    # Input image dimensions
    input_channels: int = 3
    input_height: int = 224
    input_width: int = 224

    def __post_init__(self):
        """Set model dimensions based on variant."""
        if self.variant in ["resnet18", "resnet34"]:
            # BasicBlock architecture
            self._block_type = "basic"
        else:
            # Bottleneck architecture
            self._block_type = "bottleneck"


class ResNetModel(BaseModel):
    """ResNet model implementation using kernel API.

    Supports ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152.
    """

    # ResNet architecture definitions
    # (layers_per_stage, block_type)
    ARCHITECTURES = {
        "resnet18": ([2, 2, 2, 2], "basic"),
        "resnet34": ([3, 4, 6, 3], "basic"),
        "resnet50": ([3, 4, 6, 3], "bottleneck"),
        "resnet101": ([3, 4, 23, 3], "bottleneck"),
        "resnet152": ([3, 8, 36, 3], "bottleneck"),
    }

    # Stage channel configurations
    STAGE_CHANNELS = [64, 128, 256, 512]

    def __init__(self, config: ResNetConfig):
        super().__init__(config)
        self._layers = self.build_layers()

    def build_layers(self) -> List[LayerConfig]:
        """Build ResNet layer configurations using kernel API."""
        layers = []
        cfg = self.config
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)

        variant = cfg.variant.lower()
        if variant not in self.ARCHITECTURES:
            raise ValueError(f"Unknown ResNet variant: {variant}")

        layers_per_stage, block_type = self.ARCHITECTURES[variant]

        # Initial convolution (conv1): 7x7, 64, stride 2, padding 3
        conv1_result = conv2d(
            input=(1, cfg.input_channels, cfg.input_height, cfg.input_width),
            weight=(64, cfg.input_channels, 7, 7),
            bias=(64,),  # Initial conv has bias
            stride=(2, 2),
            padding=(3, 3),
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="conv1",
            result=conv1_result))

        # MaxPool (not counted as a learnable layer but has compute)
        # 3x3, stride 2, padding 1 -> Output: 56x56 for 224x224 input
        current_h = 56
        current_w = 56

        # Residual stages (conv2_x to conv5_x)
        in_channels = 64
        stage_idx = 0

        for stage_num, num_blocks in enumerate(layers_per_stage, start=2):
            out_channels = self.STAGE_CHANNELS[stage_idx]

            for block_idx in range(num_blocks):
                stride = 2 if block_idx == 0 and stage_num > 2 else 1

                if block_type == "basic":
                    block_layers = self._build_basic_block(
                        stage=stage_num,
                        block=block_idx,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        input_h=current_h,
                        input_w=current_w,
                        stride=stride,
                        dtype_size=dtype_size,
                        dtype=cfg.dtype,
                    )
                else:  # bottleneck
                    block_layers = self._build_bottleneck_block(
                        stage=stage_num,
                        block=block_idx,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        input_h=current_h,
                        input_w=current_w,
                        stride=stride,
                        dtype_size=dtype_size,
                        dtype=cfg.dtype,
                    )

                layers.extend(block_layers)

                # Update dimensions for next block
                if stride == 2:
                    current_h //= 2
                    current_w //= 2
                in_channels = out_channels * (4 if block_type == "bottleneck" else 1)

            stage_idx += 1

        # Global average pooling
        # NOTE: Manual calculation for pooling layer (no kernel API available)
        layers.append(LayerConfig(
            name="avgpool",
            input_shape=(1, in_channels, current_h, current_w),
            output_shape=(1, in_channels, 1, 1),
            params_count=0,
            flops=in_channels * current_h * current_w * 2,  # sum + divide
            activation_bytes=in_channels * dtype_size,
        ))

        # Fully connected layer using conv2d kernel (treated as 1x1 conv)
        fc_result = conv2d(
            input=(1, in_channels, 1, 1),
            weight=(cfg.vocab_size, in_channels, 1, 1),
            bias=None,
            stride=(1, 1),
            padding=(0, 0),
            dtype=cfg.dtype
        )
        # FC layer: flatten input and output
        fc_layer = kernel_result_to_layer(
            name="fc",
            result=fc_result)
        # Override shapes for FC layer (flattened)
        fc_layer.input_shape = (1, in_channels)
        fc_layer.output_shape = (1, cfg.vocab_size)
        layers.append(fc_layer)

        return layers

    def _build_basic_block(
        self,
        stage: int,
        block: int,
        in_channels: int,
        out_channels: int,
        input_h: int,
        input_w: int,
        stride: int,
        dtype_size: int,
        dtype: str = "fp16",
    ) -> List[LayerConfig]:
        """Build a BasicBlock (for ResNet-18/34) using kernel API.

        BasicBlock consists of two 3x3 conv layers with a skip connection.
        """
        layers = []
        prefix = f"layer{stage}_{block}"

        # First conv 3x3
        conv1_result = conv2d(
            input=(1, in_channels, input_h, input_w),
            weight=(out_channels, in_channels, 3, 3),
            bias=None,
            stride=(stride, stride),
            padding=(1, 1),
            dtype=dtype
        )
        output_h = (input_h + 2 * 1 - 3) // stride + 1
        output_w = (input_w + 2 * 1 - 3) // stride + 1
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_conv1",
            result=conv1_result))

        # Second conv 3x3
        conv2_result = conv2d(
            input=(1, out_channels, output_h, output_w),
            weight=(out_channels, out_channels, 3, 3),
            bias=None,
            stride=(1, 1),
            padding=(1, 1),
            dtype=dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_conv2",
            result=conv2_result))

        # Shortcut connection (if needed)
        if stride != 1 or in_channels != out_channels:
            shortcut_result = conv2d(
                input=(1, in_channels, input_h, input_w),
                weight=(out_channels, in_channels, 1, 1),
                bias=None,
                stride=(stride, stride),
                padding=(0, 0),
                dtype=dtype
            )
            layers.append(kernel_result_to_layer(
                name=f"{prefix}_shortcut",
                result=shortcut_result))

        return layers

    def _build_bottleneck_block(
        self,
        stage: int,
        block: int,
        in_channels: int,
        out_channels: int,
        input_h: int,
        input_w: int,
        stride: int,
        dtype_size: int,
        dtype: str = "fp16",
    ) -> List[LayerConfig]:
        """Build a Bottleneck block (for ResNet-50/101/152) using kernel API.

        Bottleneck consists of 1x1, 3x3, 1x1 conv layers with a skip connection.
        """
        layers = []
        prefix = f"layer{stage}_{block}"
        expand_channels = out_channels * 4

        # First conv 1x1 (reduce)
        conv1_result = conv2d(
            input=(1, in_channels, input_h, input_w),
            weight=(out_channels, in_channels, 1, 1),
            bias=None,
            stride=(1, 1),
            padding=(0, 0),
            dtype=dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_conv1",
            result=conv1_result))

        # Second conv 3x3 (process)
        conv_h = input_h if stride == 1 else (input_h + 1) // 2
        conv_w = input_w if stride == 1 else (input_w + 1) // 2
        conv2_result = conv2d(
            input=(1, out_channels, conv_h, conv_w),
            weight=(out_channels, out_channels, 3, 3),
            bias=None,
            stride=(stride, stride),
            padding=(1, 1),
            dtype=dtype
        )
        output_h = (conv_h + 2 * 1 - 3) // stride + 1 if stride > 1 else conv_h
        output_w = (conv_w + 2 * 1 - 3) // stride + 1 if stride > 1 else conv_w
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_conv2",
            result=conv2_result))

        # Third conv 1x1 (expand)
        conv3_result = conv2d(
            input=(1, out_channels, output_h, output_w),
            weight=(expand_channels, out_channels, 1, 1),
            bias=None,
            stride=(1, 1),
            padding=(0, 0),
            dtype=dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_conv3",
            result=conv3_result))

        # Shortcut connection (if needed)
        if stride != 1 or in_channels != expand_channels:
            shortcut_result = conv2d(
                input=(1, in_channels, input_h, input_w),
                weight=(expand_channels, in_channels, 1, 1),
                bias=None,
                stride=(stride, stride),
                padding=(0, 0),
                dtype=dtype
            )
            layers.append(kernel_result_to_layer(
                name=f"{prefix}_shortcut",
                result=shortcut_result))

        return layers

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int = 1000,
        dtype: str = "fp16",
    ) -> "ResNetModel":
        """Create a ResNet model from a variant name.

        Args:
            variant: One of "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
            num_classes: Number of output classes
            dtype: Data type for computations

        Returns:
            ResNetModel instance
        """
        config = ResNetConfig(
            name=variant,
            vocab_size=num_classes,  # num_classes for image classification
            hidden_size=64,  # base channel width
            num_layers=0,  # computed dynamically
            num_attention_heads=0,  # not used in CNNs
            variant=variant,
            input_channels=3,
            input_height=224,
            input_width=224,
            dtype=dtype,
        )
        return cls(config)
