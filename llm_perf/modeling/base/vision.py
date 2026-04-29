"""Vision models using Sharded interface.

Includes:
- ShardedConv2d/ShardedConv3d: Basic convolution layers
- ShardedGroupNorm: Group normalization
- ShardedResNetBlock2d/ShardedResNetBlock3d: ResNet blocks for VAE
- ShardedAttentionBlock2d/ShardedAttentionBlock3d: Attention blocks for VAE
- ShardedResNet: Complete ResNet model for image classification
"""

from llm_perf.modeling.module import ShardedModule
from llm_perf.modeling.tensor import ShardedTensor, ShardedParameter
from llm_perf.kernels.op import Conv2dOp, Conv3dOp, GroupNormOp
from llm_perf.modeling.base.layers import silu


class ShardedConv2d(ShardedModule):
    """2D Convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), dtype="fp16"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        self.weight = ShardedParameter(
            shape=(out_channels, in_channels, *self.kernel_size),
            shardable={},
            dtype=dtype,
            name="weight",
        )

    def forward(self, x: ShardedTensor) -> ShardedTensor:
        N, C_in, H, W = x.shape
        C_out, _, kH, kW = self.weight.shape
        sh, sw = self.stride
        ph, pw = self.padding
        H_out = (H + 2 * ph - kH) // sh + 1
        W_out = (W + 2 * pw - kW) // sw + 1

        output = ShardedTensor(
            shape=(N, C_out, H_out, W_out),
            shardable={},
            dtype=self.weight.dtype,
            name="conv2d_output",
        )

        output._op_history = x._op_history + [
            Conv2dOp(
                dtype=x.dtype,
                input=x,
                weight=self.weight,
                output=output,
                stride=self.stride,
                padding=self.padding,
            )
        ]

        return output


class ShardedConv3d(ShardedModule):
    """3D Convolution layer.

    Sharding:
    - out_channels dimension TP-sharded (output channel parallelism)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dtype="fp16"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        self.weight = ShardedParameter(
            shape=(out_channels, in_channels, *self.kernel_size),
            shardable={0: "tp"},
            dtype=dtype,
            name="weight",
        )

    def forward(self, x: ShardedTensor) -> ShardedTensor:
        N, C_in, D, H, W = x.shape
        C_out, _, kD, kH, kW = self.weight.shape
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding
        D_out = (D + 2 * pd - kD) // sd + 1
        H_out = (H + 2 * ph - kH) // sh + 1
        W_out = (W + 2 * pw - kW) // sw + 1

        output = ShardedTensor(
            shape=(N, C_out, D_out, H_out, W_out),
            shardable={1: "tp"},
            dtype=self.weight.dtype,
            name="conv3d_output",
        )

        output._op_history = x._op_history + [
            Conv3dOp(
                dtype=x.dtype,
                input=x,
                weight=self.weight,
                output=output,
                stride=self.stride,
                padding=self.padding,
            )
        ]

        return output


class ShardedGroupNorm(ShardedModule):
    """Group Normalization layer."""

    def __init__(self, num_groups, num_channels, dtype="fp16"):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels

        self.weight = ShardedParameter(
            shape=(num_channels,),
            shardable={},
            dtype=dtype,
            name="weight",
        )
        self.bias = ShardedParameter(
            shape=(num_channels,),
            shardable={},
            dtype=dtype,
            name="bias",
        )

    def forward(self, x: ShardedTensor) -> ShardedTensor:
        output = ShardedTensor(
            shape=x.shape,
            shardable=x.shardable,
            dtype=self.weight.dtype,
            name="groupnorm_output",
        )

        output._op_history = x._op_history + [
            GroupNormOp(
                dtype=x.dtype,
                input=x,
                weight=self.weight,
                output=output,
                num_groups=self.num_groups,
            )
        ]

        return output


class ShardedResNetBlock2d(ShardedModule):
    """ResNet block for 2D VAE.

    Structure: norm -> conv -> norm -> conv (+ shortcut)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = ShardedGroupNorm(32, in_channels, dtype=dtype)
        self.conv1 = ShardedConv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), dtype=dtype)
        self.norm2 = ShardedGroupNorm(32, out_channels, dtype=dtype)
        self.conv2 = ShardedConv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), dtype=dtype)

        if in_channels != out_channels:
            self.shortcut = ShardedConv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), dtype=dtype)

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """ResNet block forward."""
        residual = hidden

        hidden = self.norm1(hidden)
        hidden = silu(hidden)
        hidden = self.conv1(hidden)

        hidden = self.norm2(hidden)
        hidden = silu(hidden)
        hidden = self.conv2(hidden)

        if self.in_channels != self.out_channels:
            residual = self.shortcut(residual)

        output = hidden + residual
        return output


class ShardedResNetBlock3d(ShardedModule):
    """ResNet block for 3D VAE.

    Structure: norm -> conv -> norm -> conv (+ shortcut)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = ShardedGroupNorm(32, in_channels, dtype=dtype)
        self.conv1 = ShardedConv3d(in_channels, out_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1), dtype=dtype)
        self.norm2 = ShardedGroupNorm(32, out_channels, dtype=dtype)
        self.conv2 = ShardedConv3d(out_channels, out_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1), dtype=dtype)

        if in_channels != out_channels:
            self.shortcut = ShardedConv3d(in_channels, out_channels, (1, 1, 1), (1, 1, 1), (0, 0, 0), dtype=dtype)

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """ResNet block forward."""
        residual = hidden

        hidden = self.norm1(hidden)
        hidden = silu(hidden)
        hidden = self.conv1(hidden)

        hidden = self.norm2(hidden)
        hidden = silu(hidden)
        hidden = self.conv2(hidden)

        if self.in_channels != self.out_channels:
            residual = self.shortcut(residual)

        output = hidden + residual
        return output


class ShardedAttentionBlock2d(ShardedModule):
    """Self-attention block for 2D VAE.

    Structure: norm -> qkv_proj -> attention -> out_proj
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.norm = ShardedGroupNorm(32, channels, dtype=dtype)

        qkv_channels = channels * 3
        self.qkv_proj = ShardedConv2d(channels, qkv_channels, (1, 1), (1, 1), (0, 0), dtype=dtype)

        self.out_proj = ShardedConv2d(channels, channels, (1, 1), (1, 1), (0, 0), dtype=dtype)

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """Attention block forward."""
        batch = hidden.shape[0] if len(hidden.shape) >= 1 else 1
        h = hidden.shape[2] if len(hidden.shape) >= 3 else 1
        w = hidden.shape[3] if len(hidden.shape) >= 4 else 1

        residual = hidden

        hidden = self.norm(hidden)

        qkv = self.qkv_proj(hidden)

        qkv_flat = ShardedTensor(
            shape=(batch, self.num_heads * 3, h * w, self.head_dim),
            shardable=qkv.shardable,
            dtype=hidden.dtype,
            name="qkv_flat",
        )

        from llm_perf.modeling.base.layers import flash_attention

        q = ShardedTensor(
            shape=(batch, self.num_heads, h * w, self.head_dim),
            shardable=qkv_flat.shardable,
            dtype=hidden.dtype,
            name="q",
        )
        k = ShardedTensor(
            shape=(batch, self.num_heads, h * w, self.head_dim),
            shardable=qkv_flat.shardable,
            dtype=hidden.dtype,
            name="k",
        )
        v = ShardedTensor(
            shape=(batch, self.num_heads, h * w, self.head_dim),
            shardable=qkv_flat.shardable,
            dtype=hidden.dtype,
            name="v",
        )

        attn_out = flash_attention(q, k, v, is_causal=False)

        attn_reshaped = ShardedTensor(
            shape=(batch, self.channels, h, w),
            shardable=attn_out.shardable,
            dtype=hidden.dtype,
            name="attn_reshaped",
        )

        output = self.out_proj(attn_reshaped)

        output = output + residual

        return output


class ShardedAttentionBlock3d(ShardedModule):
    """Self-attention block for 3D VAE.

    Structure: norm -> qkv_proj -> attention -> out_proj
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.norm = ShardedGroupNorm(32, channels, dtype=dtype)

        qkv_channels = channels * 3
        self.qkv_proj = ShardedConv3d(channels, qkv_channels, (1, 1, 1), (1, 1, 1), (0, 0, 0), dtype=dtype)

        self.out_proj = ShardedConv3d(channels, channels, (1, 1, 1), (1, 1, 1), (0, 0, 0), dtype=dtype)

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """Attention block forward."""
        batch = hidden.shape[0] if len(hidden.shape) >= 1 else 1
        t = hidden.shape[2] if len(hidden.shape) >= 3 else 1
        h = hidden.shape[3] if len(hidden.shape) >= 4 else 1
        w = hidden.shape[4] if len(hidden.shape) >= 5 else 1

        residual = hidden

        hidden = self.norm(hidden)

        qkv = self.qkv_proj(hidden)

        seq_len = h * w
        qkv_flat = ShardedTensor(
            shape=(batch, t, self.num_heads * 3, seq_len, self.head_dim),
            shardable=qkv.shardable,
            dtype=hidden.dtype,
            name="qkv_flat",
        )

        from llm_perf.modeling.base.layers import flash_attention

        q = ShardedTensor(
            shape=(batch, self.num_heads, t * seq_len, self.head_dim),
            shardable=qkv_flat.shardable,
            dtype=hidden.dtype,
            name="q",
        )
        k = ShardedTensor(
            shape=(batch, self.num_heads, t * seq_len, self.head_dim),
            shardable=qkv_flat.shardable,
            dtype=hidden.dtype,
            name="k",
        )
        v = ShardedTensor(
            shape=(batch, self.num_heads, t * seq_len, self.head_dim),
            shardable=qkv_flat.shardable,
            dtype=hidden.dtype,
            name="v",
        )

        attn_out = flash_attention(q, k, v, is_causal=False)

        attn_reshaped = ShardedTensor(
            shape=(batch, self.channels, t, h, w),
            shardable=attn_out.shardable,
            dtype=hidden.dtype,
            name="attn_reshaped",
        )

        output = self.out_proj(attn_reshaped)

        output = output + residual

        return output


class ShardedResNet(ShardedModule):
    """Complete ResNet model for image classification.

    Supports variants: resnet18, resnet34, resnet50, resnet101, resnet152

    Args:
        variant: ResNet variant name
        num_classes: Number of output classes
        input_channels: Input image channels
        input_height: Input image height
        input_width: Input image width
        dtype: Data type
    """

    ARCHITECTURES = {
        "resnet18": ([2, 2, 2, 2], "basic"),
        "resnet34": ([3, 4, 6, 3], "basic"),
        "resnet50": ([3, 4, 6, 3], "bottleneck"),
        "resnet101": ([3, 4, 23, 3], "bottleneck"),
        "resnet152": ([3, 8, 36, 3], "bottleneck"),
    }

    STAGE_CHANNELS = [64, 128, 256, 512]

    def __init__(
        self,
        variant: str = "resnet50",
        num_classes: int = 1000,
        input_channels: int = 3,
        input_height: int = 224,
        input_width: int = 224,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.variant = variant.lower()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        if self.variant not in self.ARCHITECTURES:
            raise ValueError(f"Unknown ResNet variant: {self.variant}")

        layers_per_stage, block_type = self.ARCHITECTURES[self.variant]

        self.conv1 = ShardedConv2d(
            input_channels,
            64,
            (7, 7),
            (2, 2),
            (3, 3),
            dtype=dtype,
        )

        current_h = input_height // 4
        current_w = input_width // 4

        self.stages = []
        in_channels = 64
        stage_idx = 0

        for stage_num, num_blocks in enumerate(layers_per_stage, start=2):
            out_channels = self.STAGE_CHANNELS[stage_idx]
            expand_factor = 4 if block_type == "bottleneck" else 1

            for block_idx in range(num_blocks):
                stride = 2 if block_idx == 0 and stage_num > 2 else 1
                block_in_ch = in_channels if block_idx == 0 else out_channels * expand_factor
                block_out_ch = out_channels

                if block_type == "basic":
                    block = self._build_basic_block(
                        stage_num, block_idx, block_in_ch, block_out_ch, current_h, current_w, stride, dtype
                    )
                else:
                    block = self._build_bottleneck_block(
                        stage_num, block_idx, block_in_ch, block_out_ch, current_h, current_w, stride, dtype
                    )

                self.stages.append(block)
                setattr(self, f"layer{stage_num}_{block_idx}", block)

                if stride == 2:
                    current_h //= 2
                    current_w //= 2

                in_channels = out_channels * expand_factor
            stage_idx += 1

        final_channels = in_channels

        self.fc_weight = ShardedParameter(
            shape=(final_channels, num_classes),
            shardable={0: "tp"},
            dtype=dtype,
            name="fc_weight",
        )

    def _build_basic_block(
        self,
        stage: int,
        block: int,
        in_channels: int,
        out_channels: int,
        h: int,
        w: int,
        stride: int,
        dtype: str,
    ) -> ShardedModule:
        """Build a BasicBlock (ResNet-18/34)."""

        class BasicBlock(ShardedModule):
            def __init__(self, in_ch, out_ch, stride, dtype):
                super().__init__()
                self.conv1 = ShardedConv2d(in_ch, out_ch, (3, 3), (stride, stride), (1, 1), dtype=dtype)
                self.conv2 = ShardedConv2d(out_ch, out_ch, (3, 3), (1, 1), (1, 1), dtype=dtype)
                if stride != 1 or in_ch != out_ch:
                    self.shortcut = ShardedConv2d(in_ch, out_ch, (1, 1), (stride, stride), (0, 0), dtype=dtype)
                else:
                    self.shortcut = None

            def forward(self, x):
                residual = x
                out = self.conv1(x)
                out = self.conv2(out)
                if self.shortcut:
                    residual = self.shortcut(residual)
                return out + residual

        return BasicBlock(in_channels, out_channels, stride, dtype)

    def _build_bottleneck_block(
        self,
        stage: int,
        block: int,
        in_channels: int,
        out_channels: int,
        h: int,
        w: int,
        stride: int,
        dtype: str,
    ) -> ShardedModule:
        """Build a Bottleneck block (ResNet-50/101/152)."""

        class Bottleneck(ShardedModule):
            def __init__(self, in_ch, out_ch, stride, dtype):
                super().__init__()
                expand_ch = out_ch * 4
                self.conv1 = ShardedConv2d(in_ch, out_ch, (1, 1), (1, 1), (0, 0), dtype=dtype)
                self.conv2 = ShardedConv2d(out_ch, out_ch, (3, 3), (stride, stride), (1, 1), dtype=dtype)
                self.conv3 = ShardedConv2d(out_ch, expand_ch, (1, 1), (1, 1), (0, 0), dtype=dtype)
                if stride != 1 or in_ch != expand_ch:
                    self.shortcut = ShardedConv2d(in_ch, expand_ch, (1, 1), (stride, stride), (0, 0), dtype=dtype)
                else:
                    self.shortcut = None

            def forward(self, x):
                residual = x
                out = self.conv1(x)
                out = self.conv2(out)
                out = self.conv3(out)
                if self.shortcut:
                    residual = self.shortcut(residual)
                return out + residual

        return Bottleneck(in_channels, out_channels, stride, dtype)

    def forward(self, image: ShardedTensor) -> ShardedTensor:
        """ResNet forward.

        Args:
            image: (batch, channels, height, width)

        Returns:
            logits: (batch, num_classes)
        """
        hidden = self.conv1(image)

        for stage in self.stages:
            hidden = stage(hidden)

        final_channels = self.fc_weight.shape[0]
        hidden_flat = ShardedTensor(
            shape=(hidden.shape[0], final_channels),
            shardable={},
            dtype=hidden.dtype,
            name="flattened",
        )

        logits = hidden_flat @ self.fc_weight

        return logits

    @classmethod
    def from_variant(cls, variant: str, num_classes: int = 1000, dtype: str = "fp16") -> "ShardedResNet":
        """Create ResNet from variant name."""
        return cls(
            variant=variant,
            num_classes=num_classes,
            input_channels=3,
            input_height=224,
            input_width=224,
            dtype=dtype,
        )