"""Vision/Video modules for VAE and DiT models.

Includes:
- ShardedConv2d: 2D Convolution (no shardable dims currently)
- ShardedConv3d: 3D Convolution for video (no shardable dims currently)
- ShardedGroupNorm: Group Normalization
"""

from typing import Tuple
from llm_perf.modeling.base.module import ShardedModule
from llm_perf.modeling.base.tensor import ShardedTensor
from llm_perf.modeling.base.op import Conv2dOp, Conv3dOp, GroupNormOp


class ShardedConv2d(ShardedModule):
    """2D Convolution layer.

    Note: Currently no shardable dimensions for conv2d.
    Conv layers require special sharding strategies not implemented yet.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size (int or tuple)
        stride: Stride (default (1, 1))
        padding: Padding (default (0, 0))
        dtype: Data type
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dtype: str = "fp16",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kh, kw = kernel_size
        self.weight = ShardedTensor(
            shape=(out_channels, in_channels, kh, kw),
            shardable={},  # No shardable dims for conv currently
            dtype=dtype,
            name="conv2d_weight",
        )

    def forward(self, input_tensor: ShardedTensor) -> ShardedTensor:
        """2D Convolution forward.

        Args:
            input: (batch, channels, height, width)

        Returns:
            output: (batch, out_channels, height', width')
        """
        batch, in_ch, height, width = input_tensor.shape

        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        out_h = (height + 2 * ph - kh) // sh + 1
        out_w = (width + 2 * pw - kw) // sw + 1

        output = ShardedTensor(
            shape=(batch, self.out_channels, out_h, out_w),
            shardable={},  # No shardable dims
            dtype=input_tensor.dtype,
            name="conv2d_output",
        )

        output._op_history = input_tensor._op_history + [
            Conv2dOp(
                dtype=input_tensor.dtype,
                input=input_tensor,
                weight=self.weight,
                output=output,
                stride=self.stride,
                padding=self.padding,
            )
        ]

        return output


class ShardedConv3d(ShardedModule):
    """3D Convolution layer for video.

    Note: Currently no shardable dimensions for conv3d.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size (int or tuple)
        stride: Stride (default (1, 1, 1))
        padding: Padding (default (0, 0, 0))
        dtype: Data type
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
        dtype: str = "fp16",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kt, kh, kw = kernel_size
        self.weight = ShardedTensor(
            shape=(out_channels, in_channels, kt, kh, kw),
            shardable={},  # No shardable dims
            dtype=dtype,
            name="conv3d_weight",
        )

    def forward(self, input_tensor: ShardedTensor) -> ShardedTensor:
        """3D Convolution forward.

        Args:
            input: (batch, channels, time, height, width)

        Returns:
            output: (batch, out_channels, time', height', width')
        """
        batch, in_ch, time, height, width = input_tensor.shape

        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        pt, ph, pw = self.padding

        out_t = (time + 2 * pt - kt) // st + 1
        out_h = (height + 2 * ph - kh) // sh + 1
        out_w = (width + 2 * pw - kw) // sw + 1

        output = ShardedTensor(
            shape=(batch, self.out_channels, out_t, out_h, out_w),
            shardable={},  # No shardable dims
            dtype=input_tensor.dtype,
            name="conv3d_output",
        )

        output._op_history = input_tensor._op_history + [
            Conv3dOp(
                dtype=input_tensor.dtype,
                input=input_tensor,
                weight=self.weight,
                output=output,
                stride=self.stride,
                padding=self.padding,
            )
        ]

        return output


class ShardedGroupNorm(ShardedModule):
    """Group Normalization layer.

    Args:
        num_groups: Number of groups
        num_channels: Number of channels
        dtype: Data type
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels

        self.weight = ShardedTensor(
            shape=(num_channels,),
            shardable={},  # Norm weights not shardable
            dtype=dtype,
            name="groupnorm_weight",
        )
        self.bias = ShardedTensor(
            shape=(num_channels,),
            shardable={},
            dtype=dtype,
            name="groupnorm_bias",
        )

    def forward(self, input_tensor: ShardedTensor) -> ShardedTensor:
        """Group Normalization forward.

        Args:
            input: (batch, channels, ...) tensor

        Returns:
            output: Same shape as input
        """
        output = ShardedTensor(
            shape=input_tensor.shape,
            shardable={},  # No shardable dims
            dtype=input_tensor.dtype,
            name="groupnorm_output",
        )

        output._op_history = input_tensor._op_history + [
            GroupNormOp(
                dtype=input_tensor.dtype,
                input=input_tensor,
                weight=self.weight,
                output=output,
                num_groups=self.num_groups,
            )
        ]

        return output


def conv2d(
    input_tensor: ShardedTensor,
    weight: ShardedTensor,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
) -> ShardedTensor:
    """Functional 2D convolution.

    Args:
        input: Input tensor
        weight: Weight tensor
        stride: Stride
        padding: Padding

    Returns:
        output: Convolved tensor
    """
    batch, in_ch, height, width = input_tensor.shape
    out_ch, _, kh, kw = weight.shape

    sh, sw = stride
    ph, pw = padding

    out_h = (height + 2 * ph - kh) // sh + 1
    out_w = (width + 2 * pw - kw) // sw + 1

    output = ShardedTensor(
        shape=(batch, out_ch, out_h, out_w),
        shardable={},
        dtype=input_tensor.dtype,
        name="conv2d_output",
    )

    output._op_history = input_tensor._op_history + [
        Conv2dOp(
            dtype=input_tensor.dtype,
            input=input_tensor,
            weight=weight,
            output=output,
            stride=stride,
            padding=padding,
        )
    ]

    return output


def conv3d(
    input_tensor: ShardedTensor,
    weight: ShardedTensor,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
) -> ShardedTensor:
    """Functional 3D convolution.

    Args:
        input: Input tensor
        weight: Weight tensor
        stride: Stride
        padding: Padding

    Returns:
        output: Convolved tensor
    """
    batch, in_ch, time, height, width = input_tensor.shape
    out_ch, _, kt, kh, kw = weight.shape

    st, sh, sw = stride
    pt, ph, pw = padding

    out_t = (time + 2 * pt - kt) // st + 1
    out_h = (height + 2 * ph - kh) // sh + 1
    out_w = (width + 2 * pw - kw) // sw + 1

    output = ShardedTensor(
        shape=(batch, out_ch, out_t, out_h, out_w),
        shardable={},
        dtype=input_tensor.dtype,
        name="conv3d_output",
    )

    output._op_history = input_tensor._op_history + [
        Conv3dOp(
            dtype=input_tensor.dtype,
            input=input_tensor,
            weight=weight,
            output=output,
            stride=stride,
            padding=padding,
        )
    ]

    return output
