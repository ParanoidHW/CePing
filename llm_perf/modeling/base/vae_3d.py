"""3D Video VAE (Variational Autoencoder) for video compression.

This module provides reusable 3D VAE components for video generation models.
Includes:
- ShardedVideoVAEEncoder: 3D Video VAE encoder (temporal 4x, spatial 16x compression)
- ShardedVideoVAEDecoder: 3D Video VAE decoder (reconstruction)
- ShardedVideoVAE: Complete VAE model

Reference: HunyuanVideo Technical Report
https://arxiv.org/abs/2412.03603
"""

from typing import Tuple

from llm_perf.kernels.op import Conv3dOp
from llm_perf.modeling.module import ShardedModule
from llm_perf.modeling.tensor import ShardedParameter, ShardedTensor


class ShardedVideoVAEEncoder(ShardedModule):
    """3D Video VAE Encoder.

    Compresses video to latent space with:
    - Temporal compression: 4x (stride=(4,1,1) at stage 0)
    - Spatial compression: 16x (4 stages with stride=2 each)

    Args:
        in_channels: Input video channels (3 for RGB)
        latent_channels: Latent space channels
        base_channels: Base channel count
        channel_multipliers: Channel multipliers per stage
        num_res_blocks: Number of residual blocks per stage
        dtype: Data type
    """

    _submodule_name = "video_vae_encoder"

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        dtype: str = "bf16",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.dtype = dtype

        self.conv_in_weight = ShardedParameter(
            shape=(base_channels, in_channels, 3, 3, 3),
            shardable={0: "tp"},
            dtype=dtype,
            name="conv_in_weight",
        )

        self.down_blocks = {}
        ch = base_channels
        for i, ch_mult in enumerate(channel_multipliers):
            ch_out = base_channels * ch_mult

            for j in range(num_res_blocks):
                self.down_blocks[f"stage{i}_res{j}"] = ShardedParameter(
                    shape=(ch_out, ch, 3, 3, 3),
                    shardable={0: "tp"},
                    dtype=dtype,
                    name=f"stage{i}_res{j}_weight",
                )
                ch = ch_out

            if i < len(channel_multipliers) - 1:
                self.down_blocks[f"stage{i}_down"] = ShardedParameter(
                    shape=(ch_out, ch_out, 3, 3, 3),
                    shardable={0: "tp"},
                    dtype=dtype,
                    name=f"stage{i}_down_weight",
                )

        final_channels = base_channels * channel_multipliers[-1]
        self.conv_out_weight = ShardedParameter(
            shape=(latent_channels, final_channels, 3, 3, 3),
            shardable={0: "tp"},
            dtype=dtype,
            name="conv_out_weight",
        )

    def forward(self, video: ShardedTensor) -> ShardedTensor:
        """Encode video to latent space.

        Args:
            video: Video tensor (batch, 3, frames, height, width)

        Returns:
            latent: Latent tensor (batch, 16, frames//4, height//16, width//16)
        """
        x = self._conv3d(
            video,
            self.conv_in_weight,
            out_channels=self.base_channels,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            name="conv_in",
        )

        for i, ch_mult in enumerate(self.channel_multipliers):
            ch_out = self.base_channels * ch_mult

            for j in range(self.num_res_blocks):
                weight_key = f"stage{i}_res{j}"
                x = self._conv3d(
                    x,
                    self.down_blocks[weight_key],
                    out_channels=ch_out,
                    stride=(1, 1, 1),
                    padding=(1, 1, 1),
                    name=f"stage{i}_res{j}",
                )

            if i < len(self.channel_multipliers) - 1:
                stride = (4, 2, 2) if i == 0 else (1, 2, 2)
                padding = (0, 1, 1) if i == 0 else (1, 1, 1)
                down_key = f"stage{i}_down"

                x = self._conv3d(
                    x,
                    self.down_blocks[down_key],
                    out_channels=ch_out,
                    stride=stride,
                    padding=padding,
                    name=f"stage{i}_down",
                )

        latent = self._conv3d(
            x,
            self.conv_out_weight,
            out_channels=self.latent_channels,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            name="conv_out",
        )

        return latent

    def _conv3d(
        self,
        input_tensor: ShardedTensor,
        weight: ShardedParameter,
        out_channels: int,
        stride: Tuple[int, int, int],
        name: str,
        padding: Tuple[int, int, int] = (0, 0, 0),
    ) -> ShardedTensor:
        """Helper function for 3D convolution.

        Args:
            input_tensor: Input tensor
            weight: Convolution weight
            out_channels: Output channels
            stride: Convolution stride
            name: Output tensor name
            padding: Convolution padding

        Returns:
            Output tensor
        """
        N = input_tensor.shape[0] if len(input_tensor.shape) >= 1 else 1
        D = input_tensor.shape[2] if len(input_tensor.shape) >= 3 else 1
        H = input_tensor.shape[3] if len(input_tensor.shape) >= 4 else 1
        W = input_tensor.shape[4] if len(input_tensor.shape) >= 5 else 1

        kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]
        sd, sh, sw = stride
        pd, ph, pw = padding

        D_out = (D + 2 * pd - kD) // sd + 1
        H_out = (H + 2 * ph - kH) // sh + 1
        W_out = (W + 2 * pw - kW) // sw + 1

        output = ShardedTensor(
            shape=(N, out_channels, D_out, H_out, W_out),
            shardable={1: "tp"},
            dtype=self.dtype,
            name=name,
        )

        output._op_history = input_tensor._op_history + [
            Conv3dOp(
                dtype=self.dtype,
                input=input_tensor,
                weight=weight,
                output=output,
                stride=stride,
                padding=padding,
            )
        ]

        self._activations[name] = output
        return output


class ShardedVideoVAEDecoder(ShardedModule):
    """3D Video VAE Decoder.

    Reconstructs video from latent space.

    Args:
        latent_channels: Latent space channels
        out_channels: Output video channels (3 for RGB)
        base_channels: Base channel count
        channel_multipliers: Channel multipliers per stage
        num_res_blocks: Number of residual blocks per stage
        dtype: Data type
    """

    _submodule_name = "video_vae_decoder"

    def __init__(
        self,
        latent_channels: int = 16,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 3,
        dtype: str = "bf16",
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.dtype = dtype

        final_channels = base_channels * channel_multipliers[-1]
        self.conv_in_weight = ShardedParameter(
            shape=(final_channels, latent_channels, 3, 3, 3),
            shardable={0: "tp"},
            dtype=dtype,
            name="conv_in_weight",
        )

        self.up_blocks = {}
        reversed_multipliers = list(reversed(channel_multipliers[:-1]))

        for i, ch_mult in enumerate(reversed_multipliers):
            ch_in = (
                base_channels * channel_multipliers[-1]
                if i == 0
                else base_channels * reversed_multipliers[i - 1]
            )
            ch_out = base_channels * ch_mult

            for j in range(num_res_blocks):
                self.up_blocks[f"stage{i}_res{j}"] = ShardedParameter(
                    shape=(ch_out, ch_in, 3, 3, 3),
                    shardable={0: "tp"},
                    dtype=dtype,
                    name=f"stage{i}_res{j}_weight",
                )
                ch_in = ch_out

            if i < len(reversed_multipliers) - 1:
                self.up_blocks[f"stage{i}_up"] = ShardedParameter(
                    shape=(ch_out, ch_out, 4, 4, 4),
                    shardable={0: "tp"},
                    dtype=dtype,
                    name=f"stage{i}_up_weight",
                )

        self.conv_out_weight = ShardedParameter(
            shape=(out_channels, base_channels, 3, 3, 3),
            shardable={0: "tp"},
            dtype=dtype,
            name="conv_out_weight",
        )

    def forward(self, latent: ShardedTensor) -> ShardedTensor:
        """Decode latent to video.

        Args:
            latent: Latent tensor (batch, 16, frames//4, height//16, width//16)

        Returns:
            video: Video tensor (batch, 3, frames, height, width)
        """
        final_channels = self.base_channels * self.channel_multipliers[-1]
        x = self._conv3d(
            latent,
            self.conv_in_weight,
            out_channels=final_channels,
            stride=(1, 1, 1),
            name="conv_in",
        )

        reversed_multipliers = list(reversed(self.channel_multipliers[:-1]))

        for i, ch_mult in enumerate(reversed_multipliers):
            ch_out = self.base_channels * ch_mult

            for j in range(self.num_res_blocks):
                weight_key = f"stage{i}_res{j}"
                x = self._conv3d(
                    x,
                    self.up_blocks[weight_key],
                    out_channels=ch_out,
                    stride=(1, 1, 1),
                    name=f"stage{i}_res{j}",
                )

            if i < len(reversed_multipliers) - 1:
                x = self._upsample_and_conv(
                    x,
                    stage_idx=i,
                    ch_out=ch_out,
                    name=f"stage{i}_up",
                )

        video = self._conv3d(
            x,
            self.conv_out_weight,
            out_channels=self.out_channels,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            name="conv_out",
        )

        return video

    def _conv3d(
        self,
        input_tensor: ShardedTensor,
        weight: ShardedParameter,
        out_channels: int,
        stride: Tuple[int, int, int],
        name: str,
        padding: Tuple[int, int, int] = (0, 0, 0),
    ) -> ShardedTensor:
        """Helper function for 3D convolution.

        Args:
            input_tensor: Input tensor
            weight: Convolution weight
            out_channels: Output channels
            stride: Convolution stride
            name: Output tensor name
            padding: Convolution padding

        Returns:
            Output tensor
        """
        N = input_tensor.shape[0] if len(input_tensor.shape) >= 1 else 1
        D = input_tensor.shape[2] if len(input_tensor.shape) >= 3 else 1
        H = input_tensor.shape[3] if len(input_tensor.shape) >= 4 else 1
        W = input_tensor.shape[4] if len(input_tensor.shape) >= 5 else 1

        kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]
        sd, sh, sw = stride
        pd, ph, pw = padding

        D_out = (D + 2 * pd - kD) // sd + 1
        H_out = (H + 2 * ph - kH) // sh + 1
        W_out = (W + 2 * pw - kW) // sw + 1

        output = ShardedTensor(
            shape=(N, out_channels, D_out, H_out, W_out),
            shardable={1: "tp"},
            dtype=self.dtype,
            name=name,
        )

        output._op_history = input_tensor._op_history + [
            Conv3dOp(
                dtype=self.dtype,
                input=input_tensor,
                weight=weight,
                output=output,
                stride=stride,
                padding=padding,
            )
        ]

        self._activations[name] = output
        return output

    def _upsample_and_conv(
        self,
        input_tensor: ShardedTensor,
        stage_idx: int,
        ch_out: int,
        name: str,
    ) -> ShardedTensor:
        """Upsample and convolve (transposed convolution simulation).

        For simplicity, we use stride=(1,1,1) with upsampled tensor shape.

        Args:
            input_tensor: Input tensor
            stage_idx: Stage index
            ch_out: Output channels
            name: Output tensor name

        Returns:
            Upsampled output tensor
        """
        N = input_tensor.shape[0] if len(input_tensor.shape) >= 1 else 1
        D = input_tensor.shape[2] if len(input_tensor.shape) >= 3 else 1
        H = input_tensor.shape[3] if len(input_tensor.shape) >= 4 else 1
        W = input_tensor.shape[4] if len(input_tensor.shape) >= 5 else 1

        sd = 4 if stage_idx == len(self.channel_multipliers) - 3 else 1
        sh, sw = 2, 2

        D_out = D * sd
        H_out = H * sh
        W_out = W * sw

        upsampled = ShardedTensor(
            shape=(N, ch_out, D_out, H_out, W_out),
            shardable={1: "tp"},
            dtype=self.dtype,
            name=f"{name}_upsampled",
        )
        upsampled._op_history = input_tensor._op_history
        upsampled._is_view = True

        weight = self.up_blocks[name]
        kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]

        final_D = D_out - kD + 1
        final_H = H_out - kH + 1
        final_W = W_out - kW + 1

        output = ShardedTensor(
            shape=(N, ch_out, final_D, final_H, final_W),
            shardable={1: "tp"},
            dtype=self.dtype,
            name=f"{name}_output",
        )

        output._op_history = upsampled._op_history + [
            Conv3dOp(
                dtype=self.dtype,
                input=upsampled,
                weight=weight,
                output=output,
                stride=(1, 1, 1),
                padding=(0, 0, 0),
            )
        ]

        self._activations[f"{name}_upsampled"] = upsampled
        self._activations[f"{name}_output"] = output
        return output


class ShardedVideoVAE(ShardedModule):
    """Complete Video VAE model.

    Combines encoder and decoder for video compression/reconstruction.

    Args:
        encoder_config: Encoder configuration dict
        decoder_config: Decoder configuration dict
    """

    _submodule_name = "video_vae"

    def __init__(
        self,
        encoder_config: dict = None,
        decoder_config: dict = None,
    ):
        super().__init__()

        encoder_config = encoder_config or {}
        decoder_config = decoder_config or {}

        self.encoder = ShardedVideoVAEEncoder(**encoder_config)
        self.decoder = ShardedVideoVAEDecoder(**decoder_config)

    def encode(self, video: ShardedTensor) -> ShardedTensor:
        """Encode video to latent.

        Args:
            video: Video tensor (batch, 3, frames, height, width)

        Returns:
            latent: Latent tensor (batch, 16, frames//4, height//16, width//16)
        """
        return self.encoder(video)

    def decode(self, latent: ShardedTensor) -> ShardedTensor:
        """Decode latent to video.

        Args:
            latent: Latent tensor (batch, 16, frames//4, height//16, width//16)

        Returns:
            video: Video tensor (batch, 3, frames, height, width)
        """
        return self.decoder(latent)

    def forward(self, video: ShardedTensor) -> ShardedTensor:
        """Full forward pass: encode + decode.

        Args:
            video: Video tensor (batch, 3, frames, height, width)

        Returns:
            reconstructed: Reconstructed video tensor
        """
        latent = self.encode(video)
        reconstructed = self.decode(latent)
        return reconstructed