"""Video VAE (Variational Autoencoder) model for video generation.

Based on AutoencoderKL from Diffusers, adapted for video generation with 3D convolutions.
Supports both Encoder and Decoder with 3D ResNet blocks and attention layers.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

from .base import BaseModel, ModelConfig, LayerConfig
from ..utils.constants import DTYPE_SIZES


@dataclass
class VAEConfig(ModelConfig):
    """Video VAE configuration.

    For video VAE:
    - in_channels: Input video channels (typically 3 for RGB)
    - out_channels: Output video channels (typically 3 for RGB)
    - latent_channels: Latent space channels (typically 4 or 8)
    - hidden_size: Base channel width for encoder/decoder
    """

    # Video dimensions
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4

    # Video resolution
    num_frames: int = 16
    height: int = 256
    width: int = 256

    # Encoder/decoder architecture
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2

    # Whether to use 3D convolutions (True for video, False for image)
    use_3d_conv: bool = True

    # Attention configuration
    use_attention: bool = True
    attention_head_dim: int = 64

    # Scaling factor for latent space
    scaling_factor: float = 0.18215

    def __post_init__(self):
        """Set default values for ModelConfig fields."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = 0  # Not used in VAE


class VAEModel(BaseModel):
    """Video VAE model with separate Encoder and Decoder.

    Architecture based on AutoencoderKL from Diffusers:
    - Encoder: Compresses video to latent representation
    - Decoder: Reconstructs video from latent representation

    Uses 3D convolutions (temporal + spatial) for video processing.
    """

    def __init__(self, config: VAEConfig):
        super().__init__(config)
        self._layers = self.build_layers()

    def build_layers(self) -> List[LayerConfig]:
        """Build VAE layer configurations (Encoder + Decoder)."""
        layers = []
        cfg = self.config
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)

        # Build Encoder
        encoder_layers = self._build_encoder(dtype_size)
        layers.extend(encoder_layers)

        # Build Decoder
        decoder_layers = self._build_decoder(dtype_size)
        layers.extend(decoder_layers)

        return layers

    def _build_encoder(self, dtype_size: int) -> List[LayerConfig]:
        """Build Video VAE Encoder.

        Encoder structure:
        1. Initial conv3d (in_channels -> block_out_channels[0])
        2. Downsample blocks (each with ResNet blocks + optional attention)
        3. Mid block with attention
        4. Output conv (to latent_channels * 2 for mean and logvar)
        """
        layers = []
        cfg = self.config

        in_channels = cfg.in_channels
        out_channels_list = list(cfg.block_out_channels)

        # Initial convolution
        if cfg.use_3d_conv:
            layers.append(self._build_conv3d_layer(
                name="encoder_conv_in",
                in_channels=in_channels,
                out_channels=out_channels_list[0],
                kernel_size=(3, 3, 3),
                input_t=cfg.num_frames,
                input_h=cfg.height,
                input_w=cfg.width,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                dtype_size=dtype_size,
            ))

            current_t = cfg.num_frames
            current_h = cfg.height
            current_w = cfg.width
        else:
            # 2D conv for image VAE
            layers.append(self._build_conv2d_layer(
                name="encoder_conv_in",
                in_channels=in_channels,
                out_channels=out_channels_list[0],
                kernel_size=3,
                input_h=cfg.height,
                input_w=cfg.width,
                stride=1,
                padding=1,
                dtype_size=dtype_size,
            ))
            current_t = 1
            current_h = cfg.height
            current_w = cfg.width

        # Downsample blocks
        for i, out_channels in enumerate(out_channels_list):
            is_last_block = i == len(out_channels_list) - 1

            # ResNet blocks
            for j in range(cfg.layers_per_block):
                in_ch = out_channels_list[i - 1] if i > 0 and j == 0 else out_channels
                layers.extend(self._build_resnet_block(
                    name=f"encoder_down_{i}_resnet_{j}",
                    in_channels=in_ch,
                    out_channels=out_channels,
                    time_dim=current_t,
                    height=current_h,
                    width=current_w,
                    dtype_size=dtype_size,
                    use_3d=cfg.use_3d_conv,
                ))

            # Attention in later blocks
            if cfg.use_attention and i >= len(out_channels_list) - 2:
                layers.extend(self._build_attention_block(
                    name=f"encoder_down_{i}_attn",
                    channels=out_channels,
                    time_dim=current_t,
                    height=current_h,
                    width=current_w,
                    dtype_size=dtype_size,
                    use_3d=cfg.use_3d_conv,
                ))

            # Downsample (except last block)
            if not is_last_block:
                if cfg.use_3d_conv:
                    layers.append(self._build_conv3d_layer(
                        name=f"encoder_down_{i}_downsample",
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3, 3),
                        input_t=current_t,
                        input_h=current_h,
                        input_w=current_w,
                        stride=(1, 2, 2),  # Downsample spatial only
                        padding=(1, 1, 1),
                        dtype_size=dtype_size,
                    ))
                    current_h //= 2
                    current_w //= 2
                else:
                    layers.append(self._build_conv2d_layer(
                        name=f"encoder_down_{i}_downsample",
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        input_h=current_h,
                        input_w=current_w,
                        stride=2,
                        padding=1,
                        dtype_size=dtype_size,
                    ))
                    current_h //= 2
                    current_w //= 2

        # Mid block with attention
        mid_channels = out_channels_list[-1]
        layers.extend(self._build_resnet_block(
            name="encoder_mid_resnet_0",
            in_channels=mid_channels,
            out_channels=mid_channels,
            time_dim=current_t,
            height=current_h,
            width=current_w,
            dtype_size=dtype_size,
            use_3d=cfg.use_3d_conv,
        ))

        if cfg.use_attention:
            layers.extend(self._build_attention_block(
                name="encoder_mid_attn",
                channels=mid_channels,
                time_dim=current_t,
                height=current_h,
                width=current_w,
                dtype_size=dtype_size,
                use_3d=cfg.use_3d_conv,
            ))

        layers.extend(self._build_resnet_block(
            name="encoder_mid_resnet_1",
            in_channels=mid_channels,
            out_channels=mid_channels,
            time_dim=current_t,
            height=current_h,
            width=current_w,
            dtype_size=dtype_size,
            use_3d=cfg.use_3d_conv,
        ))

        # Output convolution (to latent, *2 for mean and logvar)
        if cfg.use_3d_conv:
            layers.append(self._build_conv3d_layer(
                name="encoder_conv_out",
                in_channels=mid_channels,
                out_channels=cfg.latent_channels * 2,
                kernel_size=(3, 3, 3),
                input_t=current_t,
                input_h=current_h,
                input_w=current_w,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                dtype_size=dtype_size,
            ))
        else:
            layers.append(self._build_conv2d_layer(
                name="encoder_conv_out",
                in_channels=mid_channels,
                out_channels=cfg.latent_channels * 2,
                kernel_size=3,
                input_h=current_h,
                input_w=current_w,
                stride=1,
                padding=1,
                dtype_size=dtype_size,
            ))

        return layers

    def _build_decoder(self, dtype_size: int) -> List[LayerConfig]:
        """Build Video VAE Decoder.

        Decoder structure:
        1. Initial conv (from latent_channels)
        2. Mid block with attention
        3. Upsample blocks (each with ResNet blocks + optional attention)
        4. Output conv (to out_channels)
        """
        layers = []
        cfg = self.config

        # Calculate latent dimensions after encoding
        latent_h = cfg.height // (2 ** (len(cfg.block_out_channels) - 1))
        latent_w = cfg.width // (2 ** (len(cfg.block_out_channels) - 1))
        latent_t = cfg.num_frames if cfg.use_3d_conv else 1

        block_out_channels = list(cfg.block_out_channels)
        reverse_channels = list(reversed(block_out_channels))

        # Initial convolution
        if cfg.use_3d_conv:
            layers.append(self._build_conv3d_layer(
                name="decoder_conv_in",
                in_channels=cfg.latent_channels,
                out_channels=reverse_channels[0],
                kernel_size=(3, 3, 3),
                input_t=latent_t,
                input_h=latent_h,
                input_w=latent_w,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                dtype_size=dtype_size,
            ))
            current_t = latent_t
            current_h = latent_h
            current_w = latent_w
        else:
            layers.append(self._build_conv2d_layer(
                name="decoder_conv_in",
                in_channels=cfg.latent_channels,
                out_channels=reverse_channels[0],
                kernel_size=3,
                input_h=latent_h,
                input_w=latent_w,
                stride=1,
                padding=1,
                dtype_size=dtype_size,
            ))
            current_t = 1
            current_h = latent_h
            current_w = latent_w

        # Mid block
        mid_channels = reverse_channels[0]
        layers.extend(self._build_resnet_block(
            name="decoder_mid_resnet_0",
            in_channels=mid_channels,
            out_channels=mid_channels,
            time_dim=current_t,
            height=current_h,
            width=current_w,
            dtype_size=dtype_size,
            use_3d=cfg.use_3d_conv,
        ))

        if cfg.use_attention:
            layers.extend(self._build_attention_block(
                name="decoder_mid_attn",
                channels=mid_channels,
                time_dim=current_t,
                height=current_h,
                width=current_w,
                dtype_size=dtype_size,
                use_3d=cfg.use_3d_conv,
            ))

        layers.extend(self._build_resnet_block(
            name="decoder_mid_resnet_1",
            in_channels=mid_channels,
            out_channels=mid_channels,
            time_dim=current_t,
            height=current_h,
            width=current_w,
            dtype_size=dtype_size,
            use_3d=cfg.use_3d_conv,
        ))

        # Upsample blocks
        for i, out_channels in enumerate(reverse_channels):
            is_last_block = i == len(reverse_channels) - 1

            # ResNet blocks
            for j in range(cfg.layers_per_block + 1):  # Decoder has one more block
                in_ch = reverse_channels[i - 1] if i > 0 and j == 0 else out_channels
                layers.extend(self._build_resnet_block(
                    name=f"decoder_up_{i}_resnet_{j}",
                    in_channels=in_ch,
                    out_channels=out_channels,
                    time_dim=current_t,
                    height=current_h,
                    width=current_w,
                    dtype_size=dtype_size,
                    use_3d=cfg.use_3d_conv,
                ))

            # Attention in early upsample blocks
            if cfg.use_attention and i < 2:
                layers.extend(self._build_attention_block(
                    name=f"decoder_up_{i}_attn",
                    channels=out_channels,
                    time_dim=current_t,
                    height=current_h,
                    width=current_w,
                    dtype_size=dtype_size,
                    use_3d=cfg.use_3d_conv,
                ))

            # Upsample (except last block)
            if not is_last_block:
                if cfg.use_3d_conv:
                    layers.append(self._build_conv3d_layer(
                        name=f"decoder_up_{i}_upsample",
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3, 3),
                        input_t=current_t,
                        input_h=current_h,
                        input_w=current_w,
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                        dtype_size=dtype_size,
                    ))
                    current_h *= 2
                    current_w *= 2
                else:
                    layers.append(self._build_conv2d_layer(
                        name=f"decoder_up_{i}_upsample",
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        input_h=current_h,
                        input_w=current_w,
                        stride=1,
                        padding=1,
                        dtype_size=dtype_size,
                    ))
                    current_h *= 2
                    current_w *= 2

        # Output normalization and convolution
        # GroupNorm (simplified as activation)
        layers.append(LayerConfig(
            name="decoder_norm_out",
            input_shape=(1, out_channels, current_t, current_h, current_w) if cfg.use_3d_conv
                      else (1, out_channels, current_h, current_w),
            output_shape=(1, out_channels, current_t, current_h, current_w) if cfg.use_3d_conv
                       else (1, out_channels, current_h, current_w),
            params_count=out_channels * 2,  # gamma + beta
            flops=out_channels * current_t * current_h * current_w * 7,
            activation_bytes=out_channels * current_t * current_h * current_w * dtype_size,
        ))

        # Final conv to output channels
        if cfg.use_3d_conv:
            layers.append(self._build_conv3d_layer(
                name="decoder_conv_out",
                in_channels=out_channels,
                out_channels=cfg.out_channels,
                kernel_size=(3, 3, 3),
                input_t=current_t,
                input_h=current_h,
                input_w=current_w,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                dtype_size=dtype_size,
            ))
        else:
            layers.append(self._build_conv2d_layer(
                name="decoder_conv_out",
                in_channels=out_channels,
                out_channels=cfg.out_channels,
                kernel_size=3,
                input_h=current_h,
                input_w=current_w,
                stride=1,
                padding=1,
                dtype_size=dtype_size,
            ))

        return layers

    def _build_conv2d_layer(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        input_h: int,
        input_w: int,
        stride: int,
        padding: int,
        dtype_size: int,
    ) -> LayerConfig:
        """Build a 2D convolution layer config."""
        output_h = (input_h + 2 * padding - kernel_size) // stride + 1
        output_w = (input_w + 2 * padding - kernel_size) // stride + 1

        flops = (
            2 * output_h * output_w *
            out_channels * kernel_size * kernel_size * in_channels
        )

        params = out_channels * in_channels * kernel_size * kernel_size

        return LayerConfig(
            name=name,
            input_shape=(1, in_channels, input_h, input_w),
            output_shape=(1, out_channels, output_h, output_w),
            params_count=params,
            flops=flops,
            activation_bytes=out_channels * output_h * output_w * dtype_size,
        )

    def _build_conv3d_layer(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        input_t: int,
        input_h: int,
        input_w: int,
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        dtype_size: int,
    ) -> LayerConfig:
        """Build a 3D convolution layer config."""
        kt, kh, kw = kernel_size
        st, sh, sw = stride
        pt, ph, pw = padding

        output_t = (input_t + 2 * pt - kt) // st + 1
        output_h = (input_h + 2 * ph - kh) // sh + 1
        output_w = (input_w + 2 * pw - kw) // sw + 1

        flops = (
            2 * output_t * output_h * output_w *
            out_channels * kt * kh * kw * in_channels
        )

        params = out_channels * in_channels * kt * kh * kw

        return LayerConfig(
            name=name,
            input_shape=(1, in_channels, input_t, input_h, input_w),
            output_shape=(1, out_channels, output_t, output_h, output_w),
            params_count=params,
            flops=flops,
            activation_bytes=out_channels * output_t * output_h * output_w * dtype_size,
        )

    def _build_resnet_block(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        height: int,
        width: int,
        dtype_size: int,
        use_3d: bool = True,
    ) -> List[LayerConfig]:
        """Build a ResNet block (norm1 -> conv1 -> norm2 -> conv2)."""
        layers = []

        # GroupNorm 1
        layers.append(LayerConfig(
            name=f"{name}_norm1",
            input_shape=(1, in_channels, time_dim, height, width) if use_3d
                      else (1, in_channels, height, width),
            output_shape=(1, in_channels, time_dim, height, width) if use_3d
                       else (1, in_channels, height, width),
            params_count=in_channels * 2,
            flops=in_channels * time_dim * height * width * 7,
            activation_bytes=in_channels * time_dim * height * width * dtype_size,
        ))

        # Conv1
        if use_3d:
            layers.append(self._build_conv3d_layer(
                name=f"{name}_conv1",
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3, 3),
                input_t=time_dim,
                input_h=height,
                input_w=width,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                dtype_size=dtype_size,
            ))
        else:
            layers.append(self._build_conv2d_layer(
                name=f"{name}_conv1",
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                input_h=height,
                input_w=width,
                stride=1,
                padding=1,
                dtype_size=dtype_size,
            ))

        # GroupNorm 2
        layers.append(LayerConfig(
            name=f"{name}_norm2",
            input_shape=(1, out_channels, time_dim, height, width) if use_3d
                      else (1, out_channels, height, width),
            output_shape=(1, out_channels, time_dim, height, width) if use_3d
                       else (1, out_channels, height, width),
            params_count=out_channels * 2,
            flops=out_channels * time_dim * height * width * 7,
            activation_bytes=out_channels * time_dim * height * width * dtype_size,
        ))

        # Conv2
        if use_3d:
            layers.append(self._build_conv3d_layer(
                name=f"{name}_conv2",
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3, 3),
                input_t=time_dim,
                input_h=height,
                input_w=width,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                dtype_size=dtype_size,
            ))
        else:
            layers.append(self._build_conv2d_layer(
                name=f"{name}_conv2",
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                input_h=height,
                input_w=width,
                stride=1,
                padding=1,
                dtype_size=dtype_size,
            ))

        # Shortcut if channels change
        if in_channels != out_channels:
            if use_3d:
                layers.append(self._build_conv3d_layer(
                    name=f"{name}_shortcut",
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1, 1),
                    input_t=time_dim,
                    input_h=height,
                    input_w=width,
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                    dtype_size=dtype_size,
                ))
            else:
                layers.append(self._build_conv2d_layer(
                    name=f"{name}_shortcut",
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    input_h=height,
                    input_w=width,
                    stride=1,
                    padding=0,
                    dtype_size=dtype_size,
                ))

        return layers

    def _build_attention_block(
        self,
        name: str,
        channels: int,
        time_dim: int,
        height: int,
        width: int,
        dtype_size: int,
        use_3d: bool = True,
    ) -> List[LayerConfig]:
        """Build a self-attention block for VAE.

        Uses spatial attention over (height, width) dimensions.
        For 3D, can optionally use temporal attention as well.
        """
        layers = []

        # GroupNorm
        layers.append(LayerConfig(
            name=f"{name}_norm",
            input_shape=(1, channels, time_dim, height, width) if use_3d
                      else (1, channels, height, width),
            output_shape=(1, channels, time_dim, height, width) if use_3d
                       else (1, channels, height, width),
            params_count=channels * 2,
            flops=channels * time_dim * height * width * 7,
            activation_bytes=channels * time_dim * height * width * dtype_size,
        ))

        # QKV projection (1x1 conv or 1x1x1 conv)
        head_dim = 64
        num_heads = max(1, channels // head_dim)
        qkv_channels = channels * 3

        if use_3d:
            layers.append(self._build_conv3d_layer(
                name=f"{name}_qkv",
                in_channels=channels,
                out_channels=qkv_channels,
                kernel_size=(1, 1, 1),
                input_t=time_dim,
                input_h=height,
                input_w=width,
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                dtype_size=dtype_size,
            ))
        else:
            layers.append(self._build_conv2d_layer(
                name=f"{name}_qkv",
                in_channels=channels,
                out_channels=qkv_channels,
                kernel_size=1,
                input_h=height,
                input_w=width,
                stride=1,
                padding=0,
                dtype_size=dtype_size,
            ))

        # Attention computation (Q @ K^T @ V)
        # Simplified: attention over spatial dimensions
        seq_len = height * width
        if use_3d:
            # Spatial attention per frame
            attn_flops = (
                2 * time_dim * num_heads * seq_len * seq_len * head_dim +  # QK^T
                2 * time_dim * num_heads * seq_len * seq_len * head_dim    # @V
            )
        else:
            attn_flops = (
                2 * num_heads * seq_len * seq_len * head_dim +
                2 * num_heads * seq_len * seq_len * head_dim
            )

        layers.append(LayerConfig(
            name=f"{name}_attn_compute",
            input_shape=(1, qkv_channels, time_dim, height, width) if use_3d
                      else (1, qkv_channels, height, width),
            output_shape=(1, channels, time_dim, height, width) if use_3d
                       else (1, channels, height, width),
            params_count=0,
            flops=attn_flops,
            activation_bytes=channels * time_dim * height * width * dtype_size,
        ))

        # Output projection
        if use_3d:
            layers.append(self._build_conv3d_layer(
                name=f"{name}_proj",
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 1, 1),
                input_t=time_dim,
                input_h=height,
                input_w=width,
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                dtype_size=dtype_size,
            ))
        else:
            layers.append(self._build_conv2d_layer(
                name=f"{name}_proj",
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                input_h=height,
                input_w=width,
                stride=1,
                padding=0,
                dtype_size=dtype_size,
            ))

        return layers

    @classmethod
    def from_config(
        cls,
        model_type: str = "video_vae",  # "video_vae" or "image_vae"
        num_frames: int = 16,
        height: int = 256,
        width: int = 256,
        dtype: str = "fp16",
    ) -> "VAEModel":
        """Create a VAE model from a predefined configuration.

        Args:
            model_type: "video_vae" for video generation, "image_vae" for image
            num_frames: Number of video frames (ignored for image_vae)
            height: Input height
            width: Input width
            dtype: Data type

        Returns:
            VAEModel instance
        """
        if model_type == "video_vae":
            config = VAEConfig(
                name="video_vae",
                vocab_size=0,  # Not used
                hidden_size=512,
                num_layers=0,  # Computed dynamically
                num_attention_heads=0,  # Not used
                in_channels=3,
                out_channels=3,
                latent_channels=4,
                num_frames=num_frames,
                height=height,
                width=width,
                block_out_channels=(128, 256, 512, 512),
                layers_per_block=2,
                use_3d_conv=True,
                use_attention=True,
                dtype=dtype,
            )
        else:  # image_vae
            config = VAEConfig(
                name="image_vae",
                vocab_size=0,
                hidden_size=512,
                num_layers=0,
                num_attention_heads=0,
                in_channels=3,
                out_channels=3,
                latent_channels=4,
                num_frames=1,
                height=height,
                width=width,
                block_out_channels=(128, 256, 512, 512),
                layers_per_block=2,
                use_3d_conv=False,
                use_attention=True,
                dtype=dtype,
            )
        return cls(config)
