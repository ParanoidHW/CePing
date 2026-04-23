"""Vision encoders and decoders for multimodal models.

Includes:
- ShardedViTEncoder: Vision Transformer encoder (independent module)
- ShardedVAEEncoder: VAE encoder for image/video latent representation
- ShardedVAEDecoder: VAE decoder for reconstructing image/video from latent
- ShardedVAE: Complete VAE model with encoder and decoder

VAE components:
- Encoder: Down-sampling (8x spatial compression)
- Decoder: Up-sampling (8x spatial restoration)

Reference:
- Qwen2-VL Vision Encoder: depth=27, hidden_size=1152, patch_size=16
- Wan VAE: 3D causal VAE for video generation
"""

from typing import Tuple
from llm_perf.modeling.module import ShardedModule
from llm_perf.modeling.tensor import ShardedTensor, ShardedParameter
from llm_perf.modeling.layers import (
    ShardedAttention,
    ShardedFFN,
    silu,
)
from llm_perf.kernels.op import RMSNormOp, Conv2dOp
from llm_perf.utils.constants import FFNActType
from llm_perf.modeling.vision import (
    ShardedConv2d,
    ShardedConv3d,
    ShardedGroupNorm,
    ShardedResNetBlock2d,
    ShardedResNetBlock3d,
    ShardedAttentionBlock2d,
    ShardedAttentionBlock3d,
)


class ShardedViTBlock(ShardedModule):
    """Vision Transformer Block.

    Structure:
    - LayerNorm -> Self-Attention -> Residual
    - LayerNorm -> MLP -> Residual

    Args:
        hidden_size: Hidden size (1152 for Qwen2-VL)
        num_heads: Number of attention heads (16)
        intermediate_size: MLP intermediate size (4304)
        dtype: Data type
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size

        head_dim = hidden_size // num_heads

        self.norm1_weight = ShardedParameter(
            shape=(hidden_size,),
            shardable={},
            dtype=dtype,
            name="norm1_weight",
        )

        self.attention = ShardedAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

        self.norm2_weight = ShardedParameter(
            shape=(hidden_size,),
            shardable={},
            dtype=dtype,
            name="norm2_weight",
        )

        self.mlp = ShardedFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ffn_act_type=FFNActType.GELU.value,
            dtype=dtype,
        )

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """ViT block forward with pre-norm architecture.

        Args:
            hidden: (batch, seq, hidden_size)

        Returns:
            output: (batch, seq, hidden_size)
        """
        norm_hidden = self._rms_norm(hidden, self.norm1_weight)
        attn_out = self.attention(norm_hidden, is_causal=False)
        hidden = hidden + attn_out

        norm_hidden = self._rms_norm(hidden, self.norm2_weight)
        mlp_out = self.mlp(norm_hidden)
        output = hidden + mlp_out

        self._activations["attn_out"] = attn_out
        self._activations["mlp_out"] = mlp_out

        return output

    def _rms_norm(self, input_tensor: ShardedTensor, weight: ShardedParameter) -> ShardedTensor:
        """RMS normalization."""
        output = ShardedTensor(
            shape=input_tensor.shape,
            shardable=input_tensor.shardable,
            dtype=input_tensor.dtype,
            name="rmsnorm_output",
        )
        output._op_history = input_tensor._op_history + [
            RMSNormOp(
                dtype=input_tensor.dtype,
                input=input_tensor,
                weight=weight,
                output=output,
            )
        ]
        self._track_intermediate("rmsnorm_output", output)
        return output


class ShardedPatchEmbedding(ShardedModule):
    """Patch Embedding using Conv2d.

    Converts image patches to embeddings.
    For Qwen2-VL: patch_size=16, hidden_size=1152

    Args:
        in_channels: Input channels (3 for RGB)
        hidden_size: Output embedding dimension
        patch_size: Patch size (16)
        dtype: Data type
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 1152,
        patch_size: int = 16,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size

        self.weight = ShardedParameter(
            shape=(hidden_size, in_channels, patch_size, patch_size),
            shardable={},
            dtype=dtype,
            name="patch_embed_weight",
        )

    def forward(self, image: ShardedTensor) -> ShardedTensor:
        """Patch embedding forward.

        Args:
            image: (batch, channels, height, width)

        Returns:
            patches: (batch, hidden_size, h_patches, w_patches)
        """
        batch = image.shape[0] if len(image.shape) >= 1 else 1
        h = image.shape[2] if len(image.shape) >= 3 else 224
        w = image.shape[3] if len(image.shape) >= 4 else 224
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size

        output = ShardedTensor(
            shape=(batch, self.hidden_size, h_patches, w_patches),
            shardable={},
            dtype=self.weight.dtype,
            name="patch_embed_output",
        )

        output._op_history = image._op_history + [
            Conv2dOp(
                dtype=image.dtype,
                input=image,
                weight=self.weight,
                output=output,
                stride=(self.patch_size, self.patch_size),
                padding=(0, 0),
            )
        ]

        self._activations["patch_output"] = output
        return output


class ShardedPositionalEmbedding(ShardedModule):
    """Positional Embedding for Vision Transformer.

    Supports dynamic positional embeddings for variable image sizes.
    For Qwen2-VL: num_position_embeddings=2304 (max patches)

    Args:
        hidden_size: Hidden size
        num_positions: Maximum number of positions (patches)
        dtype: Data type
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        num_positions: int = 2304,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_positions = num_positions

        self.weight = ShardedParameter(
            shape=(num_positions, hidden_size),
            shardable={},
            dtype=dtype,
            name="pos_embed_weight",
        )

    def forward(self, patches: ShardedTensor) -> ShardedTensor:
        """Add positional embeddings.

        Args:
            patches: (batch, num_patches, hidden_size)

        Returns:
            output: (batch, num_patches, hidden_size) with positional embeddings
        """
        output = ShardedTensor(
            shape=patches.shape,
            shardable=patches.shardable,
            dtype=patches.dtype,
            name="pos_embed_output",
        )
        output._op_history = patches._op_history
        self._activations["pos_embed_output"] = output
        return output


class ShardedSpatialMerge(ShardedModule):
    """Spatial Merge for reducing spatial dimensions.

    For Qwen2-VL: spatial_merge_size=2, merges 2x2 patches.

    Args:
        hidden_size: Hidden size
        spatial_merge_size: Merge factor (2 for 2x2 merge)
        dtype: Data type
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        spatial_merge_size: int = 2,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size

        self.weight = ShardedParameter(
            shape=(hidden_size * spatial_merge_size * spatial_merge_size, hidden_size),
            shardable={},
            dtype=dtype,
            name="spatial_merge_weight",
        )

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """Spatial merge forward.

        Args:
            hidden: (batch, num_patches, hidden_size)

        Returns:
            output: (batch, merged_patches, hidden_size)
        """
        batch = hidden.shape[0] if len(hidden.shape) >= 1 else 1
        num_patches = hidden.shape[1] if len(hidden.shape) >= 2 else 1
        merged_patches = num_patches // (self.spatial_merge_size * self.spatial_merge_size)

        output = ShardedTensor(
            shape=(batch, merged_patches, self.hidden_size),
            shardable=hidden.shardable,
            dtype=hidden.dtype,
            name="spatial_merge_output",
        )
        output._op_history = hidden._op_history
        self._activations["spatial_merge_output"] = output
        return output


class ShardedOutputProjection(ShardedModule):
    """Output projection to align with backbone hidden_size.

    For Qwen2-VL: hidden_size=1152 -> out_hidden_size=2048

    Args:
        hidden_size: ViT hidden size (1152)
        out_hidden_size: Backbone hidden size (2048)
        dtype: Data type
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        out_hidden_size: int = 2048,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_hidden_size = out_hidden_size

        self.weight = ShardedParameter(
            shape=(hidden_size, out_hidden_size),
            shardable={},
            dtype=dtype,
            name="output_proj_weight",
        )

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """Project to backbone hidden size.

        Args:
            hidden: (batch, seq, hidden_size)

        Returns:
            output: (batch, seq, out_hidden_size)
        """
        output = hidden @ self.weight
        self._activations["output_proj"] = output
        return output


class ShardedViTEncoder(ShardedModule):
    """Vision Transformer Encoder.

    Independent ViT encoder for multimodal models.
    Can be combined with different backbones through out_hidden_size.

    Structure:
    - PatchEmbedding (conv2d, patch_size=16)
    - PositionalEmbedding
    - N ViT Transformer Layers (depth=27)
    - SpatialMerge (optional)
    - OutputProjection (hidden_size -> out_hidden_size)

    Args:
        depth: Number of transformer layers (27)
        hidden_size: Hidden size (1152)
        num_heads: Number of attention heads (16)
        in_channels: Input channels (3 for RGB)
        patch_size: Patch size (16)
        intermediate_size: MLP intermediate size (4304)
        num_position_embeddings: Max positions (2304)
        out_hidden_size: Output hidden size to align with backbone
        spatial_merge_size: Spatial merge factor (2)
        dtype: Data type
    """

    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        intermediate_size: int = 4304,
        num_position_embeddings: int = 2304,
        out_hidden_size: int = 2048,
        spatial_merge_size: int = 2,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.intermediate_size = intermediate_size
        self.num_position_embeddings = num_position_embeddings
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.dtype = dtype

        self.patch_embed = ShardedPatchEmbedding(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            dtype=dtype,
        )

        self.pos_embed = ShardedPositionalEmbedding(
            hidden_size=hidden_size,
            num_positions=num_position_embeddings,
            dtype=dtype,
        )

        self.layers = []
        for i in range(depth):
            block = ShardedViTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dtype=dtype,
            )
            self.layers.append(block)
        self.layers = self.layers

        self.spatial_merge = ShardedSpatialMerge(
            hidden_size=hidden_size,
            spatial_merge_size=spatial_merge_size,
            dtype=dtype,
        )

        self.output_proj = ShardedOutputProjection(
            hidden_size=hidden_size,
            out_hidden_size=out_hidden_size,
            dtype=dtype,
        )

    def forward(self, image: ShardedTensor) -> ShardedTensor:
        """ViT encoder forward.

        Args:
            image: (batch, channels, height, width)

        Returns:
            output: (batch, merged_patches, out_hidden_size)
        """
        patches = self.patch_embed(image)

        batch = patches.shape[0] if len(patches.shape) >= 1 else 1
        h_patches = patches.shape[2] if len(patches.shape) >= 3 else 14
        w_patches = patches.shape[3] if len(patches.shape) >= 4 else 14
        num_patches = h_patches * w_patches

        hidden = ShardedTensor(
            shape=(batch, num_patches, self.hidden_size),
            shardable={},
            dtype=patches.dtype,
            name="patch_flatten",
        )
        hidden._op_history = patches._op_history

        hidden = self.pos_embed(hidden)
        self._activations["patch_embed_output"] = hidden

        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            self._activations[f"layer_{i}_output"] = hidden

        hidden = self.spatial_merge(hidden)
        self._activations["spatial_merge_output"] = hidden

        output = self.output_proj(hidden)
        self._activations["final_output"] = output

        return output

    def forward_video(self, video: ShardedTensor) -> ShardedTensor:
        """ViT encoder forward for video input.

        Args:
            video: (batch, frames, channels, height, width)

        Returns:
            output: (batch, frames * merged_patches, out_hidden_size)
        """
        batch = video.shape[0] if len(video.shape) >= 1 else 1
        frames = video.shape[1] if len(video.shape) >= 2 else 1
        height = video.shape[3] if len(video.shape) >= 4 else 224
        width = video.shape[4] if len(video.shape) >= 5 else 224

        h_patches = height // self.patch_size
        w_patches = width // self.patch_size
        num_patches = h_patches * w_patches
        merged_patches = num_patches // (self.spatial_merge_size * self.spatial_merge_size)

        patches = ShardedTensor(
            shape=(batch, frames * num_patches, self.hidden_size),
            shardable=video.shardable,
            dtype=video.dtype,
            name="video_patches",
        )
        patches._op_history = video._op_history

        hidden = self.pos_embed(patches)
        self._activations["video_patch_embed_output"] = hidden

        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            self._activations[f"video_layer_{i}_output"] = hidden

        hidden = ShardedTensor(
            shape=(batch, frames * merged_patches, self.hidden_size),
            shardable=hidden.shardable,
            dtype=hidden.dtype,
            name="video_spatial_merge_output",
        )
        hidden._op_history = hidden._op_history
        self._activations["video_spatial_merge_output"] = hidden

        output = self.output_proj(hidden)
        self._activations["video_final_output"] = output

        return output


class ShardedVAEEncoder(ShardedModule):
    """VAE Encoder for image/video latent representation.

    Down-sampling encoder that compresses spatial dimensions by 8x.
    
    Args:
        in_channels: Input channels (typically 3 for RGB)
        latent_channels: Latent channels
        block_out_channels: Channel progression (default: 128 -> 256 -> 512 -> 512)
        use_3d: Use 3D conv for video (True) or 2D for image (False)
        use_attention: Add attention blocks in later stages
        attention_head_dim: Attention head dimension
        dtype: Data type
    
    Spatial compression:
        - 4 downsampling stages (stride=2 each)
        - Total compression: 2^4 = 16x spatial (adjustable via block_out_channels length)
        - Typical VAE: 8x spatial compression (3 downsampling stages)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        use_3d: bool = True,
        use_attention: bool = False,
        attention_head_dim: int = 64,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.block_out_channels = block_out_channels
        self.use_3d = use_3d
        self.use_attention = use_attention

        ConvCls = ShardedConv3d if use_3d else ShardedConv2d
        ResBlockCls = ShardedResNetBlock3d if use_3d else ShardedResNetBlock2d
        AttnBlockCls = ShardedAttentionBlock3d if use_3d else ShardedAttentionBlock2d
        kernel_size = (3, 3, 3) if use_3d else (3, 3)

        self.conv_in = ConvCls(
            in_channels,
            block_out_channels[0],
            kernel_size,
            (1, 1, 1) if use_3d else (1, 1),
            (1, 1, 1) if use_3d else (1, 1),
            dtype=dtype,
        )

        self.down_blocks = []
        for i, out_ch in enumerate(block_out_channels):
            in_ch = block_out_channels[i - 1] if i > 0 else out_ch
            for j in range(2):
                self.down_blocks.append(ResBlockCls(in_ch if j == 0 else out_ch, out_ch, dtype=dtype))
                setattr(self, f"down_{i}_block_{j}", self.down_blocks[-1])

            if use_attention and i >= len(block_out_channels) - 2:
                num_heads = max(1, out_ch // attention_head_dim)
                self.down_blocks.append(AttnBlockCls(out_ch, num_heads, attention_head_dim, dtype=dtype))
                setattr(self, f"down_{i}_attn", self.down_blocks[-1])

            if i < len(block_out_channels) - 1:
                self.down_blocks.append(
                    ConvCls(
                        out_ch,
                        out_ch,
                        kernel_size,
                        (1, 2, 2) if use_3d else (2, 2),
                        (1, 1, 1) if use_3d else (1, 1),
                        dtype=dtype,
                    )
                )
                setattr(self, f"down_{i}_conv", self.down_blocks[-1])

        self.mid_block_0 = ResBlockCls(block_out_channels[-1], block_out_channels[-1], dtype=dtype)

        if use_attention:
            num_heads = max(1, block_out_channels[-1] // attention_head_dim)
            self.mid_attn = AttnBlockCls(block_out_channels[-1], num_heads, attention_head_dim, dtype=dtype)

        self.mid_block_1 = ResBlockCls(block_out_channels[-1], block_out_channels[-1], dtype=dtype)

        self.conv_out = ConvCls(
            block_out_channels[-1],
            latent_channels * 2,
            kernel_size,
            (1, 1, 1) if use_3d else (1, 1),
            (1, 1, 1) if use_3d else (1, 1),
            dtype=dtype,
        )

    def forward(self, video: ShardedTensor) -> ShardedTensor:
        """Encode video/image to latent representation.

        Args:
            video: (batch, channels, height, width) for 2D
                   (batch, channels, frames, height, width) for 3D

        Returns:
            latent: Compressed latent representation (height/8, width/8)
        """
        hidden = self.conv_in(video)

        for block in self.down_blocks:
            hidden = block(hidden)

        hidden = self.mid_block_0(hidden)

        if self.use_attention:
            hidden = self.mid_attn(hidden)

        hidden = self.mid_block_1(hidden)

        latent = self.conv_out(hidden)
        return latent


class ShardedVAEDecoder(ShardedModule):
    """VAE Decoder for reconstructing image/video from latent.

    Up-sampling decoder that restores spatial dimensions by 8x.
    
    Args:
        out_channels: Output channels (typically 3 for RGB)
        latent_channels: Latent channels
        block_out_channels: Channel progression (reversed from encoder)
        use_3d: Use 3D conv for video (True) or 2D for image (False)
        use_attention: Add attention blocks in early stages
        attention_head_dim: Attention head dimension
        dtype: Data type
    
    Spatial restoration:
        - 4 upsampling stages (reverse of encoder)
        - Total restoration: 8x spatial (height*8, width*8)
    """

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        use_3d: bool = True,
        use_attention: bool = False,
        attention_head_dim: int = 64,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.block_out_channels = block_out_channels
        self.use_3d = use_3d
        self.use_attention = use_attention

        ConvCls = ShardedConv3d if use_3d else ShardedConv2d
        ResBlockCls = ShardedResNetBlock3d if use_3d else ShardedResNetBlock2d
        AttnBlockCls = ShardedAttentionBlock3d if use_3d else ShardedAttentionBlock2d
        kernel_size = (3, 3, 3) if use_3d else (3, 3)

        reverse_channels = list(reversed(block_out_channels))

        self.conv_in = ConvCls(
            latent_channels,
            reverse_channels[0],
            kernel_size,
            (1, 1, 1) if use_3d else (1, 1),
            (1, 1, 1) if use_3d else (1, 1),
            dtype=dtype,
        )

        self.mid_block_0 = ResBlockCls(reverse_channels[0], reverse_channels[0], dtype=dtype)

        if use_attention:
            num_heads = max(1, reverse_channels[0] // attention_head_dim)
            self.mid_attn = AttnBlockCls(reverse_channels[0], num_heads, attention_head_dim, dtype=dtype)

        self.mid_block_1 = ResBlockCls(reverse_channels[0], reverse_channels[0], dtype=dtype)

        self.up_blocks = []
        for i, out_ch in enumerate(reverse_channels):
            in_ch = reverse_channels[i - 1] if i > 0 else out_ch
            for j in range(3):
                self.up_blocks.append(ResBlockCls(in_ch if j == 0 else out_ch, out_ch, dtype=dtype))
                setattr(self, f"up_{i}_block_{j}", self.up_blocks[-1])

            if use_attention and i < 2:
                num_heads = max(1, out_ch // attention_head_dim)
                self.up_blocks.append(AttnBlockCls(out_ch, num_heads, attention_head_dim, dtype=dtype))
                setattr(self, f"up_{i}_attn", self.up_blocks[-1])

            if i < len(reverse_channels) - 1:
                self.up_blocks.append(
                    ConvCls(
                        out_ch,
                        out_ch,
                        kernel_size,
                        (1, 1, 1) if use_3d else (1, 1),
                        (1, 1, 1) if use_3d else (1, 1),
                        dtype=dtype,
                    )
                )
                setattr(self, f"up_{i}_conv", self.up_blocks[-1])

        self.norm_out = ShardedGroupNorm(32, reverse_channels[-1], dtype=dtype)
        self.conv_out = ConvCls(
            reverse_channels[-1],
            out_channels,
            kernel_size,
            (1, 1, 1) if use_3d else (1, 1),
            (1, 1, 1) if use_3d else (1, 1),
            dtype=dtype,
        )

    def forward(self, latent: ShardedTensor) -> ShardedTensor:
        """Decode latent to video/image.

        Args:
            latent: Compressed latent representation (height/8, width/8)

        Returns:
            video: Reconstructed video/image (height*8, width*8)
        """
        hidden = self.conv_in(latent)

        hidden = self.mid_block_0(hidden)

        if self.use_attention:
            hidden = self.mid_attn(hidden)

        hidden = self.mid_block_1(hidden)

        for block in self.up_blocks:
            hidden = block(hidden)

        hidden = self.norm_out(hidden)
        hidden = silu(hidden)
        video = self.conv_out(hidden)

        return video


class ShardedVAE(ShardedModule):
    """Complete VAE model with Encoder and Decoder.

    Independent encoder/decoder module for image/video compression.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        latent_channels: Latent channels
        block_out_channels: Channel progression
        use_3d: Use 3D conv for video (True) or 2D for image (False)
        use_attention: Add attention blocks
        attention_head_dim: Attention head dimension
        dtype: Data type
    
    Usage:
        # For video generation models (e.g., Wan)
        vae = ShardedVAE(
            in_channels=3,
            latent_channels=16,
            use_3d=True,
            use_attention=True,
        )
        
        # For image generation models (e.g., Stable Diffusion)
        vae = ShardedVAE(
            in_channels=3,
            latent_channels=4,
            use_3d=False,
            use_attention=False,
        )
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        use_3d: bool = True,
        use_attention: bool = False,
        attention_head_dim: int = 64,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.block_out_channels = block_out_channels
        self.use_3d = use_3d
        self.use_attention = use_attention

        self.encoder = ShardedVAEEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            use_3d=use_3d,
            use_attention=use_attention,
            attention_head_dim=attention_head_dim,
            dtype=dtype,
        )

        self.decoder = ShardedVAEDecoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            use_3d=use_3d,
            use_attention=use_attention,
            attention_head_dim=attention_head_dim,
            dtype=dtype,
        )

    def encode(self, video: ShardedTensor) -> ShardedTensor:
        """Encode video/image to latent representation.

        Args:
            video: Input video/image

        Returns:
            latent: Compressed latent (height/8, width/8)
        """
        return self.encoder(video)

    def decode(self, latent: ShardedTensor) -> ShardedTensor:
        """Decode latent to video/image.

        Args:
            latent: Compressed latent representation

        Returns:
            video: Reconstructed video/image (height*8, width*8)
        """
        return self.decoder(latent)

    def forward(self, video: ShardedTensor) -> ShardedTensor:
        """Full VAE forward pass (encode + decode).

        Args:
            video: Input video/image

        Returns:
            reconstructed: Reconstructed video/image
        """
        latent = self.encode(video)
        reconstructed = self.decode(latent)
        return reconstructed