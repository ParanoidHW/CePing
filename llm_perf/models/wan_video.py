"""Wan2.1 Video Generation Model.

Based on: Wan: Open and Advanced Large-Scale Video Generative Models
Paper: https://arxiv.org/abs/2503.20314

Architecture:
- Text Encoder: umT5-XXL (multilingual T5 encoder)
- DiT Backbone: Diffusion Transformer with cross-attention for text conditioning
- VAE: 3D Causal VAE for spatio-temporal compression

Reference HF model: https://huggingface.co/Wan-AI/Wan2.1-T2V-14B
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

from .base import BaseModel, ModelConfig, LayerConfig
from ..utils.constants import DTYPE_SIZES


@dataclass
class WanTextEncoderConfig(ModelConfig):
    """Wan Text Encoder (umT5-XXL) configuration.
    
    umT5 is a multilingual T5 encoder used for encoding text prompts.
    Reference: https://arxiv.org/abs/2302.00093
    """
    # T5 architecture
    hidden_size: int = 4096
    num_layers: int = 24
    num_attention_heads: int = 64
    intermediate_size: int = 10240  # T5 uses smaller FFN
    
    # T5 specific
    d_kv: int = 64  # Key/Value dimension per head
    num_key_value_heads: int = 64  # T5 uses MHA
    
    # Text processing
    max_text_len: int = 512  # Context length for text encoder
    vocab_size: int = 256384  # umT5 vocabulary size
    
    def __post_init__(self):
        """Validate config."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass  
class WanDiTConfig(ModelConfig):
    """Wan Diffusion Transformer (DiT) configuration.
    
    DiT backbone for video generation using Flow Matching framework.
    Each transformer block uses cross-attention for text conditioning.
    """
    # DiT architecture (14B model)
    hidden_size: int = 5120  # dim in Wan paper
    num_layers: int = 40
    num_attention_heads: int = 40
    
    # FFN dimension
    intermediate_size: int = 13824  # ffn_dim in Wan paper
    
    # Patchify config
    patch_size: Tuple[int, int, int] = (1, 2, 2)  # (t, h, w)
    in_channels: int = 16  # Input latent channels from VAE
    out_channels: int = 16  # Output latent channels
    
    # Text conditioning
    text_dim: int = 4096  # Text encoder output dimension
    
    # Time embedding
    freq_dim: int = 256  # Frequency dimension for time embedding
    
    # Video dimensions (latent space after VAE encoding)
    latent_num_frames: int = 21  # (81 // 4) + 1 = 21 for 81-frame video
    latent_height: int = 90  # 720 // 8 = 90 for 720p
    latent_width: int = 160  # 1280 // 8 = 160 for 1280x720
    
    # Number of key/value heads (for GQA, Wan uses MHA)
    num_key_value_heads: int = 40
    
    # Flow matching config
    num_inference_steps: int = 50  # Default denoising steps
    
    def __post_init__(self):
        """Validate config."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class WanVAEConfig(ModelConfig):
    """Wan 3D Causal VAE configuration.
    
    3D causal VAE for spatio-temporal video compression.
    Temporal compression: 4x
    Spatial compression: 8x8
    """
    # VAE architecture
    in_channels: int = 3  # RGB input
    out_channels: int = 3  # RGB output
    latent_channels: int = 16  # Latent space channels
    
    # Video dimensions (pixel space)
    num_frames: int = 81  # Typical video length
    height: int = 720
    width: int = 1280
    
    # Compression ratios
    temporal_compression: int = 4  # T -> T/4
    spatial_compression: int = 8  # H, W -> H/8, W/8
    
    # Encoder/decoder channels
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    
    # Causal VAE specific
    use_causal_conv: bool = True
    
    # Not used but required by base class
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    
    def __post_init__(self):
        """Set derived attributes."""
        # Calculate latent dimensions
        self.latent_num_frames = (self.num_frames - 1) // self.temporal_compression + 1
        self.latent_height = self.height // self.spatial_compression
        self.latent_width = self.width // self.spatial_compression
        
        if self.num_key_value_heads is None:
            self.num_key_value_heads = 0


class WanTextEncoder(BaseModel):
    """Wan Text Encoder (umT5-XXL).
    
    Encodes text prompts into embeddings for DiT cross-attention.
    Uses standard T5 encoder architecture.
    """
    
    def __init__(self, config: WanTextEncoderConfig):
        super().__init__(config)
        self._layers = self.build_layers()
    
    def build_layers(self) -> List[LayerConfig]:
        """Build T5 encoder layers."""
        layers = []
        cfg = self.config
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)
        
        # Token embedding
        layers.append(LayerConfig(
            name="embed_tokens",
            input_shape=(1, cfg.max_text_len),
            output_shape=(1, cfg.max_text_len, cfg.hidden_size),
            params_count=cfg.vocab_size * cfg.hidden_size,
            flops=0,  # Lookup
            activation_bytes=cfg.max_text_len * cfg.hidden_size * dtype_size,
        ))
        
        # Encoder layers (24 layers for umT5-XXL)
        for i in range(cfg.num_layers):
            layers.extend(self._build_encoder_block(i, dtype_size))
        
        # Final layer norm
        layers.append(LayerConfig(
            name="final_layer_norm",
            input_shape=(1, cfg.max_text_len, cfg.hidden_size),
            output_shape=(1, cfg.max_text_len, cfg.hidden_size),
            params_count=cfg.hidden_size * 2,
            flops=cfg.max_text_len * cfg.hidden_size * 7,
            activation_bytes=cfg.max_text_len * cfg.hidden_size * dtype_size,
        ))
        
        return layers
    
    def _build_encoder_block(self, layer_idx: int, dtype_size: int) -> List[LayerConfig]:
        """Build a single T5 encoder block.
        
        Structure: Self-Attn -> LayerNorm -> FFN -> LayerNorm
        """
        layers = []
        cfg = self.config
        seq_len = cfg.max_text_len
        
        # Self-attention with relative positional bias
        # Q, K, V projections
        qkv_params = 3 * cfg.hidden_size * cfg.hidden_size
        qkv_flops = 3 * seq_len * cfg.hidden_size * cfg.hidden_size
        
        layers.append(LayerConfig(
            name=f"layer_{layer_idx}_self_attn_qkv",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, 3 * cfg.hidden_size),
            params_count=qkv_params,
            flops=qkv_flops,
            activation_bytes=seq_len * 3 * cfg.hidden_size * dtype_size,
        ))
        
        # Attention computation (Q @ K^T @ V)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        attn_flops = (
            2 * seq_len * seq_len * cfg.hidden_size +  # QK^T
            2 * seq_len * seq_len * cfg.hidden_size    # @V
        )
        
        layers.append(LayerConfig(
            name=f"layer_{layer_idx}_self_attn_compute",
            input_shape=(1, seq_len, 3 * cfg.hidden_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=0,
            flops=attn_flops,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
        ))
        
        # Output projection
        layers.append(LayerConfig(
            name=f"layer_{layer_idx}_self_attn_o",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=cfg.hidden_size * cfg.hidden_size,
            flops=seq_len * cfg.hidden_size * cfg.hidden_size,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
        ))
        
        # Layer norm after attention
        layers.append(LayerConfig(
            name=f"layer_{layer_idx}_post_attn_norm",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=cfg.hidden_size * 2,
            flops=seq_len * cfg.hidden_size * 7,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
        ))
        
        # FFN (T5 uses gated GeLU: wi_0, wi_1 -> wo)
        # Two input projections
        ffn_in_params = 2 * cfg.hidden_size * cfg.intermediate_size
        ffn_in_flops = 2 * seq_len * cfg.hidden_size * cfg.intermediate_size
        
        layers.append(LayerConfig(
            name=f"layer_{layer_idx}_ffn_in",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, 2 * cfg.intermediate_size),
            params_count=ffn_in_params,
            flops=ffn_in_flops,
            activation_bytes=seq_len * 2 * cfg.intermediate_size * dtype_size,
        ))
        
        # GeLU activation and element-wise product (gated)
        layers.append(LayerConfig(
            name=f"layer_{layer_idx}_ffn_act",
            input_shape=(1, seq_len, 2 * cfg.intermediate_size),
            output_shape=(1, seq_len, cfg.intermediate_size),
            params_count=0,
            flops=seq_len * cfg.intermediate_size * 10,  # GeLU ~10 FLOPs
            activation_bytes=seq_len * cfg.intermediate_size * dtype_size,
        ))
        
        # Output projection
        layers.append(LayerConfig(
            name=f"layer_{layer_idx}_ffn_out",
            input_shape=(1, seq_len, cfg.intermediate_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=cfg.intermediate_size * cfg.hidden_size,
            flops=seq_len * cfg.intermediate_size * cfg.hidden_size,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
        ))
        
        # Final layer norm
        layers.append(LayerConfig(
            name=f"layer_{layer_idx}_final_norm",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=cfg.hidden_size * 2,
            flops=seq_len * cfg.hidden_size * 7,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
        ))
        
        return layers


class WanDiTModel(BaseModel):
    """Wan Diffusion Transformer (DiT) Model.
    
    Core video generation backbone using Flow Matching framework.
    Each block uses:
    - Self-attention on spatial-temporal patches
    - Cross-attention for text conditioning
    - FFN with time embedding modulation
    """
    
    def __init__(self, config: WanDiTConfig):
        super().__init__(config)
        self._layers = self.build_layers()
    
    def build_layers(self) -> List[LayerConfig]:
        """Build DiT layers."""
        layers = []
        cfg = self.config
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)
        
        # Patchify: 3D conv (1, 2, 2) kernel
        layers.append(self._build_patchify_layer(dtype_size))
        
        # Time embedding MLP (shared across all blocks)
        layers.extend(self._build_time_embedding_mlp(dtype_size))
        
        # Transformer blocks
        for i in range(cfg.num_layers):
            layers.extend(self._build_transformer_block(i, dtype_size))
        
        # Unpatchify: output projection
        layers.append(self._build_unpatchify_layer(dtype_size))
        
        return layers
    
    def _build_patchify_layer(self, dtype_size: int) -> LayerConfig:
        """Build patchify convolution layer."""
        cfg = self.config
        pt, ph, pw = cfg.patch_size
        
        # Output sequence length after patchify
        out_t = cfg.latent_num_frames // pt
        out_h = cfg.latent_height // ph
        out_w = cfg.latent_width // pw
        seq_len = out_t * out_h * out_w
        
        # 3D conv for patchify
        kernel_params = (pt * ph * pw * cfg.in_channels * cfg.hidden_size)
        
        # FLOPs for convolution
        flops = (
            2 * seq_len * pt * ph * pw *
            cfg.in_channels * cfg.hidden_size
        )
        
        return LayerConfig(
            name="patchify",
            input_shape=(1, cfg.in_channels, cfg.latent_num_frames, 
                        cfg.latent_height, cfg.latent_width),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=kernel_params,
            flops=flops,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
        )
    
    def _build_time_embedding_mlp(self, dtype_size: int) -> List[LayerConfig]:
        """Build time embedding MLP (shared across blocks)."""
        layers = []
        cfg = self.config
        
        # Time embedding uses freq_dim -> hidden_size
        # MLP: freq_dim -> hidden_size (with SiLU)
        layers.append(LayerConfig(
            name="time_mlp_in",
            input_shape=(1, cfg.freq_dim),
            output_shape=(1, cfg.hidden_size),
            params_count=cfg.freq_dim * cfg.hidden_size,
            flops=cfg.freq_dim * cfg.hidden_size,
            activation_bytes=cfg.hidden_size * dtype_size,
        ))
        
        # SiLU activation
        layers.append(LayerConfig(
            name="time_mlp_act",
            input_shape=(1, cfg.hidden_size),
            output_shape=(1, cfg.hidden_size),
            params_count=0,
            flops=cfg.hidden_size * 8,  # SiLU ~8 FLOPs
            activation_bytes=cfg.hidden_size * dtype_size,
        ))
        
        # Output projection (predicts 6 modulation parameters)
        layers.append(LayerConfig(
            name="time_mlp_out",
            input_shape=(1, cfg.hidden_size),
            output_shape=(1, 6 * cfg.hidden_size),  # 6 modulation params per block
            params_count=cfg.hidden_size * 6 * cfg.hidden_size,
            flops=cfg.hidden_size * 6 * cfg.hidden_size,
            activation_bytes=6 * cfg.hidden_size * dtype_size,
        ))
        
        return layers
    
    def _build_transformer_block(self, layer_idx: int, dtype_size: int) -> List[LayerConfig]:
        """Build a single DiT transformer block.
        
        Structure:
        1. Self-attention (spatial-temporal)
        2. Cross-attention (text conditioning)
        3. FFN (with time modulation via AdaLN)
        """
        layers = []
        cfg = self.config
        
        # Calculate sequence length
        pt, ph, pw = cfg.patch_size
        seq_len = (
            (cfg.latent_num_frames // pt) *
            (cfg.latent_height // ph) *
            (cfg.latent_width // pw)
        )
        text_len = 512  # Max text length
        
        # === Self-Attention ===
        # QKV projection
        qkv_params = 3 * cfg.hidden_size * cfg.hidden_size
        qkv_flops = 3 * seq_len * cfg.hidden_size * cfg.hidden_size
        
        layers.append(LayerConfig(
            name=f"block_{layer_idx}_self_attn_qkv",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, 3 * cfg.hidden_size),
            params_count=qkv_params,
            flops=qkv_flops,
            activation_bytes=seq_len * 3 * cfg.hidden_size * dtype_size,
        ))
        
        # Self-attention computation
        attn_flops = (
            2 * seq_len * seq_len * cfg.hidden_size +  # QK^T
            2 * seq_len * seq_len * cfg.hidden_size    # @V
        )
        
        layers.append(LayerConfig(
            name=f"block_{layer_idx}_self_attn_compute",
            input_shape=(1, seq_len, 3 * cfg.hidden_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=0,
            flops=attn_flops,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
        ))
        
        # Output projection
        layers.append(LayerConfig(
            name=f"block_{layer_idx}_self_attn_o",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=cfg.hidden_size * cfg.hidden_size,
            flops=seq_len * cfg.hidden_size * cfg.hidden_size,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
        ))
        
        # === Cross-Attention (Text Conditioning) ===
        # Q from visual, KV from text
        # Q projection
        layers.append(LayerConfig(
            name=f"block_{layer_idx}_cross_attn_q",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=cfg.hidden_size * cfg.hidden_size,
            flops=seq_len * cfg.hidden_size * cfg.hidden_size,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
        ))
        
        # K, V projections from text
        kv_params = 2 * cfg.text_dim * cfg.hidden_size
        kv_flops = 2 * text_len * cfg.text_dim * cfg.hidden_size
        
        layers.append(LayerConfig(
            name=f"block_{layer_idx}_cross_attn_kv",
            input_shape=(1, text_len, cfg.text_dim),
            output_shape=(1, text_len, 2 * cfg.hidden_size),
            params_count=kv_params,
            flops=kv_flops,
            activation_bytes=text_len * 2 * cfg.hidden_size * dtype_size,
        ))
        
        # Cross-attention computation
        cross_attn_flops = (
            2 * seq_len * text_len * cfg.hidden_size +  # QK^T
            2 * seq_len * text_len * cfg.hidden_size    # @V
        )
        
        layers.append(LayerConfig(
            name=f"block_{layer_idx}_cross_attn_compute",
            input_shape=(1, seq_len + text_len, cfg.hidden_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=0,
            flops=cross_attn_flops,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
        ))
        
        # Output projection
        layers.append(LayerConfig(
            name=f"block_{layer_idx}_cross_attn_o",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=cfg.hidden_size * cfg.hidden_size,
            flops=seq_len * cfg.hidden_size * cfg.hidden_size,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
        ))
        
        # === FFN with AdaLN modulation ===
        # MLP: hidden_size -> intermediate_size -> hidden_size
        # With gated GeLU similar to T5
        ffn_in_params = 2 * cfg.hidden_size * cfg.intermediate_size
        ffn_in_flops = 2 * seq_len * cfg.hidden_size * cfg.intermediate_size
        
        layers.append(LayerConfig(
            name=f"block_{layer_idx}_ffn_in",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, 2 * cfg.intermediate_size),
            params_count=ffn_in_params,
            flops=ffn_in_flops,
            activation_bytes=seq_len * 2 * cfg.intermediate_size * dtype_size,
        ))
        
        # GeLU activation (gated)
        layers.append(LayerConfig(
            name=f"block_{layer_idx}_ffn_act",
            input_shape=(1, seq_len, 2 * cfg.intermediate_size),
            output_shape=(1, seq_len, cfg.intermediate_size),
            params_count=0,
            flops=seq_len * cfg.intermediate_size * 10,
            activation_bytes=seq_len * cfg.intermediate_size * dtype_size,
        ))
        
        # Output projection
        layers.append(LayerConfig(
            name=f"block_{layer_idx}_ffn_out",
            input_shape=(1, seq_len, cfg.intermediate_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=cfg.intermediate_size * cfg.hidden_size,
            flops=seq_len * cfg.intermediate_size * cfg.hidden_size,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
        ))
        
        return layers
    
    def _build_unpatchify_layer(self, dtype_size: int) -> LayerConfig:
        """Build unpatchify output projection."""
        cfg = self.config
        pt, ph, pw = cfg.patch_size
        
        # Sequence length
        seq_len = (
            (cfg.latent_num_frames // pt) *
            (cfg.latent_height // ph) *
            (cfg.latent_width // pw)
        )
        
        # Output: hidden_size -> patch_size[0] * patch_size[1] * patch_size[2] * out_channels
        out_dim = pt * ph * pw * cfg.out_channels
        
        params = cfg.hidden_size * out_dim
        flops = seq_len * cfg.hidden_size * out_dim
        
        return LayerConfig(
            name="unpatchify",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, cfg.out_channels, cfg.latent_num_frames,
                        cfg.latent_height, cfg.latent_width),
            params_count=params,
            flops=flops,
            activation_bytes=seq_len * out_dim * dtype_size,
        )


class WanVAEModel(BaseModel):
    """Wan 3D Causal VAE Model.
    
    Spatio-temporal VAE for video compression:
    - Temporal compression: 4x
    - Spatial compression: 8x8
    - Causal convolution for temporal dimension
    """
    
    def __init__(self, config: WanVAEConfig):
        super().__init__(config)
        self._layers = self.build_layers()
    
    def build_layers(self) -> List[LayerConfig]:
        """Build VAE encoder and decoder layers."""
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
        """Build 3D causal VAE encoder."""
        layers = []
        cfg = self.config
        
        # Calculate dimensions at each level
        dims = []
        current_t = cfg.num_frames
        current_h = cfg.height
        current_w = cfg.width
        
        for i, out_ch in enumerate(cfg.block_out_channels):
            dims.append((current_t, current_h, current_w, out_ch))
            if i < len(cfg.block_out_channels) - 1:
                current_t = (current_t - 1) // cfg.temporal_compression + 1
                current_h //= cfg.spatial_compression
                current_w //= cfg.spatial_compression
        
        # Initial convolution
        layers.append(self._build_conv3d_causal(
            name="encoder_conv_in",
            in_channels=cfg.in_channels,
            out_channels=cfg.block_out_channels[0],
            kernel_size=(3, 3, 3),
            input_dims=(cfg.num_frames, cfg.height, cfg.width),
            dtype_size=dtype_size,
        ))
        
        # Encoder blocks with downsampling
        for i, out_ch in enumerate(cfg.block_out_channels):
            in_ch = cfg.block_out_channels[i-1] if i > 0 else cfg.block_out_channels[0]
            
            # ResNet blocks
            for j in range(cfg.layers_per_block):
                layers.extend(self._build_resnet_block(
                    name=f"encoder_down_{i}_resnet_{j}",
                    in_channels=in_ch if j == 0 else out_ch,
                    out_channels=out_ch,
                    dims=dims[i][:3],
                    dtype_size=dtype_size,
                ))
            
            # Downsample (except last)
            if i < len(cfg.block_out_channels) - 1:
                layers.append(self._build_conv3d_causal(
                    name=f"encoder_down_{i}_downsample",
                    in_channels=out_ch,
                    out_channels=out_ch,
                    kernel_size=(3, 3, 3),
                    input_dims=dims[i][:3],
                    stride=(1, 2, 2),
                    dtype_size=dtype_size,
                ))
        
        # Mid block
        mid_ch = cfg.block_out_channels[-1]
        mid_dims = dims[-1][:3]
        layers.extend(self._build_resnet_block(
            name="encoder_mid_resnet",
            in_channels=mid_ch,
            out_channels=mid_ch,
            dims=mid_dims,
            dtype_size=dtype_size,
        ))
        
        # Output to latent (mean and logvar)
        layers.append(self._build_conv3d_causal(
            name="encoder_conv_out",
            in_channels=mid_ch,
            out_channels=cfg.latent_channels * 2,  # mean + logvar
            kernel_size=(3, 3, 3),
            input_dims=mid_dims,
            dtype_size=dtype_size,
        ))
        
        return layers
    
    def _build_decoder(self, dtype_size: int) -> List[LayerConfig]:
        """Build 3D causal VAE decoder."""
        layers = []
        cfg = self.config
        
        # Calculate latent dimensions
        latent_t = (cfg.num_frames - 1) // cfg.temporal_compression + 1
        latent_h = cfg.height // cfg.spatial_compression
        latent_w = cfg.width // cfg.spatial_compression
        
        reverse_channels = list(reversed(cfg.block_out_channels))
        
        # Initial convolution from latent
        layers.append(self._build_conv3d_causal(
            name="decoder_conv_in",
            in_channels=cfg.latent_channels,
            out_channels=reverse_channels[0],
            kernel_size=(3, 3, 3),
            input_dims=(latent_t, latent_h, latent_w),
            dtype_size=dtype_size,
        ))
        
        # Mid block
        mid_ch = reverse_channels[0]
        layers.extend(self._build_resnet_block(
            name="decoder_mid_resnet",
            in_channels=mid_ch,
            out_channels=mid_ch,
            dims=(latent_t, latent_h, latent_w),
            dtype_size=dtype_size,
        ))
        
        # Decoder blocks with upsampling
        current_t, current_h, current_w = latent_t, latent_h, latent_w
        
        for i, out_ch in enumerate(reverse_channels):
            in_ch = reverse_channels[i-1] if i > 0 else reverse_channels[0]
            
            # ResNet blocks
            for j in range(cfg.layers_per_block + 1):
                layers.extend(self._build_resnet_block(
                    name=f"decoder_up_{i}_resnet_{j}",
                    in_channels=in_ch if j == 0 else out_ch,
                    out_channels=out_ch,
                    dims=(current_t, current_h, current_w),
                    dtype_size=dtype_size,
                ))
            
            # Upsample (except last)
            if i < len(reverse_channels) - 1:
                layers.append(self._build_conv3d_causal(
                    name=f"decoder_up_{i}_upsample",
                    in_channels=out_ch,
                    out_channels=out_ch,
                    kernel_size=(3, 3, 3),
                    input_dims=(current_t, current_h, current_w),
                    dtype_size=dtype_size,
                ))
                current_h *= 2
                current_w *= 2
        
        # Output convolution
        layers.append(self._build_conv3d_causal(
            name="decoder_conv_out",
            in_channels=reverse_channels[-1],
            out_channels=cfg.out_channels,
            kernel_size=(3, 3, 3),
            input_dims=(current_t, current_h, current_w),
            dtype_size=dtype_size,
        ))
        
        return layers
    
    def _build_conv3d_causal(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        input_dims: Tuple[int, int, int],
        dtype_size: int,
        stride: Tuple[int, int, int] = (1, 1, 1),
    ) -> LayerConfig:
        """Build 3D causal convolution layer."""
        kt, kh, kw = kernel_size
        st, sh, sw = stride
        input_t, input_h, input_w = input_dims
        
        # Causal padding for temporal dimension
        if self.config.use_causal_conv:
            output_t = (input_t - 1) // st + 1
        else:
            output_t = (input_t + 2 * 1 - kt) // st + 1
        output_h = (input_h + 2 * 1 - kh) // sh + 1
        output_w = (input_w + 2 * 1 - kw) // sw + 1
        
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
        dims: Tuple[int, int, int],
        dtype_size: int,
    ) -> List[LayerConfig]:
        """Build ResNet block with GroupNorm and 3D conv."""
        layers = []
        t, h, w = dims
        
        # GroupNorm 1
        layers.append(LayerConfig(
            name=f"{name}_norm1",
            input_shape=(1, in_channels, t, h, w),
            output_shape=(1, in_channels, t, h, w),
            params_count=in_channels * 2,
            flops=in_channels * t * h * w * 7,
            activation_bytes=in_channels * t * h * w * dtype_size,
        ))
        
        # Conv1
        layers.append(self._build_conv3d_causal(
            name=f"{name}_conv1",
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            input_dims=dims,
            dtype_size=dtype_size,
        ))
        
        # GroupNorm 2
        layers.append(LayerConfig(
            name=f"{name}_norm2",
            input_shape=(1, out_channels, t, h, w),
            output_shape=(1, out_channels, t, h, w),
            params_count=out_channels * 2,
            flops=out_channels * t * h * w * 7,
            activation_bytes=out_channels * t * h * w * dtype_size,
        ))
        
        # Conv2
        layers.append(self._build_conv3d_causal(
            name=f"{name}_conv2",
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            input_dims=dims,
            dtype_size=dtype_size,
        ))
        
        # Shortcut if channels change
        if in_channels != out_channels:
            layers.append(self._build_conv3d_causal(
                name=f"{name}_shortcut",
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1, 1),
                input_dims=dims,
                dtype_size=dtype_size,
            ))
        
        return layers


# Convenience factory functions
def create_wan_t2v_14b_text_encoder(dtype: str = "bf16") -> WanTextEncoder:
    """Create Wan2.1-T2V-14B text encoder (umT5-XXL)."""
    config = WanTextEncoderConfig(
        name="wan-t2v-14b-text-encoder",
        vocab_size=256384,
        hidden_size=4096,
        num_layers=24,
        num_attention_heads=64,
        intermediate_size=10240,
        max_text_len=512,
        dtype=dtype,
    )
    return WanTextEncoder(config)


def create_wan_t2v_14b_dit(
    num_frames: int = 81,
    height: int = 720,
    width: int = 1280,
    dtype: str = "bf16",
) -> WanDiTModel:
    """Create Wan2.1-T2V-14B DiT model."""
    # VAE compression
    temporal_comp = 4
    spatial_comp = 8
    
    config = WanDiTConfig(
        name="wan-t2v-14b-dit",
        hidden_size=5120,
        num_layers=40,
        num_attention_heads=40,
        intermediate_size=13824,
        patch_size=(1, 2, 2),
        in_channels=16,
        out_channels=16,
        text_dim=4096,
        freq_dim=256,
        latent_num_frames=(num_frames - 1) // temporal_comp + 1,
        latent_height=height // spatial_comp,
        latent_width=width // spatial_comp,
        num_inference_steps=50,
        vocab_size=0,  # Not used
        dtype=dtype,
    )
    return WanDiTModel(config)


def create_wan_t2v_vae(
    num_frames: int = 81,
    height: int = 720,
    width: int = 1280,
    dtype: str = "bf16",
) -> WanVAEModel:
    """Create Wan2.1 VAE model."""
    config = WanVAEConfig(
        name="wan-vae",
        in_channels=3,
        out_channels=3,
        latent_channels=16,
        num_frames=num_frames,
        height=height,
        width=width,
        temporal_compression=4,
        spatial_compression=8,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        use_causal_conv=True,
        vocab_size=0,  # Not used
        hidden_size=0,  # Not used
        num_layers=0,  # Computed dynamically
        num_attention_heads=0,  # Not used
        dtype=dtype,
    )
    return WanVAEModel(config)
