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
from typing import List, Tuple

from .base import BaseModel, ModelConfig, LayerConfig
from ..utils.constants import DTYPE_SIZES
from ..kernels import linear, layer_norm, rms_norm, gelu, silu, conv3d, embedding
from ..kernels.utils import kernel_result_to_layer


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
        """Build T5 encoder layers using kernel API."""
        layers = []
        cfg = self.config
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)
        
        # Token embedding using embedding kernel
        emb_result = embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.hidden_size,
            input_shape=(1, cfg.max_text_len),
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="embed_tokens",
            result=emb_result))
        
        # Encoder layers (24 layers for umT5-XXL)
        for i in range(cfg.num_layers):
            layers.extend(self._build_encoder_block(i, dtype_size))
        
        # Final layer norm using layer_norm kernel
        ln_result = layer_norm(
            input=(1, cfg.max_text_len, cfg.hidden_size),
            normalized_shape=(cfg.hidden_size,),
            elementwise_affine=True,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="final_layer_norm",
            result=ln_result))
        
        return layers
    
    def _build_encoder_block(self, layer_idx: int, dtype_size: int) -> List[LayerConfig]:
        """Build a single T5 encoder block using kernel API.
        
        Structure: Self-Attn -> LayerNorm -> FFN -> LayerNorm
        """
        layers = []
        cfg = self.config
        seq_len = cfg.max_text_len
        
        # Self-attention with relative positional bias
        # Q, K, V projections using linear kernel
        # Each projection: (seq_len, hidden_size) @ (hidden_size, hidden_size)
        m = seq_len
        
        # QKV projections: 3 separate linear layers
        qkv_result = linear(
            input=(m, cfg.hidden_size),
            weight=(3 * cfg.hidden_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"layer_{layer_idx}_self_attn_qkv",
            result=qkv_result))
        
        # Attention computation (Q @ K^T @ V)
        # Using bmm kernel for batch matrix multiply
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        from ..kernels import bmm
        
        # Softmax + @V
        attn_v_result = bmm(
            input=(cfg.num_attention_heads, seq_len, seq_len),
            mat2=(cfg.num_attention_heads, seq_len, head_dim),
            dtype=cfg.dtype
        )
        
        layers.append(kernel_result_to_layer(
            name=f"layer_{layer_idx}_self_attn_compute",
            result=attn_v_result))
        
        # Output projection using linear kernel
        o_result = linear(
            input=(m, cfg.hidden_size),
            weight=(cfg.hidden_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"layer_{layer_idx}_self_attn_o",
            result=o_result))
        
        # Layer norm after attention using layer_norm kernel
        ln1_result = layer_norm(
            input=(1, seq_len, cfg.hidden_size),
            normalized_shape=(cfg.hidden_size,),
            elementwise_affine=True,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"layer_{layer_idx}_post_attn_norm",
            result=ln1_result))
        
        # FFN (T5 uses gated GeLU: wi_0, wi_1 -> wo)
        # Two input projections using linear kernel
        ffn_in_result = linear(
            input=(m, cfg.hidden_size),
            weight=(2 * cfg.intermediate_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"layer_{layer_idx}_ffn_in",
            result=ffn_in_result))
        
        # GeLU activation using gelu kernel
        from ..kernels import gelu
        gelu_result = gelu(
            input=(1, seq_len, cfg.intermediate_size),
            approximate="tanh",
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"layer_{layer_idx}_ffn_act",
            result=gelu_result))
        
        # Output projection using linear kernel
        ffn_out_result = linear(
            input=(m, cfg.intermediate_size),
            weight=(cfg.hidden_size, cfg.intermediate_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"layer_{layer_idx}_ffn_out",
            result=ffn_out_result))
        
        # Final layer norm using layer_norm kernel
        ln2_result = layer_norm(
            input=(1, seq_len, cfg.hidden_size),
            normalized_shape=(cfg.hidden_size,),
            elementwise_affine=True,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"layer_{layer_idx}_final_norm",
            result=ln2_result))
        
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
        """Build patchify convolution layer using conv3d kernel."""
        cfg = self.config
        pt, ph, pw = cfg.patch_size
        
        # 3D conv for patchify using kernel API
        conv_result = conv3d(
            input=(1, cfg.in_channels, cfg.latent_num_frames, cfg.latent_height, cfg.latent_width),
            weight=(cfg.hidden_size, cfg.in_channels, pt, ph, pw),
            bias=None,
            stride=(pt, ph, pw),
            padding=(0, 0, 0),
            dtype=cfg.dtype
        )
        
        return kernel_result_to_layer(
            name="patchify",
            result=conv_result)
    
    def _build_time_embedding_mlp(self, dtype_size: int) -> List[LayerConfig]:
        """Build time embedding MLP (shared across all blocks) using kernel API.
        
        From Wan2.1 reference (model.py line 462-464):
        - time_embedding: freq_dim -> dim (with SiLU)
        - time_projection: dim -> dim * 6 (projects to 6 modulation params)
        """
        layers = []
        cfg = self.config
        
        # Time embedding: freq_dim -> hidden_size (with SiLU)
        te_in_result = linear(
            input=(1, cfg.freq_dim),
            weight=(cfg.hidden_size, cfg.freq_dim),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="time_embedding_in",
            result=te_in_result))
        
        # SiLU activation
        silu_result = silu(
            input=(1, cfg.hidden_size),
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="time_embedding_act",
            result=silu_result))
        
        # Output projection to shared time embedding
        te_out_result = linear(
            input=(1, cfg.hidden_size),
            weight=(cfg.hidden_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="time_embedding_out",
            result=te_out_result))
        
        # Time projection: projects time embedding to 6 modulation parameters per block
        tp_result = linear(
            input=(1, cfg.hidden_size),
            weight=(6 * cfg.hidden_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="time_projection",
            result=tp_result))
        
        return layers
    
    def _build_modulation_layer(self, layer_idx: int, dtype_size: int, seq_len: int) -> List[LayerConfig]:
        """Build modulation layer for a block.
        
        From Wan2.1 reference (model.py line 276):
        - Each block has self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        - 6 modulation parameters: shift/scale/gate for self-attn + shift/scale/gate for FFN
        - Cross-attention does NOT use modulation (model.py line 310)
        """
        layers = []
        cfg = self.config
        
        # Modulation: 6 vectors of hidden_size
        # e[0], e[1], e[2]: shift, scale, gate for self-attention
        # e[3], e[4], e[5]: shift, scale, gate for FFN
        num_modulation = 6 * cfg.hidden_size
        
        # Layer-specific modulation parameters (learned)
        # NOTE: Manual calculation for custom modulation layer (parameter storage layer)
        layers.append(LayerConfig(
            name=f"block_{layer_idx}_modulation",
            input_shape=(1, 6, cfg.hidden_size),  # (self_attn_shift, self_attn_scale, self_attn_gate, ffn_shift, ffn_scale, ffn_gate)
            output_shape=(1, 6, cfg.hidden_size),
            params_count=num_modulation,  # 6 * hidden_size learned parameters
            flops=seq_len * cfg.hidden_size * 6,  # Apply modulation: scale, shift, gate ops
            activation_bytes=num_modulation * dtype_size,
        ))
        
        return layers
    
    def _build_transformer_block(self, layer_idx: int, dtype_size: int) -> List[LayerConfig]:
        """Build a single DiT transformer block using kernel API.
        
        Structure based on Wan2.1 reference (model.py line 238-317):
        1. Modulation (6 params: shift/scale/gate for self-attn + shift/scale/gate for FFN)
        2. LayerNorm + Self-attention (with Q/K RMSNorm)
        3. LayerNorm + Cross-attention (NO modulation)
        4. LayerNorm + FFN (with modulation)
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
        m = seq_len  # Flattened batch*seq for linear ops
        
        # === Modulation (6 parameters per block) ===
        layers.extend(self._build_modulation_layer(layer_idx, dtype_size, seq_len))
        
        # === Self-Attention ===
        # Pre-self-attn LayerNorm (norm1 in Wan2.1, no elementwise_affine)
        ln1_result = layer_norm(
            input=(1, seq_len, cfg.hidden_size),
            normalized_shape=(cfg.hidden_size,),
            elementwise_affine=False,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"block_{layer_idx}_norm1",
            result=ln1_result))
        
        # QKV projection using linear kernel
        qkv_result = linear(
            input=(m, cfg.hidden_size),
            weight=(3 * cfg.hidden_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"block_{layer_idx}_self_attn_qkv",
            result=qkv_result))
        
        # Q/K RMSNorm using rms_norm kernel
        qk_norm_result = rms_norm(
            input=(1, seq_len, 2 * cfg.hidden_size),
            dim=-1,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"block_{layer_idx}_self_attn_qk_norm",
            result=qk_norm_result))
        
        # Self-attention computation using scaled_dot_product_attention kernel
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        from ..kernels import scaled_dot_product_attention
        attn_result = scaled_dot_product_attention(
            query=(1, cfg.num_attention_heads, seq_len, head_dim),
            key=(1, cfg.num_attention_heads, seq_len, head_dim),
            value=(1, cfg.num_attention_heads, seq_len, head_dim),
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"block_{layer_idx}_self_attn_compute",
            result=attn_result))
        
        # Output projection using linear kernel
        o_result = linear(
            input=(m, cfg.hidden_size),
            weight=(cfg.hidden_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"block_{layer_idx}_self_attn_o",
            result=o_result))
        
        # === Cross-Attention (Text Conditioning) ===
        # Pre-cross-attn LayerNorm with elementwise_affine
        ln3_result = layer_norm(
            input=(1, seq_len, cfg.hidden_size),
            normalized_shape=(cfg.hidden_size,),
            elementwise_affine=True,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"block_{layer_idx}_norm3",
            result=ln3_result))
        
        # Q projection for cross-attn
        cross_q_result = linear(
            input=(m, cfg.hidden_size),
            weight=(cfg.hidden_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"block_{layer_idx}_cross_attn_q",
            result=cross_q_result))
        
        # K, V projections from text
        kv_result = linear(
            input=(text_len, cfg.text_dim),
            weight=(2 * cfg.hidden_size, cfg.text_dim),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"block_{layer_idx}_cross_attn_kv",
            result=kv_result))
        
        # Cross-attention computation
        cross_attn_result = scaled_dot_product_attention(
            query=(1, cfg.num_attention_heads, seq_len, head_dim),
            key=(1, cfg.num_attention_heads, text_len, head_dim),
            value=(1, cfg.num_attention_heads, text_len, head_dim),
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"block_{layer_idx}_cross_attn_compute",
            result=cross_attn_result))
        
        # Output projection
        cross_o_result = linear(
            input=(m, cfg.hidden_size),
            weight=(cfg.hidden_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"block_{layer_idx}_cross_attn_o",
            result=cross_o_result))
        
        # === FFN with Modulation ===
        # Pre-FFN LayerNorm
        ln2_result = layer_norm(
            input=(1, seq_len, cfg.hidden_size),
            normalized_shape=(cfg.hidden_size,),
            elementwise_affine=False,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"block_{layer_idx}_norm2",
            result=ln2_result))
        
        # FFN input (gated)
        ffn_in_result = linear(
            input=(m, cfg.hidden_size),
            weight=(2 * cfg.intermediate_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"block_{layer_idx}_ffn_in",
            result=ffn_in_result))
        
        # GeLU activation
        gelu_result = gelu(
            input=(1, seq_len, cfg.intermediate_size),
            approximate="tanh",
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"block_{layer_idx}_ffn_act",
            result=gelu_result))
        
        # FFN output
        ffn_out_result = linear(
            input=(m, cfg.intermediate_size),
            weight=(cfg.hidden_size, cfg.intermediate_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"block_{layer_idx}_ffn_out",
            result=ffn_out_result))
        
        return layers
    
    def _build_unpatchify_layer(self, dtype_size: int) -> LayerConfig:
        """Build unpatchify output projection using linear kernel."""
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
        
        # Using linear kernel
        linear_result = linear(
            input=(seq_len, cfg.hidden_size),
            weight=(out_dim, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        
        return kernel_result_to_layer(
            name="unpatchify",
            result=linear_result)


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
        """Build 3D causal convolution layer using conv3d kernel."""
        kt, kh, kw = kernel_size
        st, sh, sw = stride
        input_t, input_h, input_w = input_dims
        
        # Use conv3d kernel API
        conv_result = conv3d(
            input=(1, in_channels, input_t, input_h, input_w),
            weight=(out_channels, in_channels, kt, kh, kw),
            bias=None,
            stride=(st, sh, sw),
            padding=(0 if self.config.use_causal_conv else 1, 1, 1),
            dtype=self.config.dtype
        )
        
        return kernel_result_to_layer(
            name=name,
            result=conv_result)
    
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
        # NOTE: Manual calculation for GroupNorm (non-kernel normalization layer)
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
        # NOTE: Manual calculation for GroupNorm (non-kernel normalization layer)
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
