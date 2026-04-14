"""Example: Building models with the new Kernel API.

This example demonstrates how to use the torch-like functional kernel API
to build and analyze model performance.
"""

import sys
sys.path.insert(0, '..')

from llm_perf.kernels import (
    # Functional API (torch-like)
    linear,
    scaled_dot_product_attention,
    rms_norm,
    # Layer builders
    transformer_block,
    summarize_block,
    norm_layer,
)


def example_1_basic_linear():
    """Example 1: Basic linear operation."""
    print("=" * 60)
    print("Example 1: Basic Linear Operation")
    print("=" * 60)
    
    # Similar to torch.nn.functional.linear
    result = linear(
        input=(4096, 5120),      # (batch*seq, in_features)
        weight=(13824, 5120),    # (out_features, in_features)
        bias=(13824,),           # (out_features,)
        dtype="fp16"
    )
    
    print(f"Input shape: (4096, 5120)")
    print(f"Weight shape: (13824, 5120)")
    print(f"Output shape: {result.output}")
    print(f"FLOPs: {result.flops / 1e12:.3f}T")
    print(f"Memory accessed: {result.bytes_accessed / 1024 / 1024:.2f}MB")
    print(f"Arithmetic intensity: {result.arithmetic_intensity:.2f}")
    print()


def example_2_attention():
    """Example 2: Self-attention."""
    print("=" * 60)
    print("Example 2: Self-Attention")
    print("=" * 60)
    
    batch, heads, seq, dim = 1, 32, 4096, 128
    
    # Similar to torch.nn.functional.scaled_dot_product_attention
    result = scaled_dot_product_attention(
        query=(batch, heads, seq, dim),
        key=(batch, heads, seq, dim),
        value=(batch, heads, seq, dim),
        is_causal=True,
        dtype="fp16"
    )
    
    print(f"Shape: [{batch}, {heads}, {seq}, {dim}]")
    print(f"FLOPs: {result.flops / 1e12:.3f}T")
    print(f"Memory: {result.bytes_accessed / 1024 / 1024:.2f}MB")
    print()


def example_3_transformer_block():
    """Example 3: Full transformer block."""
    print("=" * 60)
    print("Example 3: Transformer Block (LLaMA-style)")
    print("=" * 60)
    
    # Build a transformer block using layer builders
    layers = transformer_block(
        batch_size=1,
        seq_len=4096,
        hidden_size=4096,
        num_heads=32,
        intermediate_size=11008,
        norm_type="rmsnorm",
        gated_ffn=False,
        dtype="fp16"
    )
    
    summary = summarize_block(layers)
    
    print(f"Configuration: 4096 hidden, 32 heads, 11008 FFN")
    print(f"Number of sub-layers: {summary['num_layers']}")
    print(f"Total parameters: {summary['total_params_mb']:.2f}MB")
    print(f"Total FLOPs: {summary['total_flops_g']:.2f}G ({summary['total_flops_g']/1000:.2f}T)")
    print(f"Arithmetic intensity: {summary['arithmetic_intensity']:.2f}")
    print()


def example_4_wan2_1_block():
    """Example 4: Wan2.1 DiT block."""
    print("=" * 60)
    print("Example 4: Wan2.1 DiT Block")
    print("=" * 60)
    
    # Wan2.1 uses cross-attention and gated FFN
    layers = transformer_block(
        batch_size=1,
        seq_len=5120,  # 20*40*8 patches for 81x720x1280 video
        hidden_size=5120,
        num_heads=40,
        intermediate_size=13824,
        kv_seq_len=512,  # Text conditioning
        cross_attention=True,
        norm_type="layernorm",
        gated_ffn=True,
        dtype="fp16"
    )
    
    summary = summarize_block(layers)
    
    print(f"Configuration: 5120 hidden, 40 heads, 13824 FFN, cross-attn")
    print(f"Number of sub-layers: {summary['num_layers']}")
    print(f"Total parameters: {summary['total_params_mb']:.2f}MB")
    print(f"Total FLOPs: {summary['total_flops_g']:.2f}G ({summary['total_flops_g']/1000:.2f}T)")
    print()


def example_5_compare_norms():
    """Example 5: Compare different normalization types."""
    print("=" * 60)
    print("Example 5: Compare LayerNorm vs RMSNorm")
    print("=" * 60)
    
    shape = (1, 4096, 5120)
    
    # LayerNorm
    ln_layer, ln_result = norm_layer(
        1, 4096, 5120,
        norm_type="layernorm",
        elementwise_affine=True,
        dtype="fp16",
        name="layernorm"
    )
    
    # RMSNorm
    rms_layer, rms_result = norm_layer(
        1, 4096, 5120,
        norm_type="rmsnorm",
        elementwise_affine=True,
        dtype="fp16",
        name="rmsnorm"
    )
    
    print(f"Input shape: {shape}")
    print()
    print("LayerNorm:")
    print(f"  Params: {ln_layer.params_count / 1024 / 1024:.2f}MB")
    print(f"  FLOPs: {ln_result.flops / 1e6:.2f}M")
    print()
    print("RMSNorm:")
    print(f"  Params: {rms_layer.params_count / 1024 / 1024:.2f}MB")
    print(f"  FLOPs: {rms_result.flops / 1e6:.2f}M")
    print()


def example_6_full_model_estimate():
    """Example 6: Estimate full model performance."""
    print("=" * 60)
    print("Example 6: Full Model Estimate (LLaMA-7B)")
    print("=" * 60)
    
    # LLaMA-7B config
    num_layers = 32
    hidden_size = 4096
    num_heads = 32
    intermediate_size = 11008
    seq_len = 4096
    
    # Build one transformer block
    block_layers = transformer_block(
        batch_size=1,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        norm_type="rmsnorm",
        gated_ffn=False,
        dtype="fp16"
    )
    
    block_summary = summarize_block(block_layers)
    
    # Scale to full model
    total_params = block_summary['total_params'] * num_layers
    total_flops = block_summary['total_flops'] * num_layers
    
    print(f"Configuration:")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden: {hidden_size}")
    print(f"  Heads: {num_heads}")
    print(f"  FFN: {intermediate_size}")
    print()
    print(f"Per-block:")
    print(f"  Params: {block_summary['total_params_mb']:.2f}MB")
    print(f"  FLOPs: {block_summary['total_flops_g']:.2f}G")
    print()
    print(f"Full model:")
    print(f"  Total params: {total_params / 1e9:.2f}B")
    print(f"  Total FLOPs per forward: {total_flops / 1e12:.2f}T")
    print()


if __name__ == "__main__":
    example_1_basic_linear()
    example_2_attention()
    example_3_transformer_block()
    example_4_wan2_1_block()
    example_5_compare_norms()
    example_6_full_model_estimate()
