"""Tests for Vision Transformer (ViT) Encoder.

ShardedViTEncoder is an independent module for multimodal models.
"""

import pytest
from llm_perf.modeling.encoder import (
    ShardedViTEncoder,
    ShardedViTBlock,
    ShardedPatchEmbedding,
    ShardedPositionalEmbedding,
    ShardedSpatialMerge,
    ShardedOutputProjection,
)
from llm_perf.modeling import ShardedTensor, ParallelContext


class TestShardedViTBlock:
    """Test ShardedViTBlock."""

    def test_vit_block_creation(self):
        """Test creating ViT block."""
        block = ShardedViTBlock(
            hidden_size=1152,
            num_heads=16,
            intermediate_size=4304,
        )

        assert block.hidden_size == 1152
        assert block.num_heads == 16
        assert block.intermediate_size == 4304

    def test_vit_block_submodules(self):
        """Test block has correct submodules."""
        block = ShardedViTBlock(
            hidden_size=1152,
            num_heads=16,
            intermediate_size=4304,
        )

        assert "attention" in block._submodules
        assert "mlp" in block._submodules
        assert "norm1_weight" in block._weights
        assert "norm2_weight" in block._weights

    def test_vit_block_forward(self):
        """Test block forward."""
        block = ShardedViTBlock(
            hidden_size=1152,
            num_heads=16,
            intermediate_size=4304,
        )

        hidden = ShardedTensor(shape=(1, 256, 1152))
        output = block(hidden)

        assert output.shape == (1, 256, 1152)
        assert "attn_out" in block._activations
        assert "mlp_out" in block._activations


class TestShardedPatchEmbedding:
    """Test ShardedPatchEmbedding."""

    def test_patch_embedding_creation(self):
        """Test creating patch embedding."""
        patch_embed = ShardedPatchEmbedding(
            in_channels=3,
            hidden_size=1152,
            patch_size=16,
        )

        assert patch_embed.in_channels == 3
        assert patch_embed.hidden_size == 1152
        assert patch_embed.patch_size == 16
        assert patch_embed.weight.shape == (1152, 3, 16, 16)

    def test_patch_embedding_params(self):
        """Test patch embedding params count."""
        patch_embed = ShardedPatchEmbedding(
            in_channels=3,
            hidden_size=1152,
            patch_size=16,
        )

        params = patch_embed.params_count()
        expected = 1152 * 3 * 16 * 16
        assert params == expected


class TestShardedPositionalEmbedding:
    """Test ShardedPositionalEmbedding."""

    def test_pos_embedding_creation(self):
        """Test creating positional embedding."""
        pos_embed = ShardedPositionalEmbedding(
            hidden_size=1152,
            num_positions=2304,
        )

        assert pos_embed.hidden_size == 1152
        assert pos_embed.num_positions == 2304
        assert pos_embed.weight.shape == (2304, 1152)

    def test_pos_embedding_params(self):
        """Test positional embedding params count."""
        pos_embed = ShardedPositionalEmbedding(
            hidden_size=1152,
            num_positions=2304,
        )

        params = pos_embed.params_count()
        expected = 2304 * 1152
        assert params == expected


class TestShardedSpatialMerge:
    """Test ShardedSpatialMerge."""

    def test_spatial_merge_creation(self):
        """Test creating spatial merge."""
        spatial_merge = ShardedSpatialMerge(
            hidden_size=1152,
            spatial_merge_size=2,
        )

        assert spatial_merge.hidden_size == 1152
        assert spatial_merge.spatial_merge_size == 2
        assert spatial_merge.weight.shape == (1152 * 4, 1152)


class TestShardedOutputProjection:
    """Test ShardedOutputProjection."""

    def test_output_proj_creation(self):
        """Test creating output projection."""
        output_proj = ShardedOutputProjection(
            hidden_size=1152,
            out_hidden_size=2048,
        )

        assert output_proj.hidden_size == 1152
        assert output_proj.out_hidden_size == 2048
        assert output_proj.weight.shape == (1152, 2048)

    def test_output_proj_params(self):
        """Test output projection params count."""
        output_proj = ShardedOutputProjection(
            hidden_size=1152,
            out_hidden_size=2048,
        )

        params = output_proj.params_count()
        expected = 1152 * 2048
        assert params == expected


class TestShardedViTEncoder:
    """Test ShardedViTEncoder."""

    def test_vit_encoder_creation(self):
        """Test creating ViT encoder."""
        encoder = ShardedViTEncoder(
            depth=27,
            hidden_size=1152,
            num_heads=16,
            in_channels=3,
            patch_size=16,
            intermediate_size=4304,
            num_position_embeddings=2304,
            out_hidden_size=2048,
            spatial_merge_size=2,
        )

        assert encoder.depth == 27
        assert encoder.hidden_size == 1152
        assert encoder.num_heads == 16
        assert encoder.out_hidden_size == 2048
        assert len(encoder.layers) == 27

    def test_vit_encoder_params(self):
        """Test ViT encoder params count.

        Components:
        - PatchEmbed: 1152 * 3 * 16 * 16 = 884,736
        - PosEmbed: 2304 * 1152 = 2,660,352
        - 27 ViTBlocks: each block has attention + mlp + norms
        - SpatialMerge: 1152 * 4 * 1152 = 5,308,416
        - OutputProj: 1152 * 2048 = 2,359,296
        """
        encoder = ShardedViTEncoder(
            depth=27,
            hidden_size=1152,
            num_heads=16,
            in_channels=3,
            patch_size=16,
            intermediate_size=4304,
            num_position_embeddings=2304,
            out_hidden_size=2048,
            spatial_merge_size=2,
        )

        params = encoder.params_count()
        assert params > 0

        patch_embed_params = 1152 * 3 * 16 * 16
        pos_embed_params = 2304 * 1152
        spatial_merge_params = 1152 * 4 * 1152
        output_proj_params = 1152 * 2048

        base_params = patch_embed_params + pos_embed_params + spatial_merge_params + output_proj_params
        assert params >= base_params

    def test_vit_encoder_flops(self):
        """Test ViT encoder FLOPs estimation."""
        encoder = ShardedViTEncoder(
            depth=27,
            hidden_size=1152,
            num_heads=16,
            intermediate_size=4304,
            out_hidden_size=2048,
        )

        image = ShardedTensor(shape=(1, 3, 224, 224))
        output = encoder(image)

        assert output.shape[0] == 1
        assert output.shape[-1] == 2048
        assert len(output._op_history) > 0

    def test_vit_encoder_forward(self):
        """Test ViT encoder forward."""
        encoder = ShardedViTEncoder(
            depth=4,
            hidden_size=1152,
            num_heads=16,
            in_channels=3,
            patch_size=16,
            intermediate_size=4304,
            num_position_embeddings=2304,
            out_hidden_size=2048,
            spatial_merge_size=2,
        )

        image = ShardedTensor(shape=(1, 3, 224, 224))
        output = encoder(image)

        assert output.shape[0] == 1
        assert output.shape[-1] == 2048
        assert "patch_embed_output" in encoder._activations
        assert "layer_0_output" in encoder._activations
        assert "spatial_merge_output" in encoder._activations
        assert "final_output" in encoder._activations

    def test_vit_encoder_module_instance(self):
        """Test ViT encoder with ParallelContext."""
        encoder = ShardedViTEncoder(
            depth=4,
            hidden_size=1152,
            num_heads=16,
            in_channels=3,
            patch_size=16,
            intermediate_size=4304,
            out_hidden_size=2048,
        )

        ctx = ParallelContext(tp_degree=8)
        instance = encoder.bind(ctx)

        assert instance.params_count_logical == encoder.params_count()
        assert instance.params_count_physical <= instance.params_count_logical

    def test_vit_encoder_video_forward(self):
        """Test ViT encoder with video input."""
        encoder = ShardedViTEncoder(
            depth=4,
            hidden_size=1152,
            num_heads=16,
            in_channels=3,
            patch_size=16,
            intermediate_size=4304,
            num_position_embeddings=2304,
            out_hidden_size=2048,
            spatial_merge_size=2,
        )

        video = ShardedTensor(shape=(1, 8, 3, 224, 224))
        output = encoder.forward_video(video)

        assert output.shape[0] == 1
        assert output.shape[-1] == 2048


class TestViTEncoderIntegration:
    """Test ViT encoder integration scenarios."""

    def test_vit_encoder_with_different_backbones(self):
        """Test ViT encoder can align to different backbone hidden sizes."""
        encoder_2048 = ShardedViTEncoder(
            depth=27,
            hidden_size=1152,
            out_hidden_size=2048,
        )

        encoder_4096 = ShardedViTEncoder(
            depth=27,
            hidden_size=1152,
            out_hidden_size=4096,
        )

        assert encoder_2048.out_hidden_size == 2048
        assert encoder_4096.out_hidden_size == 4096

        output_proj_2048_params = encoder_2048.output_proj.params_count()
        output_proj_4096_params = encoder_4096.output_proj.params_count()

        assert output_proj_4096_params == output_proj_2048_params * 2

    def test_vit_encoder_params_breakdown(self):
        """Test ViT encoder params breakdown."""
        encoder = ShardedViTEncoder(
            depth=27,
            hidden_size=1152,
            num_heads=16,
            intermediate_size=4304,
            out_hidden_size=2048,
        )

        breakdown = encoder.params_count_breakdown()

        patch_embed_params = sum(v for k, v in breakdown.items() if "patch_embed" in k)
        assert patch_embed_params > 0

        pos_embed_params = sum(v for k, v in breakdown.items() if "pos_embed" in k)
        assert pos_embed_params > 0

        layer_params = sum(v for k, v in breakdown.items() if "layers." in k)
        assert layer_params > 0

        output_proj_params = sum(v for k, v in breakdown.items() if "output_proj" in k)
        assert output_proj_params > 0

    def test_vit_encoder_layers_count(self):
        """Test correct number of layers."""
        encoder = ShardedViTEncoder(depth=27)
        assert len(encoder.layers) == 27

        encoder_small = ShardedViTEncoder(depth=4)
        assert len(encoder_small.layers) == 4


class TestViTEncoderFLOPs:
    """Test ViT encoder FLOPs calculations."""

    def test_vit_encoder_forward_flops(self):
        """Test forward FLOPs estimation."""
        encoder = ShardedViTEncoder(
            depth=4,
            hidden_size=512,
            num_heads=8,
            patch_size=16,
            intermediate_size=1024,
            out_hidden_size=512,
        )

        image = ShardedTensor(shape=(1, 3, 64, 64))
        output = encoder(image)

        flops = encoder.flops_forward()
        assert flops > 0

    def test_vit_encoder_backward_flops(self):
        """Test backward FLOPs estimation."""
        encoder = ShardedViTEncoder(
            depth=4,
            hidden_size=512,
            num_heads=8,
            patch_size=16,
            intermediate_size=1024,
            out_hidden_size=512,
        )

        image = ShardedTensor(shape=(1, 3, 64, 64))
        output = encoder(image)

        backward_flops = encoder.flops_backward()
        forward_flops = encoder.flops_forward()

        assert backward_flops == forward_flops * 2