"""HunyuanVideo complete DiT model.

Includes:
- ShardedHYVideoDiT: Full diffusion transformer model
"""

from typing import Optional, Tuple

from llm_perf.kernels.op import RMSNormOp
from llm_perf.modeling.base.dit_blocks import (
    ShardedMMDoubleStreamBlock,
    ShardedMMSingleStreamBlock,
)
from llm_perf.modeling.base.dit_layers import (
    ShardedPatchEmbed3D,
    ShardedTimestepEmbedder,
)
from llm_perf.modeling.module import ShardedModule
from llm_perf.modeling.tensor import ShardedParameter, ShardedTensor


class ShardedHYVideoDiT(ShardedModule):
    """Complete HunyuanVideo Diffusion Transformer model.

    Architecture:
    1. Input processing: 3D patch embedding for video, linear projection for text
    2. Timestep embedding: sinusoidal + MLP
    3. Double-stream blocks: 20 blocks for image-text cross-attention
    4. Merge layer: concatenates image and text streams
    5. Single-stream blocks: 40 blocks for joint processing
    6. Final layer: normalization + linear projection

    Args:
        hidden_size: Hidden dimension (3072 for HunyuanVideo)
        heads_num: Number of attention heads (24)
        head_dim: Head dimension (128)
        double_blocks_depth: Number of double-stream blocks (20)
        single_blocks_depth: Number of single-stream blocks (40)
        mlp_width_ratio: MLP width ratio (4.0)
        in_channels: Input channels (16 for VAE latent)
        out_channels: Output channels (16)
        text_states_dim: Text embedding dimension (4096)
        patch_size: Patch size tuple (temporal, height, width)
        qk_norm: Whether to use QK normalization
        dtype: Data type
    """

    _submodule_name = "hyvideo_dit"
    supported_workloads = ["inference"]

    def __init__(
        self,
        hidden_size: int = 3072,
        heads_num: int = 24,
        head_dim: int = 128,
        double_blocks_depth: int = 20,
        single_blocks_depth: int = 40,
        mlp_width_ratio: float = 4.0,
        in_channels: int = 16,
        out_channels: int = 16,
        text_states_dim: int = 4096,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        qk_norm: bool = True,
        dtype: str = "bf16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.head_dim = head_dim
        self.double_blocks_depth = double_blocks_depth
        self.single_blocks_depth = single_blocks_depth
        self.mlp_width_ratio = mlp_width_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_states_dim = text_states_dim
        self.patch_size = patch_size
        self.qk_norm = qk_norm
        self.dtype = dtype

        self.img_in = ShardedPatchEmbed3D(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            dtype=dtype,
        )

        self.txt_in_weight = ShardedParameter(
            shape=(text_states_dim, hidden_size),
            shardable={0: "tp"},
            dtype=dtype,
            name="txt_in_weight",
        )

        self.time_in = ShardedTimestepEmbedder(hidden_size=hidden_size, dtype=dtype)

        self.vector_in_weight = ShardedParameter(
            shape=(hidden_size, hidden_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="vector_in_weight",
        )

        self.double_blocks = [
            ShardedMMDoubleStreamBlock(
                hidden_size=hidden_size,
                heads_num=heads_num,
                head_dim=head_dim,
                mlp_width_ratio=mlp_width_ratio,
                qk_norm=qk_norm,
                dtype=dtype,
            )
            for _ in range(double_blocks_depth)
        ]

        self.merge_weight = ShardedParameter(
            shape=(hidden_size, hidden_size),
            shardable={0: "tp"},
            dtype=dtype,
            name="merge_weight",
        )

        self.single_blocks = [
            ShardedMMSingleStreamBlock(
                hidden_size=hidden_size,
                heads_num=heads_num,
                head_dim=head_dim,
                mlp_width_ratio=mlp_width_ratio,
                qk_norm=qk_norm,
                dtype=dtype,
            )
            for _ in range(single_blocks_depth)
        ]

        self.final_layer_norm = ShardedParameter(
            shape=(hidden_size,),
            shardable={},
            dtype=dtype,
            name="final_layer_norm",
        )

        pt, ph, pw = patch_size
        self.final_layer_linear = ShardedParameter(
            shape=(hidden_size, pt * ph * pw * out_channels),
            shardable={1: "tp"},
            dtype=dtype,
            name="final_layer_linear",
        )

    def _rms_norm(
        self, input_tensor: ShardedTensor, weight: ShardedParameter
    ) -> ShardedTensor:
        """RMS normalization helper."""
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
        return output

    def forward(
        self,
        video: ShardedTensor,
        timestep: ShardedTensor,
        text_states: ShardedTensor,
        guidance_vec: Optional[ShardedTensor] = None,
        freqs_cis: Optional[ShardedTensor] = None,
    ) -> ShardedTensor:
        """Complete DiT forward pass.

        Args:
            video: Video latent tensor (batch, channels, frames, height, width)
            timestep: Timestep tensor (batch,)
            text_states: Text embedding (batch, txt_seq, text_dim)
            guidance_vec: Guidance vector (batch, hidden) (optional)
            freqs_cis: ROPE frequencies (optional)

        Returns:
            output: Video latent output (batch, channels, frames, height, width)
        """
        batch = video.shape[0] if len(video.shape) >= 1 else 1

        img = self.img_in(video)

        txt = text_states @ self.txt_in_weight

        vec = self.time_in(timestep)
        if guidance_vec is not None:
            vec = vec + guidance_vec @ self.vector_in_weight

        txt_seq_len = txt.shape[1] if len(txt.shape) >= 2 else 256

        for block in self.double_blocks:
            img, txt = block(img, txt, vec, freqs_cis)

        img_merge = img @ self.merge_weight
        merged = self._concat_tokens(img_merge, txt, batch)

        for block in self.single_blocks:
            merged = block(merged, vec, freqs_cis, txt_seq_len)

        img_out = self._slice_tokens(merged, batch, 0, merged.shape[1] - txt_seq_len)

        img_out = self._rms_norm(img_out, self.final_layer_norm)
        img_out = img_out @ self.final_layer_linear

        pt, ph, pw = self.patch_size
        out_channels = self.out_channels

        frames = video.shape[2] if len(video.shape) >= 3 else 1
        height = video.shape[3] if len(video.shape) >= 4 else 1
        width = video.shape[4] if len(video.shape) >= 5 else 1

        output = ShardedTensor(
            shape=(batch, out_channels, frames // pt, height // ph, width // pw),
            shardable={1: "tp"},
            dtype=self.dtype,
            name="output",
        )
        output._op_history = img_out._op_history
        output._is_view = True

        self._activations["img"] = img
        self._activations["txt"] = txt
        self._activations["vec"] = vec
        self._activations["merged"] = merged
        self._activations["img_out"] = img_out
        self._activations["output"] = output

        return output

    def _concat_tokens(
        self, img_tensor: ShardedTensor, txt_tensor: ShardedTensor, batch: int
    ) -> ShardedTensor:
        """Concatenate image and text tokens along sequence dimension."""
        img_seq = img_tensor.shape[1] if len(img_tensor.shape) >= 2 else 1
        txt_seq = txt_tensor.shape[1] if len(txt_tensor.shape) >= 2 else 1
        total_seq = img_seq + txt_seq

        merged = ShardedTensor(
            shape=(batch, total_seq, self.hidden_size),
            shardable=img_tensor.shardable,
            dtype=img_tensor.dtype,
            name="merged",
        )

        merged._op_history = img_tensor._op_history
        merged._is_view = True

        return merged

    def _slice_tokens(
        self, tensor: ShardedTensor, batch: int, start: int, end: int
    ) -> ShardedTensor:
        """Slice tensor along sequence dimension."""
        seq_len = end - start

        sliced = ShardedTensor(
            shape=(batch, seq_len, self.hidden_size),
            shardable=tensor.shardable,
            dtype=tensor.dtype,
            name="img_out",
        )

        sliced._op_history = tensor._op_history
        sliced._is_view = True

        return sliced