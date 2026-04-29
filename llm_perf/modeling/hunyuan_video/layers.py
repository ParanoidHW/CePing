"""HunyuanVideo base layers.

Includes:
- ShardedModulateDiT: Modulation layer for DiT blocks
- ShardedPatchEmbed3D: 3D patch embedding for video input
- ShardedTimestepEmbedder: Timestep embedding for diffusion
"""

from typing import Tuple

from llm_perf.kernels.op import Conv3dOp
from llm_perf.modeling.module import ShardedModule
from llm_perf.modeling.tensor import ShardedParameter, ShardedTensor


class ShardedModulateDiT(ShardedModule):
    """Modulation layer for HunyuanVideo DiT blocks.

    Outputs 6 modulation parameters:
    - shift1, scale1, gate1: for self-attention
    - shift2, scale2, gate2: for FFN

    Args:
        hidden_size: Hidden dimension
        dtype: Data type
    """

    _submodule_name = "modulate_dit"

    def __init__(self, hidden_size: int, dtype: str = "bf16"):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype

        self.modulation_weight = ShardedParameter(
            shape=(hidden_size, hidden_size * 6),
            shardable={1: "tp"},
            dtype=dtype,
            name="modulation_weight",
        )

    def forward(self, vec: ShardedTensor) -> Tuple[ShardedTensor, ...]:
        """Modulation forward.

        Args:
            vec: Conditioning vector (batch, hidden_size)

        Returns:
            Tuple of 6 modulation params: shift1, scale1, gate1, shift2, scale2, gate2
        """
        modulation = vec @ self.modulation_weight

        batch = vec.shape[0] if len(vec.shape) >= 1 else 1
        chunk_size = self.hidden_size

        shift1 = ShardedTensor(
            shape=(batch, chunk_size),
            shardable=modulation.shardable,
            dtype=modulation.dtype,
            name="shift1",
        )
        scale1 = ShardedTensor(
            shape=(batch, chunk_size),
            shardable=modulation.shardable,
            dtype=modulation.dtype,
            name="scale1",
        )
        gate1 = ShardedTensor(
            shape=(batch, chunk_size),
            shardable=modulation.shardable,
            dtype=modulation.dtype,
            name="gate1",
        )
        shift2 = ShardedTensor(
            shape=(batch, chunk_size),
            shardable=modulation.shardable,
            dtype=modulation.dtype,
            name="shift2",
        )
        scale2 = ShardedTensor(
            shape=(batch, chunk_size),
            shardable=modulation.shardable,
            dtype=modulation.dtype,
            name="scale2",
        )
        gate2 = ShardedTensor(
            shape=(batch, chunk_size),
            shardable=modulation.shardable,
            dtype=modulation.dtype,
            name="gate2",
        )

        shift1._op_history = modulation._op_history
        scale1._op_history = modulation._op_history
        gate1._op_history = modulation._op_history
        shift2._op_history = modulation._op_history
        scale2._op_history = modulation._op_history
        gate2._op_history = modulation._op_history

        self._activations["modulation"] = modulation
        self._activations["shift1"] = shift1
        self._activations["scale1"] = scale1
        self._activations["gate1"] = gate1
        self._activations["shift2"] = shift2
        self._activations["scale2"] = scale2
        self._activations["gate2"] = gate2

        return shift1, scale1, gate1, shift2, scale2, gate2


class ShardedPatchEmbed3D(ShardedModule):
    """3D Patch Embedding layer for video input.

    Converts video tensor to latent tokens via 3D convolution.

    Args:
        in_channels: Input video channels (3 for RGB)
        hidden_size: Hidden dimension
        patch_size: Patch size tuple (temporal, height, width)
        dtype: Data type
    """

    _submodule_name = "patch_embed_3d"

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 3072,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        dtype: str = "bf16",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.dtype = dtype

        self.proj_weight = ShardedParameter(
            shape=(hidden_size, in_channels, *patch_size),
            shardable={0: "tp"},
            dtype=dtype,
            name="proj_weight",
        )

    def forward(self, video: ShardedTensor) -> ShardedTensor:
        """Patch embedding forward.

        Args:
            video: Video tensor (batch, channels, frames, height, width)

        Returns:
            tokens: Latent tokens (batch, seq_len, hidden_size)
        """
        batch = video.shape[0] if len(video.shape) >= 1 else 1
        frames = video.shape[2] if len(video.shape) >= 3 else 1
        height = video.shape[3] if len(video.shape) >= 4 else 1
        width = video.shape[4] if len(video.shape) >= 5 else 1

        pt, ph, pw = self.patch_size
        out_frames = frames // pt
        out_height = height // ph
        out_width = width // pw
        seq_len = out_frames * out_height * out_width

        conv_out = ShardedTensor(
            shape=(batch, self.hidden_size, out_frames, out_height, out_width),
            shardable={1: "tp"},
            dtype=self.dtype,
            name="conv_output",
        )

        conv_out._op_history = video._op_history + [
            Conv3dOp(
                dtype=self.dtype,
                input=video,
                weight=self.proj_weight,
                output=conv_out,
                stride=self.patch_size,
                padding=(0, 0, 0),
            )
        ]

        tokens = ShardedTensor(
            shape=(batch, seq_len, self.hidden_size),
            shardable={2: "tp"},
            dtype=self.dtype,
            name="tokens",
        )

        tokens._op_history = conv_out._op_history
        tokens._is_view = True

        self._activations["conv_out"] = conv_out
        self._activations["tokens"] = tokens

        return tokens


class ShardedTimestepEmbedder(ShardedModule):
    """Timestep Embedder for diffusion models.

    Converts timestep scalar to hidden embedding via sinusoidal + MLP.

    Args:
        hidden_size: Hidden dimension
        frequency_embedding_size: Frequency embedding dimension
        dtype: Data type
    """

    _submodule_name = "timestep_embedder"

    def __init__(
        self,
        hidden_size: int = 3072,
        frequency_embedding_size: int = 256,
        dtype: str = "bf16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

        self.mlp_fc1_weight = ShardedParameter(
            shape=(frequency_embedding_size * 4, hidden_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="mlp_fc1_weight",
        )
        self.mlp_fc2_weight = ShardedParameter(
            shape=(hidden_size, hidden_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="mlp_fc2_weight",
        )

    def forward(self, timestep: ShardedTensor) -> ShardedTensor:
        """Timestep embedding forward.

        Args:
            timestep: Timestep tensor (batch,) or (batch, 1)

        Returns:
            embed: Hidden embedding (batch, hidden_size)
        """
        batch = timestep.shape[0] if len(timestep.shape) >= 1 else 1

        freq_embed = ShardedTensor(
            shape=(batch, self.frequency_embedding_size * 4),
            shardable={},
            dtype=self.dtype,
            name="freq_embed",
        )

        freq_embed._op_history = timestep._op_history

        fc1_out = freq_embed @ self.mlp_fc1_weight

        fc1_activated = ShardedTensor(
            shape=(batch, self.hidden_size),
            shardable=fc1_out.shardable,
            dtype=self.dtype,
            name="fc1_activated",
        )
        fc1_activated._op_history = fc1_out._op_history

        hidden = fc1_activated @ self.mlp_fc2_weight

        self._activations["freq_embed"] = freq_embed
        self._activations["fc1_out"] = fc1_out
        self._activations["fc1_activated"] = fc1_activated
        self._activations["hidden"] = hidden

        return hidden
