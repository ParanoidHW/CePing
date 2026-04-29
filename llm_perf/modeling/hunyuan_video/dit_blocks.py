"""HunyuanVideo DiT blocks.

Includes:
- ShardedMMDoubleStreamBlock: Double stream block for image and text branches
- ShardedMMSingleStreamBlock: Single stream block for merged processing
"""

from typing import Optional, Tuple

from llm_perf.kernels.op import RMSNormOp
from llm_perf.modeling.hunyuan_video.layers import ShardedModulateDiT
from llm_perf.modeling.layers import flash_attention, silu
from llm_perf.modeling.module import ShardedModule
from llm_perf.modeling.tensor import ShardedParameter, ShardedTensor


class ShardedMMDoubleStreamBlock(ShardedModule):
    """Double stream DiT block for HunyuanVideo.

    Processes image and text branches independently, with cross-attention.

    Args:
        hidden_size: Hidden dimension (3072 for HunyuanVideo)
        heads_num: Number of attention heads (24)
        head_dim: Head dimension (128)
        mlp_width_ratio: MLP width ratio (4.0)
        qk_norm: Whether to use QK normalization
        dtype: Data type
    """

    _submodule_name = "double_stream_block"

    def __init__(
        self,
        hidden_size: int = 3072,
        heads_num: int = 24,
        head_dim: int = 128,
        mlp_width_ratio: float = 4.0,
        qk_norm: bool = True,
        dtype: str = "bf16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.head_dim = head_dim
        self.mlp_width_ratio = mlp_width_ratio
        self.qk_norm = qk_norm
        self.dtype = dtype

        intermediate_size = int(hidden_size * mlp_width_ratio)

        # Image branch weights
        self.img_attn_qkv = ShardedParameter(
            shape=(hidden_size, hidden_size * 3),
            shardable={1: "tp"},
            dtype=dtype,
            name="img_attn_qkv",
        )

        self.img_attn_q_norm = (
            ShardedParameter(
                shape=(head_dim,),
                shardable={},
                dtype=dtype,
                name="img_attn_q_norm",
            )
            if qk_norm
            else None
        )

        self.img_attn_k_norm = (
            ShardedParameter(
                shape=(head_dim,),
                shardable={},
                dtype=dtype,
                name="img_attn_k_norm",
            )
            if qk_norm
            else None
        )

        self.img_attn_proj = ShardedParameter(
            shape=(hidden_size, hidden_size),
            shardable={0: "tp"},
            dtype=dtype,
            name="img_attn_proj",
        )

        self.img_mlp_gate = ShardedParameter(
            shape=(hidden_size, intermediate_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="img_mlp_gate",
        )

        self.img_mlp_up = ShardedParameter(
            shape=(hidden_size, intermediate_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="img_mlp_up",
        )

        self.img_mlp_down = ShardedParameter(
            shape=(intermediate_size, hidden_size),
            shardable={0: "tp"},
            dtype=dtype,
            name="img_mlp_down",
        )

        # Text branch weights
        self.txt_attn_qkv = ShardedParameter(
            shape=(hidden_size, hidden_size * 3),
            shardable={1: "tp"},
            dtype=dtype,
            name="txt_attn_qkv",
        )

        self.txt_attn_q_norm = (
            ShardedParameter(
                shape=(head_dim,),
                shardable={},
                dtype=dtype,
                name="txt_attn_q_norm",
            )
            if qk_norm
            else None
        )

        self.txt_attn_k_norm = (
            ShardedParameter(
                shape=(head_dim,),
                shardable={},
                dtype=dtype,
                name="txt_attn_k_norm",
            )
            if qk_norm
            else None
        )

        self.txt_attn_proj = ShardedParameter(
            shape=(hidden_size, hidden_size),
            shardable={0: "tp"},
            dtype=dtype,
            name="txt_attn_proj",
        )

        self.txt_mlp_gate = ShardedParameter(
            shape=(hidden_size, intermediate_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="txt_mlp_gate",
        )

        self.txt_mlp_up = ShardedParameter(
            shape=(hidden_size, intermediate_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="txt_mlp_up",
        )

        self.txt_mlp_down = ShardedParameter(
            shape=(intermediate_size, hidden_size),
            shardable={0: "tp"},
            dtype=dtype,
            name="txt_mlp_down",
        )

        # Modulation layers
        self.img_mod = ShardedModulateDiT(hidden_size, dtype)
        self.txt_mod = ShardedModulateDiT(hidden_size, dtype)

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

    def _split_qkv(
        self, qkv: ShardedTensor, batch: int, seq: int
    ) -> Tuple[ShardedTensor, ShardedTensor, ShardedTensor]:
        """Split QKV tensor into Q, K, V."""
        q = ShardedTensor(
            shape=(batch, seq, self.hidden_size),
            shardable=qkv.shardable,
            dtype=qkv.dtype,
            name="q",
        )
        k = ShardedTensor(
            shape=(batch, seq, self.hidden_size),
            shardable=qkv.shardable,
            dtype=qkv.dtype,
            name="k",
        )
        v = ShardedTensor(
            shape=(batch, seq, self.hidden_size),
            shardable=qkv.shardable,
            dtype=qkv.dtype,
            name="v",
        )

        q._op_history = qkv._op_history
        k._op_history = qkv._op_history
        v._op_history = qkv._op_history
        q._is_view = True
        k._is_view = True
        v._is_view = True

        return q, k, v

    def _reshape_for_attention(
        self, tensor: ShardedTensor, batch: int, seq: int
    ) -> ShardedTensor:
        """Reshape tensor to (batch, heads, seq, head_dim) for attention."""
        reshaped = tensor.view(batch, seq, self.heads_num, self.head_dim)
        return reshaped.transpose(1, 2)

    def _broadcast_modulation(
        self,
        param: ShardedTensor,
        batch: int,
        seq: int,
        hidden_size: int,
    ) -> ShardedTensor:
        """Broadcast modulation parameter from (batch, hidden) to (batch, seq, hidden)."""
        broadcast_param = ShardedTensor(
            shape=(batch, seq, hidden_size),
            shardable=param.shardable,
            dtype=param.dtype,
            name=f"{param.name}_broadcast",
        )
        broadcast_param._op_history = param._op_history
        broadcast_param._is_view = True
        return broadcast_param

    def forward(
        self,
        img: ShardedTensor,
        txt: ShardedTensor,
        vec: ShardedTensor,
        freqs_cis: Optional[ShardedTensor] = None,
    ) -> Tuple[ShardedTensor, ShardedTensor]:
        """Double stream block forward.

        Args:
            img: Image latent tokens (batch, img_seq, hidden)
            txt: Text tokens (batch, txt_seq, hidden)
            vec: Modulation vector (batch, hidden)
            freqs_cis: ROPE frequencies (optional)

        Returns:
            img_out: Image output (batch, img_seq, hidden)
            txt_out: Text output (batch, txt_seq, hidden)
        """
        batch = img.shape[0] if len(img.shape) >= 1 else 1
        img_seq = img.shape[1] if len(img.shape) >= 2 else 1
        txt_seq = txt.shape[1] if len(txt.shape) >= 2 else 1

        # Get modulation parameters
        img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = (
            self.img_mod(vec)
        )
        txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = (
            self.txt_mod(vec)
        )

        # Broadcast modulation parameters to sequence dimension
        img_shift1_b = self._broadcast_modulation(img_shift1, batch, img_seq, self.hidden_size)
        img_scale1_b = self._broadcast_modulation(img_scale1, batch, img_seq, self.hidden_size)
        img_gate1_b = self._broadcast_modulation(img_gate1, batch, img_seq, self.hidden_size)
        img_shift2_b = self._broadcast_modulation(img_shift2, batch, img_seq, self.hidden_size)
        img_scale2_b = self._broadcast_modulation(img_scale2, batch, img_seq, self.hidden_size)
        img_gate2_b = self._broadcast_modulation(img_gate2, batch, img_seq, self.hidden_size)

        txt_shift1_b = self._broadcast_modulation(txt_shift1, batch, txt_seq, self.hidden_size)
        txt_scale1_b = self._broadcast_modulation(txt_scale1, batch, txt_seq, self.hidden_size)
        txt_gate1_b = self._broadcast_modulation(txt_gate1, batch, txt_seq, self.hidden_size)
        txt_shift2_b = self._broadcast_modulation(txt_shift2, batch, txt_seq, self.hidden_size)
        txt_scale2_b = self._broadcast_modulation(txt_scale2, batch, txt_seq, self.hidden_size)
        txt_gate2_b = self._broadcast_modulation(txt_gate2, batch, txt_seq, self.hidden_size)

        # Image branch: modulation + attention
        # Modulation formula: x_modulated = x + x * scale + shift (equivalent to x * (1 + scale) + shift)
        img_scaled = img * img_scale1_b
        img_modulated = img + img_scaled + img_shift1_b
        img_qkv = img_modulated @ self.img_attn_qkv
        img_q, img_k, img_v = self._split_qkv(img_qkv, batch, img_seq)

        # QK-Norm
        if self.qk_norm:
            img_q = self._rms_norm(img_q, self.img_attn_q_norm)
            img_k = self._rms_norm(img_k, self.img_attn_k_norm)

        # Reshape for attention
        img_q_4d = self._reshape_for_attention(img_q, batch, img_seq)
        img_k_4d = self._reshape_for_attention(img_k, batch, img_seq)
        img_v_4d = self._reshape_for_attention(img_v, batch, img_seq)

        # Text branch: modulation + attention
        txt_scaled = txt * txt_scale1_b
        txt_modulated = txt + txt_scaled + txt_shift1_b
        txt_qkv = txt_modulated @ self.txt_attn_qkv
        txt_q, txt_k, txt_v = self._split_qkv(txt_qkv, batch, txt_seq)

        # QK-Norm
        if self.qk_norm:
            txt_q = self._rms_norm(txt_q, self.txt_attn_q_norm)
            txt_k = self._rms_norm(txt_k, self.txt_attn_k_norm)

        # Reshape for attention
        txt_q_4d = self._reshape_for_attention(txt_q, batch, txt_seq)
        txt_k_4d = self._reshape_for_attention(txt_k, batch, txt_seq)
        txt_v_4d = self._reshape_for_attention(txt_v, batch, txt_seq)

        # Concatenate Q/K/V for joint attention
        # Note: In HunyuanVideo, image and text attend to each other
        # This is cross-modal attention
        q_combined = self._concat_seq(img_q_4d, txt_q_4d, batch)
        k_combined = self._concat_seq(img_k_4d, txt_k_4d, batch)
        v_combined = self._concat_seq(img_v_4d, txt_v_4d, batch)

        # Flash attention (non-causal for cross-modal)
        attn_out = flash_attention(q_combined, k_combined, v_combined, is_causal=False)

        # Split attention output
        img_attn_out_4d = self._slice_seq(attn_out, batch, 0, img_seq)
        txt_attn_out_4d = self._slice_seq(attn_out, batch, img_seq, img_seq + txt_seq)

        # Reshape back
        img_attn_out = img_attn_out_4d.transpose(1, 2).view(
            batch, img_seq, self.heads_num * self.head_dim
        )
        txt_attn_out = txt_attn_out_4d.transpose(1, 2).view(
            batch, txt_seq, self.heads_num * self.head_dim
        )

        # Image branch: projection + MLP + gate
        img_attn_out = img_attn_out @ self.img_attn_proj
        img_scaled_gate1 = img * img_gate1_b
        img = img + img_scaled_gate1 * img_attn_out

        img_scaled2 = img * img_scale2_b
        img_modulated2 = img + img_scaled2 + img_shift2_b
        img_mlp_gate_out = silu(img_modulated2 @ self.img_mlp_gate)
        img_mlp_up_out = img_modulated2 @ self.img_mlp_up
        img_mlp_intermediate = img_mlp_gate_out * img_mlp_up_out
        img_mlp_out = img_mlp_intermediate @ self.img_mlp_down
        img_scaled_gate2 = img * img_gate2_b
        img = img + img_scaled_gate2 * img_mlp_out

        # Text branch: projection + MLP + gate
        txt_attn_out = txt_attn_out @ self.txt_attn_proj
        txt_scaled_gate1 = txt * txt_gate1_b
        txt = txt + txt_scaled_gate1 * txt_attn_out

        txt_scaled2 = txt * txt_scale2_b
        txt_modulated2 = txt + txt_scaled2 + txt_shift2_b
        txt_mlp_gate_out = silu(txt_modulated2 @ self.txt_mlp_gate)
        txt_mlp_up_out = txt_modulated2 @ self.txt_mlp_up
        txt_mlp_intermediate = txt_mlp_gate_out * txt_mlp_up_out
        txt_mlp_out = txt_mlp_intermediate @ self.txt_mlp_down
        txt_scaled_gate2 = txt * txt_gate2_b
        txt = txt + txt_scaled_gate2 * txt_mlp_out

        # Track activations
        self._activations["img"] = img
        self._activations["txt"] = txt

        return img, txt

    def _concat_seq(
        self, img_tensor: ShardedTensor, txt_tensor: ShardedTensor, batch: int
    ) -> ShardedTensor:
        """Concatenate image and text tensors along sequence dimension."""
        img_seq = img_tensor.shape[2]
        txt_seq = txt_tensor.shape[2]
        total_seq = img_seq + txt_seq

        combined = ShardedTensor(
            shape=(batch, self.heads_num, total_seq, self.head_dim),
            shardable=img_tensor.shardable,
            dtype=img_tensor.dtype,
            name="combined",
        )

        combined._op_history = img_tensor._op_history
        combined._is_view = True

        return combined

    def _slice_seq(
        self,
        tensor: ShardedTensor,
        batch: int,
        start_seq: int,
        end_seq: int,
    ) -> ShardedTensor:
        """Slice tensor along sequence dimension."""
        heads_num = tensor.shape[1]
        head_dim = tensor.shape[3]
        sliced_seq = end_seq - start_seq

        sliced = ShardedTensor(
            shape=(batch, heads_num, sliced_seq, head_dim),
            shardable=tensor.shardable,
            dtype=tensor.dtype,
            name=f"{tensor.name}_slice",
        )

        sliced._op_history = tensor._op_history
        sliced._is_view = True

        return sliced


class ShardedMMSingleStreamBlock(ShardedModule):
    """Single stream DiT block for HunyuanVideo.

    Processes merged image+text tokens with parallel attention+MLP.

    Args:
        hidden_size: Hidden dimension (3072)
        heads_num: Number of attention heads (24)
        head_dim: Head dimension (128)
        mlp_width_ratio: MLP width ratio (4.0)
        qk_norm: Whether to use QK normalization
        dtype: Data type
    """

    _submodule_name = "single_stream_block"

    def __init__(
        self,
        hidden_size: int = 3072,
        heads_num: int = 24,
        head_dim: int = 128,
        mlp_width_ratio: float = 4.0,
        qk_norm: bool = True,
        dtype: str = "bf16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.head_dim = head_dim
        self.mlp_width_ratio = mlp_width_ratio
        self.qk_norm = qk_norm
        self.dtype = dtype

        intermediate_size = int(hidden_size * mlp_width_ratio)

        # Modulation layer
        self.modulation = ShardedModulateDiT(hidden_size, dtype)

        # Parallel structure: linear1 outputs qkv and mlp_in
        self.linear1 = ShardedParameter(
            shape=(hidden_size, hidden_size * 3 + intermediate_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="linear1",
        )

        # QK-Norm weights
        self.q_norm = (
            ShardedParameter(
                shape=(head_dim,),
                shardable={},
                dtype=dtype,
                name="q_norm",
            )
            if qk_norm
            else None
        )

        self.k_norm = (
            ShardedParameter(
                shape=(head_dim,),
                shardable={},
                dtype=dtype,
                name="k_norm",
            )
            if qk_norm
            else None
        )

        # linear2: attention output + MLP output
        self.linear2 = ShardedParameter(
            shape=(hidden_size + intermediate_size, hidden_size),
            shardable={0: "tp"},
            dtype=dtype,
            name="linear2",
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

    def _split_qkv_mlp(
        self,
        linear1_out: ShardedTensor,
        batch: int,
        seq: int,
        intermediate_size: int,
    ) -> Tuple[ShardedTensor, ShardedTensor, ShardedTensor, ShardedTensor]:
        """Split linear1 output into Q, K, V, and MLP input."""
        hidden_size = self.hidden_size

        qkv = ShardedTensor(
            shape=(batch, seq, hidden_size * 3),
            shardable=linear1_out.shardable,
            dtype=linear1_out.dtype,
            name="qkv",
        )
        mlp_in = ShardedTensor(
            shape=(batch, seq, intermediate_size),
            shardable=linear1_out.shardable,
            dtype=linear1_out.dtype,
            name="mlp_in",
        )

        qkv._op_history = linear1_out._op_history
        mlp_in._op_history = linear1_out._op_history
        qkv._is_view = True
        mlp_in._is_view = True

        q = ShardedTensor(
            shape=(batch, seq, hidden_size),
            shardable=qkv.shardable,
            dtype=qkv.dtype,
            name="q",
        )
        k = ShardedTensor(
            shape=(batch, seq, hidden_size),
            shardable=qkv.shardable,
            dtype=qkv.dtype,
            name="k",
        )
        v = ShardedTensor(
            shape=(batch, seq, hidden_size),
            shardable=qkv.shardable,
            dtype=qkv.dtype,
            name="v",
        )

        q._op_history = qkv._op_history
        k._op_history = qkv._op_history
        v._op_history = qkv._op_history
        q._is_view = True
        k._is_view = True
        v._is_view = True

        return q, k, v, mlp_in

    def _reshape_for_attention(
        self, tensor: ShardedTensor, batch: int, seq: int
    ) -> ShardedTensor:
        """Reshape tensor to (batch, heads, seq, head_dim) for attention."""
        reshaped = tensor.view(batch, seq, self.heads_num, self.head_dim)
        return reshaped.transpose(1, 2)

    def _broadcast_modulation(
        self,
        param: ShardedTensor,
        batch: int,
        seq: int,
        hidden_size: int,
    ) -> ShardedTensor:
        """Broadcast modulation parameter from (batch, hidden) to (batch, seq, hidden)."""
        broadcast_param = ShardedTensor(
            shape=(batch, seq, hidden_size),
            shardable=param.shardable,
            dtype=param.dtype,
            name=f"{param.name}_broadcast",
        )
        broadcast_param._op_history = param._op_history
        broadcast_param._is_view = True
        return broadcast_param

    def forward(
        self,
        x: ShardedTensor,
        vec: ShardedTensor,
        freqs_cis: Optional[ShardedTensor] = None,
        txt_seq_len: int = 256,
    ) -> ShardedTensor:
        """Single stream block forward.

        Args:
            x: Merged tokens (batch, img_seq+txt_seq, hidden)
            vec: Modulation vector (batch, hidden)
            freqs_cis: ROPE frequencies (optional)
            txt_seq_len: Text sequence length for separating image part

        Returns:
            x_out: Output (batch, img_seq+txt_seq, hidden)
        """
        batch = x.shape[0] if len(x.shape) >= 1 else 1
        total_seq = x.shape[1] if len(x.shape) >= 2 else 1

        intermediate_size = int(self.hidden_size * self.mlp_width_ratio)

        # Modulation
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(vec)

        # Broadcast modulation parameters to sequence dimension
        shift1_b = self._broadcast_modulation(shift1, batch, total_seq, self.hidden_size)
        scale1_b = self._broadcast_modulation(scale1, batch, total_seq, self.hidden_size)
        gate1_b = self._broadcast_modulation(gate1, batch, total_seq, self.hidden_size)

        # Modulation formula: x_modulated = x + x * scale + shift (equivalent to x * (1 + scale) + shift)
        x_scaled = x * scale1_b
        x_modulated = x + x_scaled + shift1_b

        # linear1: parallel output qkv + mlp_in
        linear1_out = x_modulated @ self.linear1
        q, k, v, mlp_in = self._split_qkv_mlp(
            linear1_out, batch, total_seq, intermediate_size
        )

        # QK-Norm
        if self.qk_norm:
            q = self._rms_norm(q, self.q_norm)
            k = self._rms_norm(k, self.k_norm)

        # Reshape for attention
        q_4d = self._reshape_for_attention(q, batch, total_seq)
        k_4d = self._reshape_for_attention(k, batch, total_seq)
        v_4d = self._reshape_for_attention(v, batch, total_seq)

        # Apply ROPE to image part only (first img_seq_len tokens)
        # Note: ROPE is applied in the forward pass but we don't modify
        # the tensor here since it's just shape tracking
        # In actual implementation, this would be: apply_rope(q, k, freqs_cis, img_seq_len)

        # Flash attention (non-causal for cross-modal)
        attn_out = flash_attention(q_4d, k_4d, v_4d, is_causal=False)

        # Reshape back
        attn_out_flat = attn_out.transpose(1, 2).view(
            batch, total_seq, self.heads_num * self.head_dim
        )

        # MLP (SwiGLU-style: gate * up)
        mlp_gate_out = silu(mlp_in)
        mlp_intermediate = mlp_gate_out * mlp_in

        # Concatenate attention and MLP outputs for linear2
        linear2_in = self._concat_hidden(attn_out_flat, mlp_intermediate, batch, total_seq)

        # linear2: combine outputs
        linear2_out = linear2_in @ self.linear2

        # Gate + residual
        x_scaled_gate1 = x * gate1_b
        x = x + x_scaled_gate1 * linear2_out

        # Track activations
        self._activations["x"] = x

        return x

    def _concat_hidden(
        self,
        attn_out: ShardedTensor,
        mlp_out: ShardedTensor,
        batch: int,
        seq: int,
    ) -> ShardedTensor:
        """Concatenate attention and MLP outputs along hidden dimension."""
        hidden_size = self.hidden_size
        intermediate_size = int(hidden_size * self.mlp_width_ratio)
        total_hidden = hidden_size + intermediate_size

        combined = ShardedTensor(
            shape=(batch, seq, total_hidden),
            shardable={},
            dtype=attn_out.dtype,
            name="linear2_input",
        )

        combined._op_history = attn_out._op_history
        combined._is_view = True

        return combined
