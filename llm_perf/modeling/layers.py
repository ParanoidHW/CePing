"""Basic modules for transformer models.

Includes:
- ShardedEmbedding: Embedding layer with TP vocab sharding
- ShardedRMSNorm: RMS Normalization
- ShardedAttention: Attention layer with TP/SP sharding
- ShardedFFN: Feed-Forward Network with TP sharding
- ShardedLMHead: LM Head with TP vocab sharding
"""

from typing import Optional
from llm_perf.modeling.module import ShardedModule
from llm_perf.modeling.tensor import ShardedTensor, ShardedParameter
from llm_perf.kernels.op import RMSNormOp, EmbeddingOp, ActivationOp
from llm_perf.utils.constants import FFNActType


class ShardedEmbedding(ShardedModule):
    """Embedding layer, similar to torch.nn.Embedding.

    Sharding:
    - vocab dimension is TP-sharded
    - Output needs AllGather for inference

    Args:
        num_embeddings: Vocabulary size
        embedding_dim: Embedding dimension
        dtype: Data type
    """

    _submodule_name = "embedding"

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = ShardedParameter(
            shape=(num_embeddings, embedding_dim),
            shardable={0: "tp"},
            dtype=dtype,
            name="embedding_weight",
        )

    def forward(self, input_ids: ShardedTensor) -> ShardedTensor:
        """Embedding lookup.

        Args:
            input_ids: (batch, seq) or complex shape

        Returns:
            output: (batch, seq, embedding_dim)
        """
        input_shape = input_ids.shape
        output_shape = (*input_shape, self.embedding_dim)

        output_shardable = {}
        if len(input_shape) >= 2 and 1 in input_ids.shardable:
            output_shardable[len(input_shape) - 1] = input_ids.shardable[1]

        output = ShardedTensor(
            shape=output_shape,
            shardable=output_shardable,
            dtype=self.weight.dtype,
            name="embedding_output",
        )

        output._op_history = input_ids._op_history + [
            EmbeddingOp(
                dtype=self.weight.dtype,
                input_ids=input_ids,
                weight=self.weight,
                output=output,
            )
        ]

        self._activations["output"] = output

        return output


class ShardedRMSNorm(ShardedModule):
    """RMS Normalization layer.

    Sharding:
    - hidden dimension not sharded (weight is replicated)

    Args:
        hidden_size: Hidden size
        dtype: Data type
    """

    _submodule_name = "rms_norm"

    def __init__(
        self,
        hidden_size: int,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.weight = ShardedParameter(
            shape=(hidden_size,),
            shardable={},
            dtype=dtype,
            name="rmsnorm_weight",
        )

    def forward(self, input_tensor: ShardedTensor) -> ShardedTensor:
        """RMS Normalization.

        Args:
            input: (..., hidden_size)

        Returns:
            output: (..., hidden_size), sharding constraints preserved
        """
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
                weight=self.weight,
                output=output,
            )
        ]

        self._track_intermediate("output", output)
        self._activations["rmsnorm_output"] = output

        return output


def silu(input_tensor: ShardedTensor) -> ShardedTensor:
    """SiLU activation: x * sigmoid(x).

    Args:
        input: Input tensor

    Returns:
        output: Same shape, same sharding
    """
    output = ShardedTensor(
        shape=input_tensor.shape,
        shardable=input_tensor.shardable,
        dtype=input_tensor.dtype,
        name="silu_output",
    )

    output._op_history = input_tensor._op_history + [
        ActivationOp(
            dtype=input_tensor.dtype,
            input=input_tensor,
            output=output,
            activation_type="silu",
        )
    ]

    return output


def gelu(input_tensor: ShardedTensor, approximate: str = "none") -> ShardedTensor:
    """GELU activation.

    Args:
        input: Input tensor
        approximate: Approximation method

    Returns:
        output: Same shape, same sharding
    """
    output = ShardedTensor(
        shape=input_tensor.shape,
        shardable=input_tensor.shardable,
        dtype=input_tensor.dtype,
        name="gelu_output",
    )

    output._op_history = input_tensor._op_history + [
        ActivationOp(
            dtype=input_tensor.dtype,
            input=input_tensor,
            output=output,
            activation_type="gelu",
        )
    ]

    return output


def relu(input_tensor: ShardedTensor) -> ShardedTensor:
    """ReLU activation.

    Args:
        input: Input tensor

    Returns:
        output: Same shape, same sharding
    """
    output = ShardedTensor(
        shape=input_tensor.shape,
        shardable=input_tensor.shardable,
        dtype=input_tensor.dtype,
        name="relu_output",
    )

    output._op_history = input_tensor._op_history + [
        ActivationOp(
            dtype=input_tensor.dtype,
            input=input_tensor,
            output=output,
            activation_type="relu",
        )
    ]

    return output


def flash_attention(
    query: ShardedTensor,
    key: ShardedTensor,
    value: ShardedTensor,
    is_causal: bool = True,
) -> ShardedTensor:
    """Flash Attention operation.

    Args:
        query: (batch, heads, seq, head_dim)
        key: (batch, kv_heads, kv_seq, head_dim)
        value: (batch, kv_heads, kv_seq, head_dim)
        is_causal: Whether causal mask

    Returns:
        output: (batch, heads, seq, head_dim)
    """
    from llm_perf.kernels.op import AttentionOp

    batch, heads, seq, head_dim = query.shape

    output_shardable = {}
    if 1 in query.shardable:
        output_shardable[1] = query.shardable[1]
    if 2 in query.shardable:
        output_shardable[2] = query.shardable[2]

    output = ShardedTensor(
        shape=(batch, heads, seq, head_dim),
        shardable=output_shardable,
        dtype=query.dtype,
        name="attention_output",
    )

    output._op_history = query._op_history + [
        AttentionOp(
            dtype=query.dtype,
            query=query,
            key=key,
            value=value,
            output=output,
            is_causal=is_causal,
        )
    ]

    return output


class ShardedAttention(ShardedModule):
    """Attention layer.

    Sharding:
    - Q/K/V weights: heads dimension TP-sharded (column sharding)
    - O weight: heads dimension TP-sharded (row sharding)
    - Output needs AllReduce aggregation

    Supports GQA: num_kv_heads < num_heads

    Args:
        hidden_size: Hidden size
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (GQA)
        head_dim: Head dimension
        dtype: Data type
    """

    _submodule_name = "attention"

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)

        self.q_weight = ShardedParameter(
            shape=(hidden_size, num_heads * self.head_dim),
            shardable={1: "tp"},
            dtype=dtype,
            name="q_weight",
        )

        self.k_weight = ShardedParameter(
            shape=(hidden_size, self.num_kv_heads * self.head_dim),
            shardable={1: "tp"},
            dtype=dtype,
            name="k_weight",
        )

        self.v_weight = ShardedParameter(
            shape=(hidden_size, self.num_kv_heads * self.head_dim),
            shardable={1: "tp"},
            dtype=dtype,
            name="v_weight",
        )

        self.o_weight = ShardedParameter(
            shape=(num_heads * self.head_dim, hidden_size),
            shardable={0: "tp"},
            dtype=dtype,
            name="o_weight",
        )

    def forward(
        self,
        hidden: ShardedTensor,
        is_causal: bool = True,
    ) -> ShardedTensor:
        """Attention forward with Flash Attention optimization.

        Memory optimization: Only save backward-required activations.
        - q_proj/k_proj/v_proj: projection outputs (needed for gradient)
        - output: final output (needed for gradient)

        Not saved (can be recomputed in backward):
        - q/k/v: view tensors from projections
        - attn_out: Flash Attention output (recomputed in backward)

        Args:
            hidden: (batch, seq, hidden_size)

        Returns:
            output: (batch, seq, hidden_size)
        """
        batch = hidden.shape[0] if len(hidden.shape) >= 1 else 1
        seq = hidden.shape[1] if len(hidden.shape) >= 2 else 1

        kv_seq = self._kv_seq_len if self._kv_seq_len is not None else seq

        q_proj = self._track_intermediate("q_proj", hidden @ self.q_weight)
        k_proj = self._track_intermediate("k_proj", hidden @ self.k_weight)
        v_proj = self._track_intermediate("v_proj", hidden @ self.v_weight)

        q = q_proj.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = ShardedTensor(
            shape=(batch, self.num_kv_heads, kv_seq, self.head_dim),
            shardable={1: "tp"},
            dtype=hidden.dtype,
            name="k_for_attention",
        )
        v = ShardedTensor(
            shape=(batch, self.num_kv_heads, kv_seq, self.head_dim),
            shardable={1: "tp"},
            dtype=hidden.dtype,
            name="v_for_attention",
        )

        attn_out = flash_attention(q, k, v, is_causal=is_causal)

        # Reshape - don't save (view tensor)
        attn_flat = attn_out.transpose(1, 2).view(batch, seq, self.num_heads * self.head_dim)

        # Save output (needed for backward)
        output = self._track_intermediate("output", attn_flat @ self.o_weight)

        self._activations["q_proj"] = q_proj
        self._activations["k_proj"] = k_proj
        self._activations["v_proj"] = v_proj
        self._activations["output"] = output

        return output


class ShardedFFN(ShardedModule):
    """Feed-Forward Network layer.

    Sharding:
    - gate/up weights: intermediate dimension TP-sharded (column)
    - down weight: intermediate dimension TP-sharded (row)
    - Output needs AllReduce aggregation

    Args:
        hidden_size: Hidden size
        intermediate_size: Intermediate size
        ffn_act_type: FFN activation type (swiglu, gelu, relu, silu)
        dtype: Data type
    """

    _submodule_name = "ffn"

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        ffn_act_type: str = FFNActType.SWIGLU.value,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.ffn_act_type = ffn_act_type

        if ffn_act_type == FFNActType.SWIGLU.value:
            self.gate_weight = ShardedParameter(
                shape=(hidden_size, intermediate_size),
                shardable={1: "tp"},
                dtype=dtype,
                name="gate_weight",
            )
            self.up_weight = ShardedParameter(
                shape=(hidden_size, intermediate_size),
                shardable={1: "tp"},
                dtype=dtype,
                name="up_weight",
            )
        else:
            self.up_weight = ShardedParameter(
                shape=(hidden_size, intermediate_size),
                shardable={1: "tp"},
                dtype=dtype,
                name="up_weight",
            )

        self.down_weight = ShardedParameter(
            shape=(intermediate_size, hidden_size),
            shardable={0: "tp"},
            dtype=dtype,
            name="down_weight",
        )

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """FFN forward.

        Args:
            hidden: (batch, seq, hidden_size)

        Returns:
            output: (batch, seq, hidden_size)
        """
        if self.ffn_act_type == FFNActType.SWIGLU.value:
            gate_proj = self._track_intermediate("gate_proj", hidden @ self.gate_weight)
            gate_silu = self._track_intermediate("gate_silu", silu(gate_proj))
            up_proj = self._track_intermediate("up_proj", hidden @ self.up_weight)
            intermediate = self._track_intermediate("intermediate", gate_silu * up_proj)
        elif self.ffn_act_type == FFNActType.GELU.value:
            up_proj = self._track_intermediate("up_proj", hidden @ self.up_weight)
            intermediate = self._track_intermediate("intermediate", gelu(up_proj))
        elif self.ffn_act_type == FFNActType.RELU.value:
            up_proj = self._track_intermediate("up_proj", hidden @ self.up_weight)
            intermediate = self._track_intermediate("intermediate", relu(up_proj))
        elif self.ffn_act_type == FFNActType.SILU.value:
            up_proj = self._track_intermediate("up_proj", hidden @ self.up_weight)
            intermediate = self._track_intermediate("intermediate", silu(up_proj))
        else:
            raise ValueError(f"Unknown FFN activation type: {self.ffn_act_type}")

        output = self._track_intermediate("output", intermediate @ self.down_weight)

        self._activations["up_proj"] = up_proj
        self._activations["intermediate"] = intermediate

        return output


class ShardedLMHead(ShardedModule):
    """LM Head layer.

    Sharding:
    - vocab dimension TP-sharded
    - Output needs AllGather for inference

    Args:
        hidden_size: Hidden size
        vocab_size: Vocabulary size
        dtype: Data type
    """

    _submodule_name = "lm_head"

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.weight = ShardedParameter(
            shape=(hidden_size, vocab_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="lm_head_weight",
        )

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """LM Head forward.

        Args:
            hidden: (batch, seq, hidden_size)

        Returns:
            logits: (batch, seq, vocab_size)
        """
        logits = hidden @ self.weight

        self._track_intermediate("logits", logits)
        self._activations["lm_head_logits"] = logits

        return logits


class ShardedMoE(ShardedModule):
    """Mixture of Experts layer.

    Sharding:
    - Router: not sharded (replicated)
    - Expert weights: EP-sharded (experts), TP-sharded (intermediate)
    - Shared experts: TP-sharded only
    - Communication: All2All (EP) + AllReduce (TP)

    Args:
        hidden_size: Hidden size
        intermediate_size: Expert intermediate size
        num_experts: Number of experts
        num_experts_per_token: Number of active experts per token
        shared_expert_intermediate: Shared expert intermediate size (optional)
        dtype: Data type
    """

    _submodule_name = "moe"

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_token: int = 1,
        shared_expert_intermediate: Optional[int] = None,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.shared_expert_intermediate = shared_expert_intermediate

        self.router_weight = ShardedParameter(
            shape=(hidden_size, num_experts),
            shardable={},
            dtype=dtype,
            name="router_weight",
        )

        self.expert_gate_weight = ShardedParameter(
            shape=(num_experts, hidden_size, intermediate_size),
            shardable={0: "ep", 2: "tp"},
            dtype=dtype,
            name="expert_gate_weight",
        )
        self.expert_up_weight = ShardedParameter(
            shape=(num_experts, hidden_size, intermediate_size),
            shardable={0: "ep", 2: "tp"},
            dtype=dtype,
            name="expert_up_weight",
        )
        self.expert_down_weight = ShardedParameter(
            shape=(num_experts, intermediate_size, hidden_size),
            shardable={0: "ep", 1: "tp"},
            dtype=dtype,
            name="expert_down_weight",
        )

        if shared_expert_intermediate:
            self.shared_gate_weight = ShardedParameter(
                shape=(hidden_size, shared_expert_intermediate),
                shardable={1: "tp"},
                dtype=dtype,
                name="shared_gate_weight",
            )
            self.shared_up_weight = ShardedParameter(
                shape=(hidden_size, shared_expert_intermediate),
                shardable={1: "tp"},
                dtype=dtype,
                name="shared_up_weight",
            )
            self.shared_down_weight = ShardedParameter(
                shape=(shared_expert_intermediate, hidden_size),
                shardable={0: "tp"},
                dtype=dtype,
                name="shared_down_weight",
            )

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """MoE forward.

        Args:
            hidden: (batch, seq, hidden_size)

        Returns:
            output: (batch, seq, hidden_size)
        """
        router_logits = hidden @ self.router_weight

        from llm_perf.kernels.op import MoEExpertOp

        expert_out = ShardedTensor(
            shape=hidden.shape,
            shardable=hidden.shardable,
            dtype=hidden.dtype,
            name="expert_out",
        )
        expert_out._op_history = hidden._op_history + [
            MoEExpertOp(
                dtype=hidden.dtype,
                hidden=hidden,
                expert_gate_weights=self.expert_gate_weight,
                expert_up_weights=self.expert_up_weight,
                expert_down_weights=self.expert_down_weight,
                num_experts_per_token=self.num_experts_per_token,
                output=expert_out,
            )
        ]

        if self.shared_expert_intermediate:
            shared_gate = hidden @ self.shared_gate_weight
            shared_gate = silu(shared_gate)
            shared_up = hidden @ self.shared_up_weight
            shared_intermediate = shared_gate * shared_up
            shared_out = shared_intermediate @ self.shared_down_weight
            output = expert_out + shared_out
        else:
            output = expert_out

        self._activations["router_logits"] = router_logits
        self._activations["expert_out"] = expert_out

        return output


def linear_attention(
    query: ShardedTensor,
    key: ShardedTensor,
    value: ShardedTensor,
    kernel_dim: int = 4,
    is_causal: bool = True,
) -> ShardedTensor:
    """Linear Attention operation.

    Args:
        query: (batch, heads, seq, head_dim)
        key: (batch, kv_heads, seq, head_dim)
        value: (batch, kv_heads, seq, head_dim)
        kernel_dim: Feature map dimension (default: 4 for Qwen3.5)
        is_causal: Whether causal mask

    Returns:
        output: (batch, heads, seq, head_dim)
    """
    from llm_perf.kernels.op import LinearAttentionOp

    batch, heads, seq, head_dim = query.shape

    output_shardable = {}
    if 1 in query.shardable:
        output_shardable[1] = query.shardable[1]

    output = ShardedTensor(
        shape=(batch, heads, seq, head_dim),
        shardable=output_shardable,
        dtype=query.dtype,
        name="linear_attention_output",
    )

    output._op_history = query._op_history + [
        LinearAttentionOp(
            dtype=query.dtype,
            query=query,
            key=key,
            value=value,
            output=output,
            kernel_dim=kernel_dim,
            is_causal=is_causal,
        )
    ]

    return output


class ShardedLinearAttention(ShardedModule):
    """Linear Attention layer with O(seq) complexity.

    Unlike standard attention which has O(seq^2) complexity,
    Linear Attention reformulates the computation to achieve O(seq):
    - Standard: softmax(QK^T)V
    - Linear: Q(K^T V) with kernel feature map

    Key differences from ShardedAttention:
    1. No KV cache needed - maintains fixed-size state per head
    2. No kv_seq_len parameter - state size is independent of sequence length
    3. Memory does not grow with sequence length

    Sharding:
    - Q/K/V weights: heads dimension TP-sharded (column sharding)
    - O weight: heads dimension TP-sharded (row sharding)
    - Output needs AllReduce aggregation

    Reference: "Linear Transformers Are Secretly Fast Weight Programmers"
    https://arxiv.org/abs/2006.16236

    Args:
        hidden_size: Hidden size
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads (for GQA support)
        head_dim: Head dimension
        kernel_dim: Feature map dimension (default: 4 for Qwen3.5)
        dtype: Data type
    """

    _submodule_name = "linear_attention"

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        kernel_dim: int = 4,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.kernel_dim = kernel_dim

        self.q_weight = ShardedParameter(
            shape=(hidden_size, num_heads * self.head_dim),
            shardable={1: "tp"},
            dtype=dtype,
            name="q_weight",
        )

        self.k_weight = ShardedParameter(
            shape=(hidden_size, self.num_kv_heads * self.head_dim),
            shardable={1: "tp"},
            dtype=dtype,
            name="k_weight",
        )

        self.v_weight = ShardedParameter(
            shape=(hidden_size, self.num_kv_heads * self.head_dim),
            shardable={1: "tp"},
            dtype=dtype,
            name="v_weight",
        )

        self.o_weight = ShardedParameter(
            shape=(num_heads * self.head_dim, hidden_size),
            shardable={0: "tp"},
            dtype=dtype,
            name="o_weight",
        )

    def forward(
        self,
        hidden: ShardedTensor,
        is_causal: bool = True,
    ) -> ShardedTensor:
        """Linear Attention forward.

        Args:
            hidden: (batch, seq, hidden_size)
            is_causal: Whether causal mask

        Returns:
            output: (batch, seq, hidden_size)
        """
        batch = hidden.shape[0] if len(hidden.shape) >= 1 else 1
        seq = hidden.shape[1] if len(hidden.shape) >= 2 else 1

        q_proj = self._track_intermediate("q_proj", hidden @ self.q_weight)
        k_proj = self._track_intermediate("k_proj", hidden @ self.k_weight)
        v_proj = self._track_intermediate("v_proj", hidden @ self.v_weight)

        q = q_proj.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_proj.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v_proj.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        attn_out = linear_attention(q, k, v, kernel_dim=self.kernel_dim, is_causal=is_causal)

        attn_flat = attn_out.transpose(1, 2).view(batch, seq, self.num_heads * self.head_dim)

        output = self._track_intermediate("output", attn_flat @ self.o_weight)

        self._activations["q_proj"] = q_proj
        self._activations["k_proj"] = k_proj
        self._activations["v_proj"] = v_proj
        self._activations["output"] = output

        return output
