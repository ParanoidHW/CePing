"""Llama model implementation using kernel API."""

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from .base import BaseModel, ModelConfig, LayerConfig, SubmoduleType
from .sharding import ShardingInfo, ShardableDim, CommPattern, CommPosition, ShardedLayerConfig, ParallelDimType
from llm_perf.utils.constants import DTYPE_SIZES
from llm_perf.kernels import linear, rms_norm, silu, scaled_dot_product_attention, embedding
from llm_perf.kernels.utils import kernel_result_to_layer

if TYPE_CHECKING:
    from ..strategy.base import StrategyConfig


@dataclass
class LlamaConfig(ModelConfig):
    """Llama-specific configuration with sensible defaults."""

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.intermediate_size == 0:
            # Default Llama ratio (about 2.67x)
            self.intermediate_size = int(self.hidden_size * 8 / 3)


class LlamaModel(BaseModel):
    """Llama model implementation using kernel API."""

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self._layers = self.build_layers()

    def build_layers(self) -> List[LayerConfig]:
        """Build Llama layer configurations using kernel API."""
        layers = []
        cfg = self.config
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)

        emb_result = embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.hidden_size,
            input_shape=(1, cfg.max_seq_len),
            dtype=cfg.dtype,
        )
        layers.append(
            kernel_result_to_layer(name="embedding", result=emb_result, submodule_type=SubmoduleType.EMBEDDING)
        )

        for i in range(cfg.num_layers):
            layers.extend(self._build_transformer_layer(i, dtype_size))

        final_norm_result = rms_norm(input=(1, cfg.max_seq_len, cfg.hidden_size), dim=-1, dtype=cfg.dtype)
        layers.append(
            kernel_result_to_layer(name="final_norm", result=final_norm_result, submodule_type=SubmoduleType.LM_HEAD)
        )

        lm_head_result = linear(
            input=(cfg.max_seq_len, cfg.hidden_size),
            weight=(cfg.vocab_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype,
        )
        layers.append(
            kernel_result_to_layer(name="lm_head", result=lm_head_result, submodule_type=SubmoduleType.LM_HEAD)
        )

        return layers

    def build_sharded_layers(
        self,
        strategy: "StrategyConfig",
    ) -> List[ShardedLayerConfig]:
        """Build sharded layer configurations for a single GPU rank.

        Args:
            strategy: Parallelism strategy

        Returns:
            List of ShardedLayerConfig representing computation on one GPU
        """
        cfg = self.config
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)

        tp = strategy.tp_degree
        sp = strategy.sp_degree

        q_heads_per_gpu = max(1, cfg.num_attention_heads // tp)
        kv_heads_per_gpu = max(1, (cfg.num_key_value_heads or cfg.num_attention_heads) // tp)
        intermediate_per_gpu = max(1, cfg.intermediate_size // tp)
        seq_len_per_gpu = max(1, cfg.max_seq_len // sp) if sp > 1 else cfg.max_seq_len

        sharded_layers = []

        emb_result = embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.hidden_size,
            input_shape=(1, seq_len_per_gpu),
            dtype=cfg.dtype,
        )
        emb_layer = kernel_result_to_layer(name="embedding", result=emb_result, submodule_type=SubmoduleType.EMBEDDING)
        sharded_layers.append(self._create_sharded_layer(emb_layer, 0, tp, cfg.dtype, is_tp_shardable=False))

        for i in range(cfg.num_layers):
            sharded_attn = self._build_sharded_attention_layer(
                i, dtype_size, q_heads_per_gpu, kv_heads_per_gpu, seq_len_per_gpu, tp, cfg.dtype
            )
            sharded_layers.extend(sharded_attn)

            sharded_ffn = self._build_sharded_ffn_layer(
                i, dtype_size, intermediate_per_gpu, seq_len_per_gpu, tp, cfg.dtype
            )
            sharded_layers.extend(sharded_ffn)

        final_norm_result = rms_norm(input=(1, seq_len_per_gpu, cfg.hidden_size), dim=-1, dtype=cfg.dtype)
        final_norm_layer = kernel_result_to_layer(
            name="final_norm", result=final_norm_result, submodule_type=SubmoduleType.LM_HEAD
        )
        sharded_layers.append(
            self._create_sharded_layer(final_norm_layer, len(self._layers) - 2, tp, cfg.dtype, is_tp_shardable=False)
        )

        lm_head_result = linear(
            input=(seq_len_per_gpu, cfg.hidden_size),
            weight=(cfg.vocab_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype,
        )
        lm_head_layer = kernel_result_to_layer(
            name="lm_head", result=lm_head_result, submodule_type=SubmoduleType.LM_HEAD
        )
        sharded_layers.append(
            self._create_sharded_layer(lm_head_layer, len(self._layers) - 1, tp, cfg.dtype, is_tp_shardable=False)
        )

        return sharded_layers

    def _build_sharded_attention_layer(
        self,
        layer_idx: int,
        dtype_size: int,
        q_heads: int,
        kv_heads: int,
        seq_len: int,
        tp_degree: int,
        dtype: str,
    ) -> List[ShardedLayerConfig]:
        """Build sharded attention layer for one GPU."""
        cfg = self.config
        prefix = f"layer_{layer_idx}"
        m = seq_len
        head_dim = cfg.hidden_size // cfg.num_attention_heads

        sharded_layers = []

        input_norm_result = rms_norm(input=(1, seq_len, cfg.hidden_size), dim=-1, dtype=dtype)
        input_norm_layer = kernel_result_to_layer(
            name=f"{prefix}_input_norm", result=input_norm_result, submodule_type=SubmoduleType.ATTENTION
        )
        sharded_layers.append(
            self._create_sharded_layer(
                input_norm_layer,
                self._get_original_layer_idx(layer_idx, "input_norm"),
                tp_degree,
                dtype,
                is_tp_shardable=False,
            )
        )

        q_result = linear(
            input=(m, cfg.hidden_size), weight=(q_heads * head_dim, cfg.hidden_size), bias=None, dtype=dtype
        )
        q_layer = kernel_result_to_layer(
            name=f"{prefix}_q_proj", result=q_result, submodule_type=SubmoduleType.ATTENTION
        )
        q_sharding = ShardingInfo(
            shardable_dims={
                "heads": ShardableDim(
                    name="heads", dim_type=ParallelDimType.HEADS, original_size=cfg.num_attention_heads
                ),
            },
            layer_type="attention_q_proj",
        )
        sharded_layers.append(
            self._create_sharded_layer(
                q_layer,
                self._get_original_layer_idx(layer_idx, "q_proj"),
                tp_degree,
                dtype,
                sharding_info=q_sharding,
                comm_after_bytes=m * cfg.hidden_size * dtype_size if tp_degree > 1 else 0,
            )
        )

        k_result = linear(
            input=(m, cfg.hidden_size), weight=(kv_heads * head_dim, cfg.hidden_size), bias=None, dtype=dtype
        )
        k_layer = kernel_result_to_layer(
            name=f"{prefix}_k_proj", result=k_result, submodule_type=SubmoduleType.ATTENTION
        )
        sharded_layers.append(
            self._create_sharded_layer(
                k_layer, self._get_original_layer_idx(layer_idx, "k_proj"), tp_degree, dtype, is_tp_shardable=True
            )
        )

        v_result = linear(
            input=(m, cfg.hidden_size), weight=(kv_heads * head_dim, cfg.hidden_size), bias=None, dtype=dtype
        )
        v_layer = kernel_result_to_layer(
            name=f"{prefix}_v_proj", result=v_result, submodule_type=SubmoduleType.ATTENTION
        )
        sharded_layers.append(
            self._create_sharded_layer(
                v_layer, self._get_original_layer_idx(layer_idx, "v_proj"), tp_degree, dtype, is_tp_shardable=True
            )
        )

        attn_result = scaled_dot_product_attention(
            query=(1, q_heads, seq_len, head_dim),
            key=(1, kv_heads, seq_len, head_dim),
            value=(1, kv_heads, seq_len, head_dim),
            is_causal=True,
            dtype=dtype,
        )
        attn_layer = kernel_result_to_layer(
            name=f"{prefix}_attention", result=attn_result, submodule_type=SubmoduleType.ATTENTION
        )
        sharded_layers.append(
            self._create_sharded_layer(
                attn_layer, self._get_original_layer_idx(layer_idx, "attention"), tp_degree, dtype, is_tp_shardable=True
            )
        )

        o_result = linear(input=(m, cfg.hidden_size), weight=(cfg.hidden_size, cfg.hidden_size), bias=None, dtype=dtype)
        o_layer = kernel_result_to_layer(
            name=f"{prefix}_o_proj", result=o_result, submodule_type=SubmoduleType.ATTENTION
        )
        attn_comm = ShardingInfo(
            shardable_dims={
                "hidden": ShardableDim(name="hidden", dim_type=ParallelDimType.HIDDEN, original_size=cfg.hidden_size),
            },
            comm_patterns=[
                CommPattern(
                    comm_type="allreduce",
                    position=CommPosition.AFTER,
                    data_shape=(1, seq_len, cfg.hidden_size),
                    data_dtype=dtype,
                    description="TP allreduce after O projection",
                )
            ],
            layer_type="attention_o_proj",
        )
        sharded_layers.append(
            self._create_sharded_layer(
                o_layer,
                self._get_original_layer_idx(layer_idx, "o_proj"),
                tp_degree,
                dtype,
                sharding_info=attn_comm,
                comm_after_bytes=m * cfg.hidden_size * dtype_size if tp_degree > 1 else 0,
            )
        )

        return sharded_layers

    def _build_sharded_ffn_layer(
        self,
        layer_idx: int,
        dtype_size: int,
        intermediate_size: int,
        seq_len: int,
        tp_degree: int,
        dtype: str,
    ) -> List[ShardedLayerConfig]:
        """Build sharded FFN layer for one GPU."""
        cfg = self.config
        prefix = f"layer_{layer_idx}"
        m = seq_len

        sharded_layers = []

        attn_norm_result = rms_norm(input=(1, seq_len, cfg.hidden_size), dim=-1, dtype=dtype)
        attn_norm_layer = kernel_result_to_layer(
            name=f"{prefix}_post_attn_norm", result=attn_norm_result, submodule_type=SubmoduleType.FFN
        )
        sharded_layers.append(
            self._create_sharded_layer(
                attn_norm_layer,
                self._get_original_layer_idx(layer_idx, "post_attn_norm"),
                tp_degree,
                dtype,
                is_tp_shardable=False,
            )
        )

        up_result = linear(
            input=(m, cfg.hidden_size), weight=(intermediate_size, cfg.hidden_size), bias=None, dtype=dtype
        )
        up_layer = kernel_result_to_layer(name=f"{prefix}_up_proj", result=up_result, submodule_type=SubmoduleType.FFN)
        sharded_layers.append(
            self._create_sharded_layer(
                up_layer, self._get_original_layer_idx(layer_idx, "up_proj"), tp_degree, dtype, is_tp_shardable=True
            )
        )

        gate_result = linear(
            input=(m, cfg.hidden_size), weight=(intermediate_size, cfg.hidden_size), bias=None, dtype=dtype
        )
        gate_layer = kernel_result_to_layer(
            name=f"{prefix}_gate_proj", result=gate_result, submodule_type=SubmoduleType.FFN
        )
        ffn_sharding = ShardingInfo(
            shardable_dims={
                "intermediate_size": ShardableDim(
                    name="intermediate_size",
                    dim_type=ParallelDimType.INTERMEDIATE,
                    original_size=cfg.intermediate_size,
                ),
            },
            layer_type="ffn_gate_proj",
        )
        sharded_layers.append(
            self._create_sharded_layer(
                gate_layer,
                self._get_original_layer_idx(layer_idx, "gate_proj"),
                tp_degree,
                dtype,
                sharding_info=ffn_sharding,
            )
        )

        swiglu_result = silu(input=(1, seq_len, intermediate_size), dtype=dtype)
        swiglu_layer = kernel_result_to_layer(
            name=f"{prefix}_swiglu", result=swiglu_result, submodule_type=SubmoduleType.FFN
        )
        sharded_layers.append(
            self._create_sharded_layer(
                swiglu_layer, self._get_original_layer_idx(layer_idx, "swiglu"), tp_degree, dtype, is_tp_shardable=True
            )
        )

        down_result = linear(
            input=(m, intermediate_size), weight=(cfg.hidden_size, intermediate_size), bias=None, dtype=dtype
        )
        down_layer = kernel_result_to_layer(
            name=f"{prefix}_down_proj", result=down_result, submodule_type=SubmoduleType.FFN
        )
        ffn_comm = ShardingInfo(
            shardable_dims={
                "intermediate_size": ShardableDim(
                    name="intermediate_size",
                    dim_type=ParallelDimType.INTERMEDIATE,
                    original_size=cfg.intermediate_size,
                ),
            },
            comm_patterns=[
                CommPattern(
                    comm_type="allreduce",
                    position=CommPosition.AFTER,
                    data_shape=(1, seq_len, cfg.hidden_size),
                    data_dtype=dtype,
                    description="TP allreduce after FFN down projection",
                )
            ],
            layer_type="ffn_down_proj",
        )
        sharded_layers.append(
            self._create_sharded_layer(
                down_layer,
                self._get_original_layer_idx(layer_idx, "down_proj"),
                tp_degree,
                dtype,
                sharding_info=ffn_comm,
                comm_after_bytes=m * cfg.hidden_size * dtype_size if tp_degree > 1 else 0,
            )
        )

        ffn_norm_result = rms_norm(input=(1, seq_len, cfg.hidden_size), dim=-1, dtype=dtype)
        ffn_norm_layer = kernel_result_to_layer(
            name=f"{prefix}_ffn_norm", result=ffn_norm_result, submodule_type=SubmoduleType.FFN
        )
        sharded_layers.append(
            self._create_sharded_layer(
                ffn_norm_layer,
                self._get_original_layer_idx(layer_idx, "ffn_norm"),
                tp_degree,
                dtype,
                is_tp_shardable=False,
            )
        )

        return sharded_layers

    def _create_sharded_layer(
        self,
        layer: LayerConfig,
        original_idx: int,
        tp_degree: int,
        dtype: str,
        is_tp_shardable: bool = True,
        sharding_info: Optional[ShardingInfo] = None,
        comm_after_bytes: int = 0,
    ) -> ShardedLayerConfig:
        """Create ShardedLayerConfig from LayerConfig."""
        return ShardedLayerConfig(
            name=layer.name,
            original_layer_idx=original_idx,
            sharded_input_shape=layer.input_shape,
            sharded_output_shape=layer.output_shape,
            sharded_flops=layer.flops,
            sharded_params=layer.params_count,
            sharded_activation_bytes=layer.activation_bytes,
            sharding_info=sharding_info,
            comm_before_bytes=0,
            comm_after_bytes=comm_after_bytes,
        )

    def _get_original_layer_idx(self, layer_idx: int, sublayer_name: str) -> int:
        """Get original layer index for a sublayer."""
        base_idx = 1 + layer_idx * 12
        sublayer_offsets = {
            "input_norm": 0,
            "q_proj": 1,
            "k_proj": 2,
            "v_proj": 3,
            "attention": 4,
            "o_proj": 5,
            "post_attn_norm": 6,
            "up_proj": 7,
            "gate_proj": 8,
            "swiglu": 9,
            "down_proj": 10,
            "ffn_norm": 11,
        }
        return base_idx + sublayer_offsets.get(sublayer_name, 0)

    def _build_transformer_layer(self, layer_idx: int, dtype_size: int) -> List[LayerConfig]:
        """Build a single transformer layer (attention + FFN) using kernel API."""
        layers = []
        cfg = self.config
        prefix = f"layer_{layer_idx}"
        seq_len = cfg.max_seq_len
        m = seq_len  # Flattened batch*seq for linear ops

        # === Attention Components ===
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        q_heads = cfg.num_attention_heads
        kv_heads = cfg.num_key_value_heads or cfg.num_attention_heads

        # Pre-attention RMSNorm (merged into ATTENTION)
        input_norm_result = rms_norm(input=(1, seq_len, cfg.hidden_size), dim=-1, dtype=cfg.dtype)
        layers.append(
            kernel_result_to_layer(
                name=f"{prefix}_input_norm", result=input_norm_result, submodule_type=SubmoduleType.ATTENTION
            )
        )

        # Q projection using linear kernel
        q_result = linear(
            input=(m, cfg.hidden_size), weight=(q_heads * head_dim, cfg.hidden_size), bias=None, dtype=cfg.dtype
        )
        layers.append(
            kernel_result_to_layer(name=f"{prefix}_q_proj", result=q_result, submodule_type=SubmoduleType.ATTENTION)
        )

        # K projection using linear kernel
        k_result = linear(
            input=(m, cfg.hidden_size), weight=(kv_heads * head_dim, cfg.hidden_size), bias=None, dtype=cfg.dtype
        )
        layers.append(
            kernel_result_to_layer(name=f"{prefix}_k_proj", result=k_result, submodule_type=SubmoduleType.ATTENTION)
        )

        # V projection using linear kernel
        v_result = linear(
            input=(m, cfg.hidden_size), weight=(kv_heads * head_dim, cfg.hidden_size), bias=None, dtype=cfg.dtype
        )
        layers.append(
            kernel_result_to_layer(name=f"{prefix}_v_proj", result=v_result, submodule_type=SubmoduleType.ATTENTION)
        )

        # Attention computation using scaled_dot_product_attention kernel
        attn_result = scaled_dot_product_attention(
            query=(1, q_heads, seq_len, head_dim),
            key=(1, kv_heads, seq_len, head_dim),
            value=(1, kv_heads, seq_len, head_dim),
            is_causal=True,
            dtype=cfg.dtype,
        )
        layers.append(
            kernel_result_to_layer(
                name=f"{prefix}_attention", result=attn_result, submodule_type=SubmoduleType.ATTENTION
            )
        )

        # O projection using linear kernel
        o_result = linear(
            input=(m, cfg.hidden_size), weight=(cfg.hidden_size, cfg.hidden_size), bias=None, dtype=cfg.dtype
        )
        layers.append(
            kernel_result_to_layer(name=f"{prefix}_o_proj", result=o_result, submodule_type=SubmoduleType.ATTENTION)
        )

        # Post-attention RMSNorm (merged into FFN)
        attn_norm_result = rms_norm(input=(1, seq_len, cfg.hidden_size), dim=-1, dtype=cfg.dtype)
        layers.append(
            kernel_result_to_layer(
                name=f"{prefix}_post_attn_norm", result=attn_norm_result, submodule_type=SubmoduleType.FFN
            )
        )

        # === FFN Components ===
        # Llama uses SwiGLU: up_proj, gate_proj, down_proj
        ffn_intermediate = cfg.intermediate_size

        # Up projection using linear kernel
        up_result = linear(
            input=(m, cfg.hidden_size), weight=(ffn_intermediate, cfg.hidden_size), bias=None, dtype=cfg.dtype
        )
        layers.append(
            kernel_result_to_layer(name=f"{prefix}_up_proj", result=up_result, submodule_type=SubmoduleType.FFN)
        )

        # Gate projection using linear kernel
        gate_result = linear(
            input=(m, cfg.hidden_size), weight=(ffn_intermediate, cfg.hidden_size), bias=None, dtype=cfg.dtype
        )
        layers.append(
            kernel_result_to_layer(name=f"{prefix}_gate_proj", result=gate_result, submodule_type=SubmoduleType.FFN)
        )

        # SwiGLU activation using silu kernel (x * sigmoid(x))
        swiglu_result = silu(input=(1, seq_len, ffn_intermediate), dtype=cfg.dtype)
        layers.append(
            kernel_result_to_layer(name=f"{prefix}_swiglu", result=swiglu_result, submodule_type=SubmoduleType.FFN)
        )

        # Down projection using linear kernel
        down_result = linear(
            input=(m, ffn_intermediate), weight=(cfg.hidden_size, ffn_intermediate), bias=None, dtype=cfg.dtype
        )
        layers.append(
            kernel_result_to_layer(name=f"{prefix}_down_proj", result=down_result, submodule_type=SubmoduleType.FFN)
        )

        # Post-FFN RMSNorm (merged into FFN)
        ffn_norm_result = rms_norm(input=(1, seq_len, cfg.hidden_size), dim=-1, dtype=cfg.dtype)
        layers.append(
            kernel_result_to_layer(name=f"{prefix}_ffn_norm", result=ffn_norm_result, submodule_type=SubmoduleType.FFN)
        )

        return layers
