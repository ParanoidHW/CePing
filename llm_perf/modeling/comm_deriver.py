"""Communication Pattern Deriver.

Systematically derives communication operations from tensor sharding changes.
"""

import logging
import math
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from llm_perf.kernels.op import (
    Op,
    MatmulOp,
    AttentionOp,
    EmbeddingOp,
    MoEExpertOp,
    CommOp,
)
from llm_perf.utils.constants import DTYPE_SIZES

if TYPE_CHECKING:
    from llm_perf.modeling.tensor import ShardedTensor
    from llm_perf.strategy.parallel_context import ParallelContext

logger = logging.getLogger(__name__)


class CommPatternDeriver:
    """Derives communication patterns from tensor sharding changes.

    Communication is needed when tensor sharding changes across operations:
    - Sharded -> Replicated: allgather
    - Replicated -> Sharded: reduce_scatter
    - Sharded(tp) -> Sharded(sp): alltoall
    - Different ptype sharding: alltoall
    """

    def __init__(self, parallel_degrees: Dict[str, int]):
        """Initialize deriver.

        Args:
            parallel_degrees: {parallel_type: degree} mapping
        """
        self.parallel_degrees = parallel_degrees

    def derive_comm_ops(
        self,
        op: Op,
        physical_shapes: Dict[str, tuple],
    ) -> List[CommOp]:
        """Derive communication operations from an op.

        Args:
            op: Operation to analyze
            physical_shapes: Pre-computed physical shapes for tensors

        Returns:
            List of CommOp for this operation
        """
        ops = []

        if isinstance(op, MatmulOp):
            ops.extend(self._derive_matmul_comm(op, physical_shapes))
        elif isinstance(op, AttentionOp):
            ops.extend(self._derive_attention_comm(op, physical_shapes))
        elif isinstance(op, EmbeddingOp):
            ops.extend(self._derive_embedding_comm(op, physical_shapes))
        elif isinstance(op, MoEExpertOp):
            ops.extend(self._derive_moe_comm(op, physical_shapes))

        return ops

    def _derive_matmul_comm(
        self,
        op: MatmulOp,
        physical_shapes: Dict[str, tuple],
    ) -> List[CommOp]:
        """Derive communication for MatmulOp.

        Cases:
        1. Row sharding (weight dim 0): allreduce on output
        2. Column sharding (weight dim 1): no comm on forward, allgather on backward

        Supports both 2D and 3D input tensors.
        """
        ops = []

        input_shardable = op.input.shardable
        weight_shardable = op.weight.shardable
        output_shardable = op.output.shardable

        dtype_size = DTYPE_SIZES.get(op.dtype, 2)

        weight_dim_0_sharded = 0 in weight_shardable
        weight_dim_1_sharded = 1 in weight_shardable

        need_allreduce = False
        ptype = None

        if weight_dim_1_sharded:
            ptype = weight_shardable[1]

            output_not_sharded = not output_shardable or (
                0 not in output_shardable and 1 not in output_shardable and 2 not in output_shardable
            )

            if output_not_sharded:
                need_allreduce = True

        elif weight_dim_0_sharded:
            ptype = weight_shardable[0]

            output_sharded_on_last_dim = len(op.output.shape) >= 2 and (len(op.output.shape) - 1) in output_shardable

            if not output_sharded_on_last_dim:
                need_allreduce = True

        if len(op.input.shape) >= 2:
            last_dim = len(op.input.shape) - 1
            if last_dim in input_shardable:
                input_ptype = input_shardable[last_dim]
                if last_dim not in output_shardable:
                    need_allreduce = True
                    if ptype is None:
                        ptype = input_ptype

        if need_allreduce and ptype:
            output_physical = physical_shapes.get("output", self._get_physical_shape(op.output))
            comm_bytes = math.prod(output_physical) * dtype_size
            comm_type = "allreduce"
            ops.append(CommOp(comm_type, comm_bytes, ptype, direction="forward"))
            logger.debug(
                f"[COMM_DERIVE] op=matmul, "
                f"weight_shardable={weight_shardable}, "
                f"output_shardable={output_shardable}, "
                f"need_allreduce={need_allreduce}, "
                f"ptype={ptype}, "
                f"comm_bytes={comm_bytes / 1e6:.2f}MB, "
                f"comm_type={comm_type}"
            )

        return ops

    def _derive_attention_comm(
        self,
        op: AttentionOp,
        physical_shapes: Dict[str, tuple],
    ) -> List[CommOp]:
        """Derive communication for AttentionOp.

        Cases:
        1. Sequence Parallelism (SP): alltoall on Q/K/V
        """
        ops = []

        q_shardable = op.query.shardable
        k_shardable = op.key.shardable if op.key else {}
        v_shardable = op.value.shardable if op.value else {}

        dtype_size = DTYPE_SIZES.get(op.dtype, 2)

        sp_degree = self.parallel_degrees.get("sp", 1)
        if sp_degree > 1:
            if 2 in q_shardable:
                ptype = q_shardable[2]
                if ptype == "sp":
                    q_physical = physical_shapes.get("query", self._get_physical_shape(op.query))
                    comm_bytes = math.prod(q_physical) * dtype_size
                    comm_type = "alltoall"
                    ops.append(CommOp(comm_type, comm_bytes, ptype, direction="forward"))
                    ops.append(CommOp(comm_type, comm_bytes, ptype, direction="forward"))
                    logger.debug(
                        f"[COMM_DERIVE] op=attention, "
                        f"q_shardable={q_shardable}, "
                        f"ptype={ptype}, "
                        f"comm_bytes={comm_bytes / 1e6:.2f}MB, "
                        f"comm_type={comm_type}, "
                        f"count=2"
                    )

        return ops

    def _derive_embedding_comm(
        self,
        op: EmbeddingOp,
        physical_shapes: Dict[str, tuple],
    ) -> List[CommOp]:
        """Derive communication for EmbeddingOp.

        Cases:
        1. vocab TP sharding: output is partially sharded
           - For inference: need allgather on vocab dimension
           - For training: no immediate comm (handled at next layer)
        """
        ops = []

        weight_shardable = op.weight.shardable
        output_shardable = op.output.shardable

        dtype_size = DTYPE_SIZES.get(op.dtype, 2)

        if 0 in weight_shardable:
            ptype = weight_shardable[0]
            if ptype == "tp":
                output_physical = physical_shapes.get("output", self._get_physical_shape(op.output))
                output_logical = op.output.shape
                tp_degree = self.parallel_degrees.get("tp", 1)

                if output_logical[-1] != output_physical[-1]:
                    comm_bytes = math.prod(output_physical) * dtype_size * tp_degree
                    comm_type = "allgather"
                    ops.append(CommOp(comm_type, comm_bytes, ptype, direction="forward"))
                    logger.debug(
                        f"[COMM_DERIVE] op=embedding, "
                        f"weight_shardable={weight_shardable}, "
                        f"output_shardable={output_shardable}, "
                        f"ptype={ptype}, "
                        f"comm_bytes={comm_bytes / 1e6:.2f}MB, "
                        f"comm_type={comm_type}"
                    )

        return ops

    def _derive_moe_comm(
        self,
        op: MoEExpertOp,
        physical_shapes: Dict[str, tuple],
    ) -> List[CommOp]:
        """Derive communication for MoEExpertOp.

        Cases:
        1. EP sharding: alltoall before and after expert
        """
        ops = []

        expert_shardable = op.expert_gate_weights.shardable

        dtype_size = DTYPE_SIZES.get(op.dtype, 2)

        if 0 in expert_shardable:
            ptype = expert_shardable[0]
            if ptype == "ep":
                hidden_physical = physical_shapes.get("hidden", self._get_physical_shape(op.hidden))
                comm_bytes = math.prod(hidden_physical) * dtype_size
                comm_type = "alltoall"
                ops.append(CommOp(comm_type, comm_bytes, ptype, direction="forward"))
                ops.append(CommOp(comm_type, comm_bytes, ptype, direction="forward"))
                logger.debug(
                    f"[COMM_DERIVE] op=moe, "
                    f"expert_shardable={expert_shardable}, "
                    f"ptype={ptype}, "
                    f"comm_bytes={comm_bytes / 1e6:.2f}MB, "
                    f"comm_type={comm_type}, "
                    f"count=2"
                )

        return ops

    def derive_backward_comm_ops(
        self,
        op: Op,
        physical_shapes: Dict[str, tuple],
    ) -> List[CommOp]:
        """Derive backward communication operations.

        Args:
            op: Operation to analyze
            physical_shapes: Pre-computed physical shapes

        Returns:
            List of backward CommOp
        """
        ops = []

        if isinstance(op, MatmulOp):
            ops.extend(self._derive_matmul_backward_comm(op, physical_shapes))
        elif isinstance(op, MoEExpertOp):
            ops.extend(self._derive_moe_backward_comm(op, physical_shapes))

        return ops

    def _derive_matmul_backward_comm(
        self,
        op: MatmulOp,
        physical_shapes: Dict[str, tuple],
    ) -> List[CommOp]:
        """Derive backward communication for MatmulOp.

        Cases:
        1. DP gradient synchronization: allreduce on weight gradients
        """
        ops = []

        weight = op.weight
        for dim, ptype in weight.shardable.items():
            if ptype == "dp":
                weight_bytes = weight.numel() * DTYPE_SIZES.get(weight.dtype, 2)
                comm_type = "allreduce"
                ops.append(CommOp(comm_type, weight_bytes, ptype, direction="backward"))
                logger.debug(
                    f"[COMM_DERIVE] op=matmul_backward, "
                    f"weight_shardable={weight.shardable}, "
                    f"ptype={ptype}, "
                    f"comm_bytes={weight_bytes / 1e6:.2f}MB, "
                    f"comm_type={comm_type}"
                )

        return ops

    def _derive_moe_backward_comm(
        self,
        op: MoEExpertOp,
        physical_shapes: Dict[str, tuple],
    ) -> List[CommOp]:
        """Derive backward communication for MoEExpertOp.

        Cases:
        1. EP: alltoall for gradient routing
        """
        ops = []

        expert_shardable = op.expert_gate_weights.shardable
        if 0 in expert_shardable:
            ptype = expert_shardable[0]
            if ptype == "ep":
                hidden_physical = physical_shapes.get("hidden", self._get_physical_shape(op.hidden))
                dtype_size = DTYPE_SIZES.get(op.dtype, 2)
                comm_bytes = math.prod(hidden_physical) * dtype_size
                comm_type = "alltoall"
                ops.append(CommOp(comm_type, comm_bytes, ptype, direction="backward"))
                logger.debug(
                    f"[COMM_DERIVE] op=moe_backward, "
                    f"expert_shardable={expert_shardable}, "
                    f"ptype={ptype}, "
                    f"comm_bytes={comm_bytes / 1e6:.2f}MB, "
                    f"comm_type={comm_type}"
                )

        return ops

    def _get_physical_shape(self, tensor: "ShardedTensor") -> tuple:
        """Get physical shape for a tensor."""
        return tensor.get_physical_shape(self.parallel_degrees)

    def get_degree(self, ptype: str) -> int:
        """Get parallel degree for a parallel type."""
        return self.parallel_degrees.get(ptype, 1)
