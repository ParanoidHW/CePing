"""ShardedModule - Module base class, similar to torch.nn.Module.

Provides automatic registration of submodules and weights,
and supports forward/backward mode for performance estimation.
"""

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from llm_perf.utils.constants import DTYPE_SIZES

from .comm_deriver import CommPatternDeriver
from .tensor import ShardedTensor

if TYPE_CHECKING:
    from llm_perf.kernels.op import CommOp
    from llm_perf.strategy.parallel_context import ParallelContext
    from llm_perf.strategy.pp_model import PPModel

logger = logging.getLogger(__name__)


class ShardedModule:
    """Module base class, similar to torch.nn.Module.

    Core features:
    1. Like torch.nn.Module for defining submodules and forward method
    2. Automatic management of weights, sharding constraints, activations
    3. bind(ctx) returns ModuleInstance (runtime physical shape)
    4. Support forward/backward mode for FLOPs derivation

    Attributes:
        _submodules: Submodules dict {name: ShardedModule}
        _weights: Weight tensors dict {name: ShardedTensor}
        _activations: Activation tensors dict {name: ShardedTensor}
        _name: Module name
    """

    def __init__(self):
        self._submodules: Dict[str, ShardedModule] = {}
        self._weights: Dict[str, ShardedTensor] = {}
        self._activations: Dict[str, ShardedTensor] = {}
        self._intermediate_tensors: Dict[str, ShardedTensor] = {}
        self._name: str = ""
        self._last_forward_input: Optional[ShardedTensor] = None
        self._last_forward_output: Optional[ShardedTensor] = None

    def __setattr__(self, name: str, value: Any):
        """Auto-register submodules and weights."""
        if isinstance(value, ShardedModule):
            self._submodules[name] = value
            value._name = name
        elif isinstance(value, ShardedTensor):
            self._weights[name] = value
        elif isinstance(value, list) and all(isinstance(v, ShardedModule) for v in value):
            for i, v in enumerate(value):
                v._name = f"{name}.{i}"
                self._submodules[f"{name}.{i}"] = v
        super().__setattr__(name, value)

    def forward(self, *args, **kwargs) -> ShardedTensor:
        """Forward method, must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement forward()")

    def __call__(self, *args, **kwargs) -> ShardedTensor:
        """Call forward, auto-record input/output."""
        if args and isinstance(args[0], ShardedTensor):
            self._last_forward_input = args[0]
        output = self.forward(*args, **kwargs)
        self._last_forward_output = output
        return output

    def _track_intermediate(self, name: str, tensor: ShardedTensor) -> ShardedTensor:
        """Track intermediate tensor during forward pass.

        Args:
            name: Tensor name (relative to this module)
            tensor: ShardedTensor to track

        Returns:
            The same tensor (for chaining)

        Usage:
            q_proj = self._track_intermediate("q_proj", hidden @ self.q_weight)
        """
        full_name = f"{self._name}.{name}" if self._name else name
        self._intermediate_tensors[full_name] = tensor
        return tensor

    def bind(
        self,
        ctx: "ParallelContext",
        pp_strategy: Optional["PPStrategy"] = None,
        pp_stage: Optional[int] = None,
        mode: str = "forward_backward",
    ) -> Union["ModuleInstance", "PPModel"]:
        """Bind to ParallelContext, return ModuleInstance or PPModel.

        Args:
            ctx: ParallelContext
            pp_strategy: Optional PP strategy (creates PPModel if provided)
            pp_stage: PP stage index (optional, used without pp_strategy)
            mode: "forward" or "forward_backward"

        Returns:
            If pp_strategy is None: ModuleInstance
            If pp_strategy is provided: PPModel
        """
        if pp_strategy is None:
            return ModuleInstance(self, ctx, pp_stage=pp_stage, mode=mode)
        else:
            from llm_perf.strategy.pp_model import PPModel

            return PPModel(self, pp_strategy)

    def get_weights(self) -> Dict[str, ShardedTensor]:
        """Get all weights (including submodules)."""
        weights = self._weights.copy()
        for name, submodule in self._submodules.items():
            for w_name, w_tensor in submodule.get_weights().items():
                weights[f"{name}.{w_name}"] = w_tensor
        return weights

    def get_activations(self) -> Dict[str, ShardedTensor]:
        """Get all activations (including submodules and intermediate tensors)."""
        activations = self._activations.copy()
        activations.update(self._intermediate_tensors)
        for name, submodule in self._submodules.items():
            for a_name, a_tensor in submodule.get_activations().items():
                activations[f"{name}.{a_name}"] = a_tensor
        return activations

    def params_count(self) -> int:
        """Total parameters (logical, unsharded)."""
        return sum(w.numel() for w in self.get_weights().values())

    def params_count_breakdown(self) -> Dict[str, int]:
        """Parameters breakdown by layer."""
        breakdown = {}
        for name, weight in self._weights.items():
            breakdown[name] = weight.numel()
        for sub_name, submodule in self._submodules.items():
            for w_name, count in submodule.params_count_breakdown().items():
                breakdown[f"{sub_name}.{w_name}"] = count
        return breakdown

    def activation_memory_logical(self, batch_size: int = 1) -> int:
        """Activation memory (logical, unsharded) for training."""
        activations = self.get_activations()
        total = sum(a.numel() * DTYPE_SIZES.get(a.dtype, 2) * batch_size for a in activations.values())
        return total

    def activation_memory_max_layer(self, batch_size: int = 1) -> int:
        """Activation memory (inference) - max single layer."""
        activations = self.get_activations()
        if not activations:
            return 0
        max_activation = max(a.numel() * DTYPE_SIZES.get(a.dtype, 2) * batch_size for a in activations.values())
        return max_activation

    def flops_forward(self, batch_size: int = 1, seq_len: int = 1) -> int:
        """Forward FLOPs (logical, unsharded)."""
        if self._last_forward_output is None:
            return 0
        return self._compute_flops_from_history(self._last_forward_output._op_history)

    def flops_backward(self, batch_size: int = 1, seq_len: int = 1) -> int:
        """Backward FLOPs (logical, unsharded).

        Rule: backward ≈ 2x forward
        """
        return self.flops_forward(batch_size, seq_len) * 2

    def flops_total(self, batch_size: int = 1, seq_len: int = 1) -> int:
        """Total FLOPs (forward + backward)."""
        return self.flops_forward(batch_size, seq_len) + self.flops_backward(batch_size, seq_len)

    def _compute_flops_from_history(self, op_history: List[Any]) -> int:
        """Compute FLOPs from operation history."""
        total_flops = 0
        for op in op_history:
            flops = self._compute_op_flops(op)
            total_flops += flops
        return total_flops

    def _compute_op_flops(self, op: Any) -> int:
        """Compute FLOPs for a single operation."""
        from llm_perf.kernels.functional import flash_attention, linear, rms_norm, silu
        from llm_perf.kernels.op import ActivationOp, AttentionOp, EmbeddingOp, MatmulOp, RMSNormOp

        try:
            if isinstance(op, MatmulOp):
                input_shape = op.input.shape
                weight_shape = op.weight.shape
                result = linear(input_shape, weight_shape, dtype=op.dtype)
                return result.flops
            elif isinstance(op, AttentionOp):
                q_shape = op.query.shape
                k_shape = op.key.shape
                v_shape = op.value.shape
                result = flash_attention(q_shape, k_shape, v_shape, dtype=op.dtype)
                return result.flops
            elif isinstance(op, RMSNormOp):
                input_shape = op.input.shape
                result = rms_norm(input_shape, dtype=op.dtype)
                return result.flops
            elif isinstance(op, ActivationOp):
                input_shape = op.input.shape
                if op.activation_type == "silu":
                    result = silu(input_shape, dtype=op.dtype)
                    return result.flops
            elif isinstance(op, EmbeddingOp):
                from llm_perf.kernels.functional import embedding

                vocab_size = op.weight.shape[0]
                embedding_dim = op.weight.shape[1]
                input_shape = op.input_ids.shape
                result = embedding(vocab_size, embedding_dim, input_shape, dtype=op.dtype)
                return result.flops
        except Exception:
            pass

        return 0


class ModuleInstance:
    """Module instance bound to ParallelContext.

    Attributes:
        module: Original ShardedModule
        ctx: ParallelContext
        pp_stage: PP stage index
        mode: "forward" or "forward_backward"
        _submodule_instances: Submodule instances
        _weight_instances: Weight physical instances
        _activation_instances: Activation physical instances
    """

    def __init__(
        self,
        module: ShardedModule,
        ctx: "ParallelContext",
        pp_stage: Optional[int] = None,
        mode: str = "forward_backward",
    ):
        self.module = module
        self.ctx = ctx
        self.pp_stage = pp_stage
        self.mode = mode

        self._parallel_degrees = self._get_parallel_degrees()
        self._comm_deriver = CommPatternDeriver(self._parallel_degrees)
        self._physical_shape_cache: Dict[str, Tuple[int, ...]] = {}

        self._submodule_instances: Dict[str, ModuleInstance] = {}
        for name, submodule in module._submodules.items():
            self._submodule_instances[name] = ModuleInstance(submodule, ctx, pp_stage=pp_stage, mode=mode)

        self._weight_instances: Dict[str, WeightInstance] = {}
        for name, weight in module.get_weights().items():
            self._weight_instances[name] = WeightInstance(weight, ctx)

        self._activation_instances: Dict[str, ActivationInstance] = {}
        for name, activation in module.get_activations().items():
            self._activation_instances[name] = ActivationInstance(activation, ctx)

        logger.info(
            f"[BIND] module={module.__class__.__name__}, "
            f"tp={ctx.tp_degree}, pp={ctx.pp_degree}, dp={ctx.dp_degree}, "
            f"ep={ctx.ep_degree}, sp={ctx.sp_degree}, mode={mode}, "
            f"logical_params={module.params_count() / 1e9:.2f}B, "
            f"physical_params={self.params_count_physical / 1e9:.2f}B, "
            f"weight_mem={self.weight_memory_physical / 1e9:.2f}GB, "
            f"num_weights={len(self._weight_instances)}, "
            f"num_submodules={len(self._submodule_instances)}"
        )

    @property
    def params_count_physical(self) -> int:
        """Physical parameters (sharded) - per GPU."""
        return sum(w.physical_numel for w in self._weight_instances.values())

    @property
    def params_count_logical(self) -> int:
        """Logical parameters (unsharded) - total."""
        return self.module.params_count()

    @property
    def flops_forward_physical(self) -> int:
        """Forward FLOPs (physical, sharded).

        Only counts current module's ops (not including submodule ops).
        Total FLOPs for a model should be computed from _last_forward_output._op_history
        which contains the complete forward pass history.
        """
        if self.module._last_forward_output is None:
            return 0

        op_history = self.module._last_forward_output._op_history
        total_flops = 0
        for op in op_history:
            total_flops += self._infer_physical_flops(op)

        self._flops_forward_physical = total_flops
        logger.info(
            f"[FLOPS] module={self.module.__class__.__name__}, "
            f"op_count={len(op_history)}, "
            f"flops_forward={self._flops_forward_physical / 1e12:.4f}T, "
            f"mode={self.mode}"
        )

        return total_flops

    @property
    def flops_backward_physical(self) -> int:
        """Backward FLOPs (physical, sharded)."""
        return self.flops_forward_physical * 2

    @property
    def flops_total_physical(self) -> int:
        """Total FLOPs (forward + backward, physical)."""
        if self.mode == "forward":
            return self.flops_forward_physical
        return self.flops_forward_physical + self.flops_backward_physical

    @property
    def activation_memory_physical(self) -> int:
        """Activation memory (physical, sharded).

        Uses intermediate tensors tracked during forward pass.
        - Training (forward_backward): all non-view intermediate tensors
        - Inference (forward): only max single layer
        """
        if self.mode == "forward":
            if not self._activation_instances:
                return 0
            return max(a.physical_bytes for a in self._activation_instances.values())
        else:
            # Use intermediate tensors tracked by _track_intermediate
            # Filter out view tensors (no new memory allocation)
            parallel_degrees = self._get_parallel_degrees()
            total = 0

            for name, activation in self.module._intermediate_tensors.items():
                # Skip view tensors
                if hasattr(activation, "_is_view") and activation._is_view:
                    continue
                if hasattr(activation, "get_physical_bytes"):
                    total += activation.get_physical_bytes(parallel_degrees)

            # Add submodule activation memory
            for sub_inst in self._submodule_instances.values():
                total += sub_inst.activation_memory_physical

            # Apply activation checkpointing ratio if enabled
            if hasattr(self.ctx, "activation_checkpointing") and self.ctx.activation_checkpointing:
                ratio = getattr(self.ctx, "activation_checkpointing_ratio", 1)
                total = total // ratio

            self._activation_memory_physical = total
            logger.info(
                f"[ACTIVATION_MEM] module={self.module.__class__.__name__}, "
                f"mode={self.mode}, num_intermediate={len(self.module._intermediate_tensors)}, "
                f"activation_mem={self._activation_memory_physical / 1e9:.4f}GB"
            )

            return total

    @property
    def weight_memory_physical(self) -> int:
        """Weight memory (physical, sharded) in bytes.

        Weight memory = params_count_physical × dtype_size
        """
        from llm_perf.utils.constants import DTYPE_SIZES

        dtype_size = DTYPE_SIZES.get(self.ctx.dtype, 2)
        return self.params_count_physical * dtype_size

    @property
    def gradient_memory_physical(self) -> int:
        """Gradient memory (physical, sharded) in bytes.

        Gradient memory = params_count_physical × dtype_size
        Only applicable for training (forward_backward mode).
        """
        if self.mode != "forward_backward":
            return 0

        from llm_perf.utils.constants import DTYPE_SIZES

        dtype_size = DTYPE_SIZES.get(self.ctx.dtype, 2)
        return self.params_count_physical * dtype_size

    @property
    def optimizer_memory_physical(self) -> int:
        """Optimizer state memory (physical, sharded) in bytes.

        For Adam optimizer:
        - Momentum (FP32): params_count × 4 bytes
        - Velocity (FP32): params_count × 4 bytes
        Total: params_count × 8 bytes

        ZeRO stage affects optimizer memory:
        - ZeRO-0: Full optimizer states on each GPU
        - ZeRO-1: Optimizer states sharded across DP
        - ZeRO-2: Optimizer + gradients sharded
        - ZeRO-3: Optimizer + gradients + weights sharded
        """
        if self.mode != "forward_backward":
            return 0

        zero_stage = getattr(self.ctx, "zero_stage", 0)
        dp_degree = self.ctx.dp_degree

        # Adam optimizer: 2 × FP32 states
        optimizer_multiplier = 8  # 2 × 4 bytes (FP32)

        base_memory = self.params_count_physical * optimizer_multiplier

        # Apply ZeRO sharding
        if zero_stage == 0:
            self._optimizer_memory_physical = base_memory
        elif zero_stage >= 1:
            self._optimizer_memory_physical = base_memory // dp_degree
        else:
            self._optimizer_memory_physical = base_memory

        logger.info(
            f"[OPTIMIZER_MEM] params_physical={self.params_count_physical / 1e9:.2f}B, "
            f"zero_stage={self.ctx.zero_stage}, dp={self.ctx.dp_degree}, "
            f"optimizer_mem={self._optimizer_memory_physical / 1e9:.4f}GB"
        )

        return self._optimizer_memory_physical

    @property
    def total_comm_ops(self) -> List["CommOp"]:
        """Total communication operations (forward + backward).

        Includes both own ops and submodule ops.
        """
        ops = []
        for inst in self._submodule_instances.values():
            ops.extend(inst.total_comm_ops)

        if self.module._last_forward_output:
            for op in self.module._last_forward_output._op_history:
                comm_ops = self._infer_comm_ops(op)
                ops.extend(comm_ops)

                if self.mode == "forward_backward":
                    backward_comm_ops = self._infer_backward_comm_ops(op)
                    ops.extend(backward_comm_ops)

        return ops

    @property
    def own_comm_ops(self) -> List["CommOp"]:
        """Own communication operations (excluding submodules).

        Note: Due to op_history structure, this currently returns
        ops from this module's direct operations. For modules with
        submodules, some ops may be double-counted.
        """
        ops = []

        if self.module._last_forward_output:
            for op in self.module._last_forward_output._op_history:
                comm_ops = self._infer_comm_ops(op)
                ops.extend(comm_ops)

                if self.mode == "forward_backward":
                    backward_comm_ops = self._infer_backward_comm_ops(op)
                    ops.extend(backward_comm_ops)

        return ops

    @property
    def kv_cache_memory(self) -> int:
        """KV cache memory for inference (bytes).

        Only relevant for forward mode inference with KV cache.
        Returns 0 for training (forward_backward mode).
        """
        if self.mode != "forward":
            return 0

        total = 0
        if self.module._last_forward_output:
            from llm_perf.kernels.op import AttentionOp

            for op in self.module._last_forward_output._op_history:
                if isinstance(op, AttentionOp):
                    total += op.kv_cache_memory()

        return total

    def _infer_physical_flops(self, op: Any) -> int:
        """Derive physical FLOPs from operation."""
        from llm_perf.kernels.functional import flash_attention, linear, rms_norm, silu
        from llm_perf.kernels.op import ActivationOp, AttentionOp, EmbeddingOp, MatmulOp, RMSNormOp

        try:
            if isinstance(op, MatmulOp):
                input_physical = self._infer_physical_shape(op.input)
                weight_physical = self._infer_physical_shape(op.weight)
                result = linear(input_physical, weight_physical, dtype=op.dtype)
                flops = result.flops
                logger.debug(
                    f"[OP_FLOPS] op_type={op.__class__.__name__}, kernel={result.kernel_name}, flops={flops / 1e9:.2f}G"
                )
                return flops
            elif isinstance(op, AttentionOp):
                q_physical = self._infer_physical_shape(op.query)
                k_physical = self._infer_physical_shape(op.key)
                v_physical = self._infer_physical_shape(op.value)
                result = flash_attention(q_physical, k_physical, v_physical, dtype=op.dtype)
                flops = result.flops
                logger.debug(
                    f"[OP_FLOPS] op_type={op.__class__.__name__}, kernel={result.kernel_name}, flops={flops / 1e9:.2f}G"
                )
                return flops
            elif isinstance(op, RMSNormOp):
                input_physical = self._infer_physical_shape(op.input)
                result = rms_norm(input_physical, dtype=op.dtype)
                flops = result.flops
                logger.debug(
                    f"[OP_FLOPS] op_type={op.__class__.__name__}, kernel={result.kernel_name}, flops={flops / 1e9:.2f}G"
                )
                return flops
            elif isinstance(op, ActivationOp):
                input_physical = self._infer_physical_shape(op.input)
                if op.activation_type == "silu":
                    result = silu(input_physical, dtype=op.dtype)
                    flops = result.flops
                    logger.debug(
                        f"[OP_FLOPS] op_type={op.__class__.__name__}, "
                        f"kernel={result.kernel_name}, flops={flops / 1e9:.2f}G"
                    )
                    return flops
            elif isinstance(op, EmbeddingOp):
                from llm_perf.kernels.functional import embedding

                vocab_physical = self._infer_physical_shape(op.weight)
                vocab_size = vocab_physical[0]
                embedding_dim = vocab_physical[1]
                input_shape = self._infer_physical_shape(op.input_ids)
                result = embedding(vocab_size, embedding_dim, input_shape, dtype=op.dtype)
                flops = result.flops
                logger.debug(
                    f"[OP_FLOPS] op_type={op.__class__.__name__}, kernel={result.kernel_name}, flops={flops / 1e9:.2f}G"
                )
                return flops
        except Exception:
            pass

        return 0

    def _infer_physical_shape(self, tensor: ShardedTensor) -> Tuple[int, ...]:
        """Derive physical shape from tensor."""
        parallel_degrees = self._get_parallel_degrees()
        return tensor.get_physical_shape(parallel_degrees)

    def _get_parallel_degrees(self) -> Dict[str, int]:
        """Get parallel degrees from context."""
        return {
            "tp": getattr(self.ctx, "tp_degree", 1),
            "sp": getattr(self.ctx, "sp_degree", 1),
            "ep": getattr(self.ctx, "ep_degree", 1),
            "dp": getattr(self.ctx, "dp_degree", 1),
            "pp": getattr(self.ctx, "pp_degree", 1),
        }

    def _infer_comm_ops(self, op: Any) -> List["CommOp"]:
        """Derive forward communication from operation.

        Uses CommPatternDeriver for systematic derivation.
        """
        physical_shapes = self._get_op_physical_shapes(op)
        return self._comm_deriver.derive_comm_ops(op, physical_shapes)

    def _get_op_physical_shapes(self, op: Any) -> Dict[str, tuple]:
        """Get physical shapes for all tensors in an op."""
        shapes = {}

        if hasattr(op, "input"):
            shapes["input"] = self._infer_physical_shape(op.input)
        if hasattr(op, "weight"):
            shapes["weight"] = self._infer_physical_shape(op.weight)
        if hasattr(op, "output"):
            shapes["output"] = self._infer_physical_shape(op.output)
        if hasattr(op, "query"):
            shapes["query"] = self._infer_physical_shape(op.query)
        if hasattr(op, "key") and op.key:
            shapes["key"] = self._infer_physical_shape(op.key)
        if hasattr(op, "value") and op.value:
            shapes["value"] = self._infer_physical_shape(op.value)
        if hasattr(op, "hidden"):
            shapes["hidden"] = self._infer_physical_shape(op.hidden)

        return shapes

    def _infer_backward_comm_ops(self, op: Any) -> List["CommOp"]:
        """Derive backward communication from operation.

        Uses CommPatternDeriver for systematic derivation.
        """
        physical_shapes = self._get_op_physical_shapes(op)
        return self._comm_deriver.derive_backward_comm_ops(op, physical_shapes)

    def estimate_memory(self, batch_size: int = 1) -> int:
        """Estimate memory (params + activations + optimizer states)."""
        dtype_size = DTYPE_SIZES.get(self.ctx.dtype if hasattr(self.ctx, "dtype") else "fp16", 2)

        param_memory = self.params_count_physical * dtype_size
        activation_memory = self.activation_memory_physical * batch_size

        optimizer_memory = 0
        if self.mode == "forward_backward":
            zero_stage = getattr(self.ctx, "zero_stage", 0)
            dp_degree = getattr(self.ctx, "dp_degree", 1)

            if zero_stage == 0:
                optimizer_memory = param_memory * 2
            elif zero_stage == 1:
                optimizer_memory = param_memory * 2 // max(dp_degree, 1)
            elif zero_stage == 2:
                optimizer_memory = param_memory * 2 // max(dp_degree, 1)
            elif zero_stage == 3:
                optimizer_memory = 0

        comm_buffer = sum(op.data_bytes for op in self.total_comm_ops) * 0.1

        total = param_memory + activation_memory + optimizer_memory + comm_buffer
        total = int(total * 1.15)

        return total

    def estimate_time(self, backend: Any) -> float:
        """Estimate time (compute + comm).

        Only counts current module's ops (not including submodule ops).
        Total time for a model should be computed from _last_forward_output._op_history
        which contains the complete forward pass history.

        Args:
            backend: Kernel backend

        Returns:
            Total time in seconds
        """
        compute_time = 0.0

        if self.module._last_forward_output:
            for op in self.module._last_forward_output._op_history:
                physical_inputs = [self._infer_physical_shape(t) for t in op.inputs]
                physical_output = self._infer_physical_shape(op.output)

                try:
                    device = getattr(backend.config, "device", None)
                    compute_time += backend.estimate_compute_time(
                        op.kernel_name,
                        physical_inputs,
                        physical_output,
                        op.dtype,
                        device,
                    )
                except Exception:
                    pass

        if self.mode == "forward_backward":
            compute_time *= 2

        comm_time = 0.0
        for comm_op in self.total_comm_ops:
            try:
                domain = self.ctx.get_comm_domain(comm_op.ptype) if hasattr(self.ctx, "get_comm_domain") else None
                if domain:
                    bandwidth = getattr(domain, "bandwidth_gbps", 100.0)
                    num_ranks = len(getattr(domain, "ranks", [0, 1]))
                else:
                    bandwidth = 100.0
                    degree = self._get_parallel_degrees().get(comm_op.ptype, 1)
                    num_ranks = max(degree, 1)

                comm_time += backend.estimate_comm_time(
                    comm_op.comm_type,
                    comm_op.data_bytes,
                    num_ranks,
                    bandwidth,
                )
            except Exception:
                pass

        return compute_time + comm_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output."""
        dtype_size = DTYPE_SIZES.get(self.ctx.dtype if hasattr(self.ctx, "dtype") else "fp16", 2)

        return {
            "module_name": self.module._name,
            "pp_stage": self.pp_stage,
            "mode": self.mode,
            "params": {
                "logical": self.params_count_logical,
                "physical": self.params_count_physical,
                "logical_gb": self.params_count_logical * dtype_size / 1e9,
                "physical_gb": self.params_count_physical * dtype_size / 1e9,
            },
            "flops": {
                "forward_physical": self.flops_forward_physical,
                "backward_physical": self.flops_backward_physical,
                "total_physical": self.flops_total_physical,
            },
            "activation": {
                "physical_bytes": self.activation_memory_physical,
                "physical_gb": self.activation_memory_physical / 1e9,
            },
            "memory": {
                "estimate_bytes": self.estimate_memory(),
                "estimate_gb": self.estimate_memory() / 1e9,
            },
            "communication": {
                "total_ops": len(self.total_comm_ops),
                "ops_breakdown": [op.to_dict() for op in self.total_comm_ops],
            },
            "submodules": {name: inst.to_dict() for name, inst in self._submodule_instances.items()},
        }


class WeightInstance:
    """Weight instance bound to ParallelContext."""

    def __init__(self, weight: ShardedTensor, ctx: "ParallelContext"):
        self.weight = weight
        self.ctx = ctx

    @property
    def physical_shape(self) -> Tuple[int, ...]:
        parallel_degrees = {
            "tp": getattr(self.ctx, "tp_degree", 1),
            "sp": getattr(self.ctx, "sp_degree", 1),
            "ep": getattr(self.ctx, "ep_degree", 1),
            "dp": getattr(self.ctx, "dp_degree", 1),
        }
        return self.weight.get_physical_shape(parallel_degrees)

    @property
    def physical_numel(self) -> int:
        return math.prod(self.physical_shape)

    @property
    def physical_bytes(self) -> int:
        """Physical memory bytes after sharding."""
        from llm_perf.utils.constants import DTYPE_SIZES

        dtype_size = DTYPE_SIZES.get(self.weight.dtype, 2)
        return self.physical_numel * dtype_size


class ActivationInstance:
    """Activation instance bound to ParallelContext."""

    def __init__(self, activation: ShardedTensor, ctx: "ParallelContext"):
        self.activation = activation
        self.ctx = ctx

    @property
    def physical_shape(self) -> Tuple[int, ...]:
        parallel_degrees = {
            "tp": getattr(self.ctx, "tp_degree", 1),
            "sp": getattr(self.ctx, "sp_degree", 1),
            "ep": getattr(self.ctx, "ep_degree", 1),
        }
        return self.activation.get_physical_shape(parallel_degrees)

    @property
    def physical_bytes(self) -> int:
        dtype_size = DTYPE_SIZES.get(self.activation.dtype, 2)
        return math.prod(self.physical_shape) * dtype_size
