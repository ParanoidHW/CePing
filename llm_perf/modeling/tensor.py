"""ShardedTensor - Tensor with sharding constraints.

Similar to torch.Tensor, but with automatic sharding constraint derivation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, List, Any
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class ShardedTensor:
    """Tensor with sharding constraints, similar to torch.Tensor.

    Core features:
    1. Like torch.Tensor operations (matmul, view, transpose)
    2. Automatic derivation of output shape and sharding constraints
    3. Record operation history for FLOPs and communication derivation

    Attributes:
        shape: Logical shape (unsharded)
        shardable: Sharding constraints for each dimension {dim_idx: parallel_type}
        dtype: Data type (fp16, bf16, fp32, etc.)
        name: Optional tensor name
        _op_history: Operation history (for FLOPs derivation)
        _is_view: Whether this tensor is a view (no new memory allocation)
    """

    shape: Tuple[int, ...]
    shardable: Dict[int, str] = field(default_factory=dict)
    dtype: str = "fp16"
    name: Optional[str] = None
    _op_history: List[Any] = field(default_factory=list)
    _is_view: bool = False

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    def size(self, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
        """Similar to torch.Tensor.size()."""
        if dim is None:
            return self.shape
        if dim < 0:
            dim = self.ndim + dim
        return self.shape[dim]

    def numel(self) -> int:
        """Total number of elements."""
        return math.prod(self.shape)

    def __matmul__(self, other: "ShardedTensor") -> "ShardedTensor":
        """Matrix multiplication: self @ other.

        Automatic derivation:
        1. Output shape: (..., self.shape[-2], other.shape[-1])
        2. Output sharding constraints: derived from input constraints

        Sharding derivation rules:
        - A(m, k) shardable={0: "tp"} @ B(k, n)
          -> output(m, n) shardable={}, needs AllReduce
        - A(m, k) @ B(k, n) shardable={1: "tp"}
          -> output(m, n) shardable={1: "tp"}, no communication
        - A(m, k) shardable={1: "tp"} @ B(k, n) shardable={0: "tp"}
          -> output(m, n) shardable={}, needs AllReduce
        """
        from llm_perf.kernels.op import MatmulOp

        assert self.shape[-1] == other.shape[-2], f"Dimension mismatch: {self.shape[-1]} vs {other.shape[-2]}"

        output_shape = (*self.shape[:-1], other.shape[-1])
        output_shardable = self._infer_matmul_shardable(other)

        output = ShardedTensor(
            shape=output_shape,
            shardable=output_shardable,
            dtype=self.dtype,
            name=f"{self.name or 'input'}_matmul_{other.name or 'weight'}",
        )

        output._op_history = self._op_history + [MatmulOp(dtype=self.dtype, input=self, weight=other, output=output)]

        return output

    def _infer_matmul_shardable(self, other: "ShardedTensor") -> Dict[int, str]:
        """Derive output sharding constraints for matmul."""
        result = {}

        self_last_dim = self.ndim - 1
        other_last_dim = other.ndim - 1

        if len(self.shape) == 2 and len(other.shape) == 2:
            if 0 in self.shardable:
                return {}
            if 1 in other.shardable:
                return {1: other.shardable[1]}
            if 1 in self.shardable and 0 in other.shardable:
                if self.shardable[1] == other.shardable[0]:
                    return {}
                result[1] = other.shardable[0]

        for dim, ptype in self.shardable.items():
            if dim < self_last_dim:
                result[dim] = ptype

        for dim, ptype in other.shardable.items():
            if dim == other_last_dim:
                output_dim = self.ndim - 1
                if output_dim not in result:
                    result[output_dim] = ptype

        return result

    def view(self, *shape) -> "ShardedTensor":
        """Reshape tensor, similar to torch.Tensor.view().

        Sharding constraints follow dimension mapping.
        """
        from llm_perf.kernels.op import ViewOp

        new_numel = math.prod(shape)
        assert self.numel() == new_numel, f"Cannot reshape {self.shape} to {shape}: {self.numel()} vs {new_numel}"

        new_shardable = self._infer_view_shardable(shape)

        output = ShardedTensor(
            shape=shape,
            shardable=new_shardable,
            dtype=self.dtype,
            name=f"{self.name or 'tensor'}_view",
        )

        output._op_history = self._op_history + [ViewOp(dtype=self.dtype, input=self, shape=shape, output=output)]
        output._is_view = True

        return output

    def _infer_view_shardable(self, new_shape: Tuple[int, ...]) -> Dict[int, str]:
        """Derive new sharding constraints after reshape."""
        if not self.shardable:
            return {}

        result = {}
        old_shape = self.shape

        new_cumprod = []
        cumprod = 1
        for s in new_shape:
            cumprod *= s
            new_cumprod.append(cumprod)

        old_cumprod = []
        cumprod = 1
        for s in old_shape:
            cumprod *= s
            old_cumprod.append(cumprod)

        for old_dim, ptype in self.shardable.items():
            old_start = old_cumprod[old_dim - 1] if old_dim > 0 else 0
            old_end = old_cumprod[old_dim]

            for new_dim in range(len(new_shape)):
                new_start = new_cumprod[new_dim - 1] if new_dim > 0 else 0
                new_end = new_cumprod[new_dim]

                if old_start >= new_start and old_end <= new_end:
                    if new_dim not in result:
                        result[new_dim] = ptype
                    break

        return result

    def transpose(self, dim0: int, dim1: int) -> "ShardedTensor":
        """Transpose two dimensions, similar to torch.Tensor.transpose().

        Sharding constraints swap with dimensions.
        """
        from llm_perf.kernels.op import TransposeOp

        if dim0 < 0:
            dim0 = self.ndim + dim0
        if dim1 < 0:
            dim1 = self.ndim + dim1

        new_shape = list(self.shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]

        new_shardable = {}
        for dim, ptype in self.shardable.items():
            if dim == dim0:
                new_shardable[dim1] = ptype
            elif dim == dim1:
                new_shardable[dim0] = ptype
            else:
                new_shardable[dim] = ptype

        output = ShardedTensor(
            shape=tuple(new_shape),
            shardable=new_shardable,
            dtype=self.dtype,
            name=f"{self.name or 'tensor'}_transpose",
        )

        output._op_history = self._op_history + [
            TransposeOp(dtype=self.dtype, input=self, dim0=dim0, dim1=dim1, output=output)
        ]
        output._is_view = True

        return output

    def __add__(self, other: "ShardedTensor") -> "ShardedTensor":
        """Element-wise addition.

        Sharding constraints must match.

        Op history: merge both tensor's op histories (take the longer one).
        """
        assert self.shape == other.shape, f"Shape mismatch: {self.shape} vs {other.shape}"

        output = ShardedTensor(
            shape=self.shape,
            shardable=self.shardable,
            dtype=self.dtype,
            name=f"{self.name or 'tensor'}_add",
        )

        output._op_history = other._op_history if len(other._op_history) > len(self._op_history) else self._op_history

        return output

    def __mul__(self, other: "ShardedTensor") -> "ShardedTensor":
        """Element-wise multiplication.

        Sharding constraints must match.

        Op history: merge both tensor's op histories (take the longer one).
        """
        assert self.shape == other.shape, f"Shape mismatch: {self.shape} vs {other.shape}"

        output = ShardedTensor(
            shape=self.shape,
            shardable=self.shardable,
            dtype=self.dtype,
            name=f"{self.name or 'tensor'}_mul",
        )

        output._op_history = other._op_history if len(other._op_history) > len(self._op_history) else self._op_history

        return output

    def get_physical_shape(self, parallel_degrees: Dict[str, int]) -> Tuple[int, ...]:
        """Get physical shape given parallel degrees.

        Args:
            parallel_degrees: {parallel_type: degree}

        Returns:
            Physical shape after sharding
        """
        shape = []
        for dim, size in enumerate(self.shape):
            if dim in self.shardable:
                ptype = self.shardable[dim]
                degree = parallel_degrees.get(ptype, 1)
                shape.append(max(1, size // degree))
            else:
                shape.append(size)
        physical_shape = tuple(shape)
        logger.info(
            f"[PHYSICAL_SHAPE] tensor={self.name or 'unnamed'}, "
            f"logical={self.shape}, shardable={self.shardable}, "
            f"parallel_degrees={parallel_degrees}, "
            f"physical={physical_shape}, dtype={self.dtype}"
        )
        return physical_shape

    def get_physical_numel(self, parallel_degrees: Dict[str, int]) -> int:
        """Get physical number of elements after sharding."""
        return math.prod(self.get_physical_shape(parallel_degrees))

    def get_physical_bytes(self, parallel_degrees: Dict[str, int]) -> int:
        """Get physical memory bytes after sharding.

        Args:
            parallel_degrees: {parallel_type: degree}

        Returns:
            Physical memory bytes (numel * dtype_size)
        """
        from llm_perf.utils.constants import DTYPE_SIZES

        dtype_size = DTYPE_SIZES.get(self.dtype, 2)
        return self.get_physical_numel(parallel_degrees) * dtype_size

    def contiguous(self) -> "ShardedTensor":
        """Return a contiguous (non-view) copy of the tensor.

        In PyTorch, contiguous() ensures the tensor has a contiguous memory layout.
        Here, we mark the output as non-view (allocates new memory).

        Returns:
            A new ShardedTensor with _is_view=False
        """
        output = ShardedTensor(
            shape=self.shape,
            shardable=self.shardable,
            dtype=self.dtype,
            name=f"{self.name or 'tensor'}_contiguous",
        )
        output._op_history = self._op_history
        output._is_view = False
        return output

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output."""
        return {
            "shape": self.shape,
            "shardable": self.shardable,
            "dtype": self.dtype,
            "name": self.name,
            "numel": self.numel(),
        }


class ShardedParameter(ShardedTensor):
    """Model weight parameter, inherits all attributes and methods from ShardedTensor.

    Used to mark a tensor as a model parameter (weight), not activation tensor.
    Only ShardedParameter is registered to _weights in __setattr__.

    Reference: torch.nn.Parameter design pattern.
    """

    pass
