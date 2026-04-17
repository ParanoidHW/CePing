"""Chunking feature for sequence chunking memory optimization."""

from typing import Dict, Any, Optional

from ..base import SchedulerFeature, SchedulerConfig


class ChunkingFeature(SchedulerFeature):
    """Models sequence chunking for memory optimization.

    Chunking (also called gradient checkpointing or sequence parallelism)
    splits long sequences into chunks that fit in memory.

    Benefits:
    - Reduces peak activation memory
    - Enables training on longer sequences
    - Trades compute for memory

    Trade-offs:
    - Adds re-computation overhead (for gradient checkpointing)
    - May increase communication if chunks are processed separately
    - Chunk granularity affects memory savings
    """

    name = "chunking"
    description = "Sequence chunking for memory optimization"

    def __init__(self, config: SchedulerConfig):
        """Initialize chunking feature.

        Args:
            config: Scheduler configuration
        """
        self.config = config
        self.chunk_size = config.chunk_size

    def apply_overhead(self, base_time: float) -> float:
        """Apply chunking overhead.

        Chunking may add overhead due to:
        - Boundary handling between chunks
        - Potential re-computation (gradient checkpointing)

        Typical overhead: 5-20%

        Args:
            base_time: Base execution time in seconds

        Returns:
            Time with chunking overhead
        """
        if self.chunk_size is None:
            return base_time

        return base_time * 1.05

    def apply_overlap(self, compute_time: float, comm_time: float) -> float:
        """Chunking doesn't directly affect overlap.

        Args:
            compute_time: Compute time in seconds
            comm_time: Communication time in seconds

        Returns:
            Sum of compute and comm time
        """
        return compute_time + comm_time

    def apply_memory_optimization(self, base_memory: int) -> int:
        """Apply memory optimization from chunking.

        Memory savings from chunking:
        - Reduces activation memory proportionally to chunk ratio
        - chunk_ratio = chunk_size / original_seq_len
        - Memory scales roughly linearly with chunk ratio

        Args:
            base_memory: Base memory in bytes

        Returns:
            Optimized memory in bytes
        """
        if self.chunk_size is None:
            return base_memory

        return int(base_memory * 0.7)

    def calculate_memory_reduction_ratio(self, original_seq_len: int, chunk_size: Optional[int] = None) -> float:
        """Calculate memory reduction ratio from chunking.

        Args:
            original_seq_len: Original sequence length
            chunk_size: Chunk size (uses config if None)

        Returns:
            Memory reduction ratio (0.0 - 1.0)
        """
        chunk = chunk_size or self.chunk_size
        if chunk is None or chunk >= original_seq_len:
            return 1.0

        chunk_ratio = chunk / original_seq_len
        return 0.3 + 0.7 * chunk_ratio

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "chunk_size": self.chunk_size,
        }
