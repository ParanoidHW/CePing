"""Pipeline bubble feature for pipeline parallelism bubble modeling."""

from typing import Dict, Any

from ..base import SchedulerFeature, SchedulerConfig


class PipelineBubbleFeature(SchedulerFeature):
    """Models pipeline parallelism bubble overhead.

    Pipeline parallelism introduces "bubbles" - idle time at the
    beginning and end of each micro-batch schedule when some stages
    are waiting for data.

    Bubble ratio depends on:
    - Number of pipeline stages (pp_degree)
    - Number of micro-batches
    - Schedule type (1F1B, GPipe, interleaved, etc.)

    Bubble ratio formulas:
    - GPipe: (pp_degree - 1) / num_micro_batches
    - 1F1B: (pp_degree - 1) / (pp_degree + num_micro_batches - 1)
    - Interleaved: Reduced by factor of num_model_chunks
    """

    name = "pipeline_bubble"
    description = "Pipeline parallelism bubble overhead modeling"

    def __init__(self, config: SchedulerConfig):
        """Initialize pipeline bubble feature.

        Args:
            config: Scheduler configuration
        """
        self.config = config
        self.bubble_ratio = config.pipeline_bubble_ratio

    def apply_overhead(self, base_time: float) -> float:
        """Apply pipeline bubble overhead.

        Total time = base_time * (1 + bubble_ratio)

        Args:
            base_time: Base execution time in seconds

        Returns:
            Time with bubble overhead
        """
        return base_time * (1 + self.bubble_ratio)

    def apply_overlap(self, compute_time: float, comm_time: float) -> float:
        """Pipeline bubble doesn't directly affect overlap.

        Args:
            compute_time: Compute time in seconds
            comm_time: Communication time in seconds

        Returns:
            Sum of compute and comm time
        """
        return compute_time + comm_time

    def apply_memory_optimization(self, base_memory: int) -> int:
        """Pipeline doesn't directly reduce memory per device.

        Memory is distributed across stages, but each stage
        holds its portion of the model.

        Args:
            base_memory: Base memory in bytes

        Returns:
            Unmodified memory
        """
        return base_memory

    def calculate_bubble_ratio(self, pp_degree: int, num_micro_batches: int, schedule: str = "1f1b") -> float:
        """Calculate bubble ratio for given configuration.

        Args:
            pp_degree: Pipeline parallelism degree
            num_micro_batches: Number of micro-batches
            schedule: Schedule type ("1f1b", "gpipe", "interleaved")

        Returns:
            Bubble ratio (0.0 - 1.0)
        """
        if pp_degree <= 1:
            return 0.0

        if num_micro_batches <= 0:
            num_micro_batches = pp_degree

        if schedule == "gpipe":
            return (pp_degree - 1) / num_micro_batches

        if schedule == "1f1b":
            return (pp_degree - 1) / (pp_degree + num_micro_batches - 1)

        if schedule == "interleaved":
            base_bubble = (pp_degree - 1) / (pp_degree + num_micro_batches - 1)
            return base_bubble / 2

        return self.bubble_ratio

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "bubble_ratio": self.bubble_ratio,
        }
