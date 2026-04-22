"""Pipeline Parallelism Strategy.

Includes:
- PPStrategy: PP configuration (stages, schedule, micro-batches)
- PPSchedule: Schedule generation and visualization
"""

from typing import Dict, Optional, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_perf.modeling.module import ShardedModule


class PPStrategy:
    """Pipeline Parallelism Strategy.

    Attributes:
        num_stages: Number of PP stages (physical division)
        num_virtual_stages: Number of virtual stages per device (vpp)
        schedule: Schedule type ("1f1b", "gpipe", "interleaved")
        num_micro_batches: Number of micro-batches
        micro_batch_size: Size of each micro-batch
        stage_assignment: Custom layer-to-stage assignment
    """

    def __init__(
        self,
        num_stages: int = 1,
        num_virtual_stages: int = 1,
        schedule: str = "1f1b",
        num_micro_batches: int = 1,
        micro_batch_size: int = 1,
        stage_assignment: Optional[Dict[str, int]] = None,
    ):
        self.num_stages = num_stages
        self.num_virtual_stages = num_virtual_stages
        self.schedule = schedule
        self.num_micro_batches = num_micro_batches
        self.micro_batch_size = micro_batch_size
        self.stage_assignment = stage_assignment or {}

        self._validate()

    def _validate(self):
        """Validate strategy configuration."""
        if self.num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {self.num_stages}")

        if self.num_virtual_stages < 1:
            raise ValueError(f"num_virtual_stages must be >= 1, got {self.num_virtual_stages}")

        if self.num_micro_batches < 1:
            raise ValueError(f"num_micro_batches must be >= 1, got {self.num_micro_batches}")

        valid_schedules = ["1f1b", "gpipe", "interleaved", "vpp"]
        if self.schedule not in valid_schedules:
            raise ValueError(f"Invalid schedule: {self.schedule}, must be one of {valid_schedules}")

        if self.num_virtual_stages > 1 and self.schedule not in ["interleaved", "vpp"]:
            raise ValueError(f"num_virtual_stages > 1 requires 'interleaved' or 'vpp' schedule, got {self.schedule}")

    def assign_layers(
        self,
        model: "ShardedModule",
        method: str = "balanced",
    ) -> Dict[str, int]:
        """Assign layers to stages.

        Args:
            model: ShardedModule model
            method: Assignment method
                - "balanced": Even distribution by layer count
                - "memory_balanced": Balance by memory usage
                - "custom": Use stage_assignment

        Returns:
            {layer_name: stage_idx}
        """
        if method == "custom":
            return self.stage_assignment

        layer_names = self._get_layer_names(model)

        if method == "balanced":
            return self._assign_balanced(layer_names)

        elif method == "memory_balanced":
            return self._assign_memory_balanced(model, layer_names)

        raise ValueError(f"Unknown method: {method}")

    def _get_layer_names(self, model: "ShardedModule") -> List[str]:
        """Get layer names that can be assigned to stages."""
        layer_names = []

        for name, submodule in model._submodules.items():
            if name.startswith("layers."):
                layer_names.append(name)

        return sorted(layer_names, key=lambda x: int(x.split(".")[-1]) if "." in x else 0)

    def _assign_balanced(self, layer_names: List[str]) -> Dict[str, int]:
        """Assign layers evenly across stages."""
        assignment = {}

        if len(layer_names) == 0:
            return assignment

        layers_per_stage = max(1, len(layer_names) // self.num_stages)

        for i, layer_name in enumerate(layer_names):
            stage_idx = min(i // layers_per_stage, self.num_stages - 1)
            assignment[layer_name] = stage_idx

        return assignment

    def _assign_memory_balanced(self, model: "ShardedModule", layer_names: List[str]) -> Dict[str, int]:
        """Assign layers to balance memory across stages."""
        assignment = {}
        layer_memory = {}

        for name in layer_names:
            if name in model._submodules:
                submodule = model._submodules[name]
                layer_memory[name] = submodule.params_count()

        if len(layer_memory) == 0:
            return self._assign_balanced(layer_names)

        stage_memory = [0] * self.num_stages

        sorted_layers = sorted(layer_memory.keys(), key=lambda x: layer_memory[x], reverse=True)

        for layer_name in sorted_layers:
            min_stage = min(range(self.num_stages), key=lambda s: stage_memory[s])
            assignment[layer_name] = min_stage
            stage_memory[min_stage] += layer_memory[layer_name]

        return assignment

    def get_bubble_ratio(self) -> float:
        """Calculate bubble time ratio.

        Different schedules have different bubble ratios:
        - GPipe: (num_stages - 1) / num_micro_batches
        - 1F1B: (num_stages - 1) / (num_stages + num_micro_batches - 1)
        - Interleaved/VPP: (num_stages - 1) / (num_stages * num_virtual_stages + num_micro_batches - 1)
        """
        if self.num_micro_batches <= 0:
            return 0.0

        if self.schedule == "gpipe":
            return (self.num_stages - 1) / self.num_micro_batches

        elif self.schedule == "1f1b":
            denominator = self.num_stages + self.num_micro_batches - 1
            return (self.num_stages - 1) / denominator

        elif self.schedule in ["interleaved", "vpp"]:
            effective_stages = self.num_stages * self.num_virtual_stages
            denominator = effective_stages + self.num_micro_batches - 1
            return (self.num_stages - 1) / denominator

        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_stages": self.num_stages,
            "num_virtual_stages": self.num_virtual_stages,
            "schedule": self.schedule,
            "num_micro_batches": self.num_micro_batches,
            "micro_batch_size": self.micro_batch_size,
            "bubble_ratio": self.get_bubble_ratio(),
            "stage_assignment": self.stage_assignment,
        }


class PPSchedule:
    """Pipeline Parallelism Schedule generator and visualizer.

    Generates operation sequences for different schedule types:
    - GPipe: All forward then all backward
    - 1F1B: Interleaved forward/backward
    - Interleaved 1F1B: VPP schedule
    """

    @staticmethod
    def generate_gpipe_schedule(num_stages: int, num_micro_batches: int) -> List[List[str]]:
        """Generate GPipe schedule.

        GPipe schedule: All forwards then all backwards for each stage.

        Example (4 stages, 8 micro-batches):
            Stage 0: [F0, F1, F2, F3, F4, F5, F6, F7] -> [B0, B1, B2, B3, B4, B5, B6, B7]
            Stage 1: [F0, F1, ...] -> [B0, B1, ...]

        Returns:
            List of operation sequences per stage
        """
        schedules = []

        for stage in range(num_stages):
            forward_ops = [f"F{mb}" for mb in range(num_micro_batches)]
            backward_ops = [f"B{mb}" for mb in range(num_micro_batches)]
            schedules.append(forward_ops + backward_ops)

        return schedules

    @staticmethod
    def generate_1f1b_schedule(num_stages: int, num_micro_batches: int) -> List[List[str]]:
        """Generate 1F1B schedule.

        1F1B schedule: Warmup, steady (1F1B), cooldown.

        Example (4 stages, 8 micro-batches):
            Stage 0: [F0, F1, F2, F3] -> [F4, B0] -> [F5, B1] -> [F6, B2] -> [F7, B3] -> [B4, B5, B6, B7]

        Returns:
            List of operation sequences per stage
        """
        schedules = []

        for stage in range(num_stages):
            ops = []

            warmup_count = min(num_stages, num_micro_batches)
            for mb in range(warmup_count):
                ops.append(f"F{mb}")

            for mb in range(num_stages, num_micro_batches):
                ops.append(f"F{mb}")
                ops.append(f"B{mb - num_stages}")

            cooldown_start = num_micro_batches - num_stages
            for mb in range(cooldown_start, num_micro_batches):
                ops.append(f"B{mb}")

            schedules.append(ops)

        return schedules

    @staticmethod
    def generate_interleaved_schedule(
        num_stages: int,
        num_virtual_stages: int,
        num_micro_batches: int,
    ) -> List[List[str]]:
        """Generate Interleaved 1F1B schedule for VPP.

        Interleaved schedule: One device handles multiple stages in interleaved manner.

        Example (8 stages, vpp=2, 16 micro-batches):
            device 0 (Stage 0, 4): [F0_s0, F0_s4, F1_s0, F1_s4, ...]

        Returns:
            List of operation sequences per device
        """
        num_devices = num_stages // num_virtual_stages
        schedules = []

        for device in range(num_devices):
            ops = []
            stages_for_device = [device * num_virtual_stages + v for v in range(num_virtual_stages)]

            warmup_count = min(num_stages * num_virtual_stages, num_micro_batches)
            for mb in range(warmup_count):
                for stage in stages_for_device:
                    ops.append(f"F{mb}_s{stage}")

            steady_start = num_stages * num_virtual_stages
            for mb in range(steady_start, num_micro_batches):
                for stage in stages_for_device:
                    ops.append(f"F{mb}_s{stage}")
                    ops.append(f"B{mb - num_stages}_s{stage}")

            cooldown_start = num_micro_batches - num_stages
            for mb in range(cooldown_start, num_micro_batches):
                for stage in stages_for_device:
                    ops.append(f"B{mb}_s{stage}")

            schedules.append(ops)

        return schedules

    @staticmethod
    def visualize_schedule(
        schedules: List[List[str]],
        stage_names: Optional[List[str]] = None,
    ) -> str:
        """Visualize schedule as string.

        Args:
            schedules: List of operation sequences
            stage_names: Optional stage names

        Returns:
            Visualization string
        """
        if stage_names is None:
            stage_names = [f"Stage {i}" for i in range(len(schedules))]

        max_ops = max(len(s) for s in schedules) if schedules else 0
        lines = []

        lines.append("\nPP Schedule Visualization:")
        lines.append("-" * (max_ops * 6 + 15))

        for stage_name, ops in zip(stage_names, schedules):
            ops_str = " ".join(f"{op:5s}" for op in ops)
            lines.append(f"{stage_name:12s} | {ops_str}")

        lines.append("-" * (max_ops * 6 + 15))

        return "\n".join(lines)

    @staticmethod
    def count_operations(schedules: List[List[str]]) -> Dict[str, int]:
        """Count forward and backward operations.

        Args:
            schedules: List of operation sequences

        Returns:
            {"forward": count, "backward": count}
        """
        forward_count = 0
        backward_count = 0

        for ops in schedules:
            for op in ops:
                if op.startswith("F"):
                    forward_count += 1
                elif op.startswith("B"):
                    backward_count += 1

        return {"forward": forward_count, "backward": backward_count}
