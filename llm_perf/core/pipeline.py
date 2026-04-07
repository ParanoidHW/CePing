"""Pipeline abstraction for model composition and iteration.

Provides a unified interface for chaining sub-models, including support for
iterative processes like diffusion denoising.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from ..models.base import BaseModel
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig

# Type variable for pipeline results
ResultT = TypeVar("ResultT")


@dataclass
class IterationConfig:
    """Configuration for iterative pipeline steps.

    Used for processes like diffusion denoising that require multiple iterations.

    Attributes:
        num_iterations: Number of iterations to run
        early_stopping: Whether to support early stopping
        stop_condition: Optional callable to check stop condition
        iteration_callback: Optional callback after each iteration
    """

    num_iterations: int = 1
    early_stopping: bool = False
    stop_condition: Optional[Callable[[int, Dict[str, Any]], bool]] = None
    iteration_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None

    def should_stop(self, iteration: int, state: Dict[str, Any]) -> bool:
        """Check if iteration should stop early.

        Args:
            iteration: Current iteration number
            state: Current state dictionary

        Returns:
            True if should stop, False otherwise
        """
        if not self.early_stopping:
            return False
        if self.stop_condition is not None:
            return self.stop_condition(iteration, state)
        return False


@dataclass
class PipelineStep:
    """A single step in a pipeline.

    Represents either a model inference step or a custom computation step.

    Attributes:
        name: Step identifier
        model: Model to use (optional for custom steps)
        is_iterative: Whether this step uses iteration
        iteration_config: Configuration for iterative execution
        custom_fn: Optional custom function for non-model steps
        depends_on: List of step names this step depends on
    """

    name: str
    model: Optional[BaseModel] = None
    is_iterative: bool = False
    iteration_config: Optional[IterationConfig] = None
    custom_fn: Optional[Callable[..., Any]] = None
    depends_on: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate step configuration."""
        if self.model is None and self.custom_fn is None:
            raise ValueError(f"Step '{self.name}' must have either model or custom_fn")
        if self.is_iterative and self.iteration_config is None:
            self.iteration_config = IterationConfig()


@dataclass
class PipelineConfig:
    """Configuration for a pipeline.

    Attributes:
        name: Pipeline identifier
        steps: Ordered list of pipeline steps
        device: Device configuration
        cluster: Cluster configuration
        strategy: Parallelism strategy
        batch_size: Batch size for inference
        enable_profiling: Whether to collect detailed timing
    """

    name: str
    steps: List[PipelineStep] = field(default_factory=list)
    device: Optional[Device] = None
    cluster: Optional[Cluster] = None
    strategy: Optional[StrategyConfig] = None
    batch_size: int = 1
    enable_profiling: bool = False


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    Attributes:
        total_time_sec: Total execution time in seconds
        step_times: Dictionary of step name to execution time
        step_results: Dictionary of step name to result data
        memory_peak_gb: Peak memory usage in GB
        throughput: Optional throughput metric
        metadata: Additional metadata
    """

    total_time_sec: float
    step_times: Dict[str, float] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    memory_peak_gb: float = 0.0
    throughput: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_time_sec": self.total_time_sec,
            "step_times": self.step_times,
            "step_results": {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.step_results.items()
            },
            "memory_peak_gb": self.memory_peak_gb,
            "throughput": self.throughput,
            "metadata": self.metadata,
        }


class Pipeline(ABC):
    """Abstract base class for model pipelines.

    Pipelines chain multiple models together for end-to-end evaluation.
    Supports both linear pipelines and complex workflows with iteration.

    Example implementations:
        - Simple LLM inference (single model)
        - Text-to-image generation (Text Encoder -> DiT -> VAE)
        - Text-to-video generation (Text Encoder -> DiT (iterative) -> VAE)

    Example:
        >>> class DiffusionPipeline(Pipeline):
        ...     def build_steps(self) -> List[PipelineStep]:
        ...         return [
        ...             PipelineStep("text_encoder", model=text_encoder),
        ...             PipelineStep("dit", model=dit, is_iterative=True,
        ...                         iteration_config=IterationConfig(num_iterations=50)),
        ...             PipelineStep("vae_decoder", model=vae_decoder),
        ...         ]
        ...
        ...     def execute_step(self, step, inputs):
        ...         # Custom execution logic
        ...         return step.model.forward(inputs)
    """

    def __init__(
        self,
        config: PipelineConfig,
    ):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self._steps: List[PipelineStep] = []
        self._step_map: Dict[str, PipelineStep] = {}
        self._built = False

    @property
    def steps(self) -> List[PipelineStep]:
        """Get pipeline steps.

        Returns:
            List of pipeline steps
        """
        if not self._built:
            self._build()
        return self._steps

    def _build(self) -> None:
        """Build the pipeline steps."""
        self._steps = self.build_steps()
        self._step_map = {step.name: step for step in self._steps}
        self._validate_dependencies()
        self._built = True

    @abstractmethod
    def build_steps(self) -> List[PipelineStep]:
        """Build and return pipeline steps.

        Subclasses must implement this to define the pipeline structure.

        Returns:
            Ordered list of pipeline steps
        """
        pass

    def _validate_dependencies(self) -> None:
        """Validate step dependencies.

        Raises:
            ValueError: If dependencies are invalid
        """
        step_names = {step.name for step in self._steps}
        for step in self._steps:
            for dep in step.depends_on:
                if dep not in step_names:
                    raise ValueError(
                        f"Step '{step.name}' depends on unknown step '{dep}'"
                    )

    @abstractmethod
    def execute_step(
        self, step: PipelineStep, inputs: Dict[str, Any]
    ) -> Tuple[Any, float]:
        """Execute a single pipeline step.

        Args:
            step: Pipeline step to execute
            inputs: Input data for the step

        Returns:
            Tuple of (result, execution_time_seconds)
        """
        pass

    def execute_iterative_step(
        self,
        step: PipelineStep,
        inputs: Dict[str, Any],
        iteration_config: IterationConfig,
    ) -> Tuple[Any, float, int]:
        """Execute an iterative pipeline step.

        Args:
            step: Pipeline step to execute
            inputs: Initial input data
            iteration_config: Iteration configuration

        Returns:
            Tuple of (final_result, total_time_seconds, num_iterations_executed)
        """
        total_time = 0.0
        state = dict(inputs)
        final_result = None

        for i in range(iteration_config.num_iterations):
            # Execute single iteration
            result, step_time = self.execute_step(step, state)
            total_time += step_time
            final_result = result

            # Update state for next iteration
            state["previous_result"] = result
            state["iteration"] = i

            # Call iteration callback if provided
            if iteration_config.iteration_callback is not None:
                iteration_config.iteration_callback(i, state)

            # Check early stopping
            if iteration_config.should_stop(i, state):
                break

        return final_result, total_time, i + 1

    def run(self, inputs: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """Run the full pipeline.

        Args:
            inputs: Optional initial inputs

        Returns:
            PipelineResult with execution metrics
        """
        if not self._built:
            self._build()

        inputs = inputs or {}
        step_times: Dict[str, float] = {}
        step_results: Dict[str, Any] = {}
        total_time = 0.0
        max_memory = 0.0

        for step in self._steps:
            # Prepare inputs based on dependencies
            step_inputs = {"inputs": inputs}
            for dep in step.depends_on:
                step_inputs[dep] = step_results.get(dep)

            # Execute step
            if step.is_iterative and step.iteration_config is not None:
                result, step_time, num_iters = self.execute_iterative_step(
                    step, step_inputs, step.iteration_config
                )
                step_results[step.name] = {
                    "result": result,
                    "iterations": num_iters,
                }
            else:
                result, step_time = self.execute_step(step, step_inputs)
                step_results[step.name] = result

            step_times[step.name] = step_time
            total_time += step_time

        return PipelineResult(
            total_time_sec=total_time,
            step_times=step_times,
            step_results=step_results,
            memory_peak_gb=max_memory,
        )

    def get_step(self, name: str) -> Optional[PipelineStep]:
        """Get a step by name.

        Args:
            name: Step name

        Returns:
            PipelineStep if found, None otherwise
        """
        if not self._built:
            self._build()
        return self._step_map.get(name)

    def add_step(self, step: PipelineStep) -> None:
        """Add a step to the pipeline.

        Args:
            step: Step to add

        Raises:
            ValueError: If step name already exists
        """
        if step.name in self._step_map:
            raise ValueError(f"Step '{step.name}' already exists in pipeline")
        self._steps.append(step)
        self._step_map[step.name] = step

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary representation."""
        return {
            "name": self.config.name,
            "steps": [
                {
                    "name": step.name,
                    "has_model": step.model is not None,
                    "is_iterative": step.is_iterative,
                    "depends_on": step.depends_on,
                }
                for step in self.steps
            ],
            "batch_size": self.config.batch_size,
            "enable_profiling": self.config.enable_profiling,
        }


class SimplePipeline(Pipeline):
    """Simple pipeline that executes models sequentially.

    Basic implementation for pipelines without complex orchestration needs.
    """

    def __init__(
        self,
        config: PipelineConfig,
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        """Initialize simple pipeline.

        Args:
            config: Pipeline configuration
            device: Device configuration
            cluster: Cluster configuration
            strategy: Parallelism strategy
        """
        super().__init__(config)
        self.device = device
        self.cluster = cluster
        self.strategy = strategy

    def build_steps(self) -> List[PipelineStep]:
        """Build steps from config.

        Returns:
            List of pipeline steps
        """
        return self.config.steps

    def execute_step(
        self, step: PipelineStep, inputs: Dict[str, Any]
    ) -> Tuple[Any, float]:
        """Execute a single step using roofline estimation.

        Args:
            step: Pipeline step
            inputs: Input data

        Returns:
            Tuple of (result, execution_time_seconds)
        """
        import time

        start_time = time.perf_counter()

        if step.custom_fn is not None:
            # Execute custom function
            result = step.custom_fn(inputs)
        elif step.model is not None:
            # Estimate execution time using model FLOPs
            result = self._estimate_model_execution(step.model)
        else:
            raise ValueError(f"Step '{step.name}' has no execution method")

        elapsed = time.perf_counter() - start_time
        return result, elapsed

    def _estimate_model_execution(self, model: BaseModel) -> Dict[str, Any]:
        """Estimate model execution using roofline model.

        Args:
            model: Model to estimate

        Returns:
            Execution result metadata
        """
        # Use roofline model for estimation
        total_flops = model.total_flops_forward

        return {
            "flops": total_flops,
            "params": model.total_params,
            "memory_bytes": model.activation_memory,
        }
