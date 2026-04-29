"""Workload and Model configuration loader.

Loads configurations from:
- configs/workloads/*.yaml
- configs/models/*.yaml

Core principle: Reuse existing configs, avoid redefinition.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml

from .schema import (
    WorkloadSchema,
    StageSchema,
    ModelSchema,
    ParamSchemaItem,
    WorkloadCategory,
)


CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
WORKLOADS_DIR = CONFIGS_DIR / "workloads"
MODELS_DIR = CONFIGS_DIR / "models"


def _format_param_label(param_name: str) -> str:
    """Convert parameter name to display label."""
    labels = {
        "batch_size": "Batch Size",
        "seq_len": "Sequence Length",
        "prompt_len": "Prompt Length",
        "generation_len": "Generation Length",
        "num_steps": "Diffusion Steps",
        "num_frames": "Number of Frames",
        "height": "Height",
        "width": "Width",
        "diffusion_steps": "Diffusion Steps",
        "micro_batch_size": "Micro Batch Size",
        "gradient_accumulation_steps": "Gradient Accumulation Steps",
        "image_height": "Image Height",
        "image_width": "Image Width",
    }
    return labels.get(param_name, param_name.replace("_", " ").title())


def _infer_param_type(value: Any) -> str:
    """Infer parameter type from value."""
    if isinstance(value, bool):
        return "boolean"
    elif isinstance(value, (int, float)):
        return "number"
    else:
        return "string"


class WorkloadLoader:
    """Workload and Model configuration loader.

    Loads configurations from YAML files and converts to schemas.
    """

    def __init__(self):
        self._workload_cache: Dict[str, Dict] = {}
        self._model_cache: Dict[str, Dict] = {}

    def list_workloads(self) -> List[str]:
        """List all workload names.

        Returns:
            List of workload names like ["training/training", "inference/autoregressive", ...]
        """
        workloads = []
        if not WORKLOADS_DIR.exists():
            return workloads

        for category_dir in WORKLOADS_DIR.iterdir():
            if category_dir.is_dir() and category_dir.name != "custom":
                for yaml_file in category_dir.glob("*.yaml"):
                    workloads.append(f"{category_dir.name}/{yaml_file.stem}")
        return sorted(workloads)

    def list_workload_categories(self) -> Dict[str, List[str]]:
        """List workloads grouped by category.

        Returns:
            Dict like {"training": ["training", "denoise"], "inference": [...]}
        """
        categories = {}
        if not WORKLOADS_DIR.exists():
            return categories

        for category_dir in WORKLOADS_DIR.iterdir():
            if category_dir.is_dir() and category_dir.name != "custom":
                workloads = [f.stem for f in category_dir.glob("*.yaml")]
                categories[category_dir.name] = sorted(workloads)
        return categories

    def load_workload_yaml(self, workload_name: str) -> Dict[str, Any]:
        """Load workload YAML configuration.

        Args:
            workload_name: Like "inference/autoregressive" or "training"

        Returns:
            Workload config dict
        """
        if workload_name in self._workload_cache:
            return self._workload_cache[workload_name]

        workload_path = self._resolve_workload_path(workload_name)
        if not workload_path.exists():
            raise FileNotFoundError(f"Workload not found: {workload_name}")

        with open(workload_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._workload_cache[workload_name] = config
        return config

    def _resolve_workload_path(self, name: str) -> Path:
        """Resolve workload path from name.

        Supports:
        - Full path: "inference/autoregressive" -> configs/workloads/inference/autoregressive.yaml
        - Short name: "training" -> configs/workloads/training/training.yaml
        - Absolute path: "/path/to/workload.yaml"
        """
        if "/" in name:
            return WORKLOADS_DIR / f"{name}.yaml"

        for category_dir in WORKLOADS_DIR.iterdir():
            if category_dir.is_dir():
                candidate = category_dir / f"{name}.yaml"
                if candidate.exists():
                    return candidate

        return WORKLOADS_DIR / name / f"{name}.yaml"

    def list_models(self) -> List[str]:
        """List all model names.

        Returns:
            List like ["llama-7b", "deepseek-v3", ...]
        """
        models = []
        if not MODELS_DIR.exists():
            return models

        for yaml_file in MODELS_DIR.glob("*.yaml"):
            models.append(yaml_file.stem)
        return sorted(models)

    def load_model_yaml(self, model_name: str) -> Dict[str, Any]:
        """Load model YAML configuration.

        Args:
            model_name: Like "llama-7b", "deepseek-v3"

        Returns:
            Model config dict
        """
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        model_path = MODELS_DIR / f"{model_name}.yaml"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_name}")

        with open(model_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._model_cache[model_name] = config
        return config

    def get_workload_schema(self, workload_name: str) -> WorkloadSchema:
        """Get WorkloadSchema for frontend rendering.

        Args:
            workload_name: Like "inference/autoregressive"

        Returns:
            WorkloadSchema instance
        """
        yaml_config = self.load_workload_yaml(workload_name)
        return self._yaml_to_workload_schema(workload_name, yaml_config)

    def _yaml_to_workload_schema(
        self, workload_name: str, yaml_config: Dict[str, Any]
    ) -> WorkloadSchema:
        """Convert YAML config to WorkloadSchema."""
        category_str = workload_name.split("/")[0] if "/" in workload_name else "inference"
        try:
            category = WorkloadCategory(category_str)
        except ValueError:
            category = WorkloadCategory.INFERENCE

        stages = []
        for phase_data in yaml_config.get("phases", []):
            stages.append(
                StageSchema(
                    name=phase_data.get("name", "phase"),
                    compute_type=phase_data.get("compute_type", "forward"),
                    component=phase_data.get("component", "main"),
                    repeat=phase_data.get("repeat", 1),
                    compute_pattern=phase_data.get("compute_pattern"),
                    extra_params=phase_data.get("extra_params", {}),
                )
            )

        parameters = {}
        for param_name, default_value in yaml_config.get("default_params", {}).items():
            parameters[param_name] = ParamSchemaItem(
                name=param_name,
                label=_format_param_label(param_name),
                type=_infer_param_type(default_value),
                default=default_value,
                required=True,
            )

        models = self.list_models()

        return WorkloadSchema(
            name=yaml_config.get("name", workload_name),
            workload_name=workload_name,
            description=yaml_config.get("description", ""),
            category=category,
            workload_type=yaml_config.get("workload_type", "inference"),
            stages=stages,
            parameters=parameters,
            throughput_metric=yaml_config.get("throughput_metric", "tokens_per_sec"),
            supported_models=models,
        )

    def get_model_schema(self, model_name: str) -> ModelSchema:
        """Get ModelSchema for frontend rendering.

        Args:
            model_name: Like "llama-7b"

        Returns:
            ModelSchema instance
        """
        yaml_config = self.load_model_yaml(model_name)
        return self._yaml_to_model_schema(model_name, yaml_config)

    def _yaml_to_model_schema(
        self, model_name: str, yaml_config: Dict[str, Any]
    ) -> ModelSchema:
        """Convert YAML config to ModelSchema."""
        param_schema = {}
        for workload_type, params in yaml_config.get("param_schema", {}).items():
            param_items = []
            for p in params:
                param_items.append(
                    ParamSchemaItem(
                        name=p.get("name", ""),
                        label=p.get("label", p.get("name", "")),
                        type=p.get("type", "number"),
                        default=p.get("default"),
                        min=p.get("min"),
                        max=p.get("max"),
                        required=True,
                        description=p.get("description", ""),
                    )
                )
            param_schema[workload_type] = param_items

        return ModelSchema(
            name=model_name,
            description=yaml_config.get("description", ""),
            architecture=yaml_config.get("architecture", ""),
            sparse_type=yaml_config.get("sparse_type", "dense"),
            attention_features=yaml_config.get("attention_features", []),
            supported_workloads=yaml_config.get("supported_workloads", []),
            config=yaml_config.get("config", {}),
            param_schema=param_schema,
        )

    def get_workload_config(self, workload_name: str) -> Dict[str, Any]:
        """Get raw workload config for UnifiedAnalyzer.

        This is the raw YAML config, used to create WorkloadConfig
        in analyzer/base.py.

        Args:
            workload_name: Like "inference/autoregressive"

        Returns:
            Raw workload config dict
        """
        return self.load_workload_yaml(workload_name)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get raw model config for model instantiation.

        Args:
            model_name: Like "llama-7b"

        Returns:
            Raw model config dict
        """
        return self.load_model_yaml(model_name)

    def get_supported_workloads_for_model(self, model_name: str) -> List[str]:
        """Get workload categories supported by a model.

        Args:
            model_name: Like "llama-7b"

        Returns:
            List of supported workload categories
        """
        yaml_config = self.load_model_yaml(model_name)
        return yaml_config.get("supported_workloads", ["training", "inference"])

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._workload_cache.clear()
        self._model_cache.clear()


_loader_instance: Optional[WorkloadLoader] = None


def get_loader() -> WorkloadLoader:
    """Get singleton WorkloadLoader instance."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = WorkloadLoader()
    return _loader_instance