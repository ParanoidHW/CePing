"""Interactive mode for CLI.

Provides step-by-step interactive configuration:
1. Select Workload Category
2. Select Workload
3. Select Model
4. Configure Hardware
5. Configure Parameters
6. Configure Parallel Strategy
7. Select Output Format
8. Confirm and Run
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import readline

from llm_perf.workload.loader import WorkloadLoader, get_loader
from llm_perf.workload.schema import (
    WorkloadSchema,
    ModelSchema,
    HardwareSchema,
    StrategySchema,
    ParamSchemaItem,
    WorkloadCategory,
)


class InteractiveSession:
    """Interactive configuration session.

    Provides step-by-step wizard for evaluation configuration.
    """

    def __init__(self, loader: WorkloadLoader):
        self.loader = loader
        self.config: Dict[str, Any] = {}
        self.output_format: str = "json"
        self._should_save: bool = False
        self._save_path: Optional[Path] = None
        self._should_run: bool = True

    def run(self) -> Optional[Dict[str, Any]]:
        """Run interactive configuration.

        Returns:
            Config dict if confirmed, None if cancelled
        """
        self._print_banner()

        try:
            workload_name = self._select_workload()
            if workload_name is None:
                return None

            model_name = self._select_model(workload_name)
            if model_name is None:
                return None

            hardware = self._configure_hardware()

            params = self._configure_params(workload_name)

            strategy = self._configure_strategy()

            self.output_format = self._select_output_format()

            self._print_summary(workload_name, model_name, hardware, params, strategy)

            if not self._confirm():
                return None

            self.config = {
                "workload_name": workload_name,
                "model_name": model_name,
                "hardware": hardware.to_dict(),
                "strategy": strategy.to_dict(),
                "params": params,
            }

            self._should_run = self._ask_run()

            if self._should_run:
                self._should_save = self._ask_save()
                if self._should_save:
                    self._save_path = self._get_save_path()

            return self.config

        except KeyboardInterrupt:
            print("\nConfiguration cancelled.")
            return None

    def should_run(self) -> bool:
        """Return whether to run evaluation."""
        return self._should_run

    def should_save(self) -> bool:
        """Return whether to save results."""
        return self._should_save

    def get_save_path(self) -> Path:
        """Return save path."""
        if self._save_path:
            return self._save_path

        workload_name = self.config.get("workload_name", "evaluation")
        model_name = self.config.get("model_name", "model")
        ext = ".json" if self.output_format == "json" else ".yaml"
        return Path(f"results/{workload_name}_{model_name}{ext}")

    def get_output_format(self) -> str:
        """Return output format."""
        return self.output_format

    def _print_banner(self) -> None:
        """Print welcome banner."""
        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║           LLM Performance Evaluator CLI                  ║")
        print("║                  Interactive Mode                        ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()

    def _select_workload(self) -> Optional[str]:
        """Select workload category and workload.

        Returns:
            Workload name (e.g., "inference/autoregressive")
        """
        print("Step 1: Select Workload Category")
        print()

        categories = self.loader.list_workload_categories()
        category_list = sorted(categories.keys())

        for i, cat in enumerate(category_list, 1):
            count = len(categories[cat])
            print(f"  {i}. {cat} ({count} workloads)")

        print()

        choice = self._input_choice(f"Enter choice [1-{len(category_list)}]: ", len(category_list))
        if choice is None:
            return None

        category = category_list[choice - 1]
        print()

        print(f"Step 2: Select Workload [{category}]")
        print()

        workloads = categories[category]
        for i, w in enumerate(workloads, 1):
            workload_name = f"{category}/{w}"
            schema = self.loader.get_workload_schema(workload_name)
            print(f"  {i}. {w} - {schema.description}")

        print()

        choice = self._input_choice(f"Enter choice [1-{len(workloads)}]: ", len(workloads))
        if choice is None:
            return None

        workload = workloads[choice - 1]
        return f"{category}/{workload}"

    def _select_model(self, workload_name: str) -> Optional[str]:
        """Select model.

        Args:
            workload_name: Selected workload name

        Returns:
            Model name
        """
        print()
        print("Step 3: Select Model")
        print()

        models = self.loader.list_models()

        for i, m in enumerate(models, 1):
            schema = self.loader.get_model_schema(m)
            supported = schema.supported_workloads

            workload_category = workload_name.split("/")[0]
            is_supported = workload_category in supported or len(supported) == 0

            marker = "✓" if is_supported else " "
            print(f"  {i}. [{marker}] {m} - {schema.description}")

        print()
        print("  [✓] = supported by workload")

        print()

        choice = self._input_choice(f"Enter choice [1-{len(models)}]: ", len(models))
        if choice is None:
            return None

        return models[choice - 1]

    def _configure_hardware(self) -> HardwareSchema:
        """Configure hardware.

        Returns:
            HardwareSchema
        """
        print()
        print("Step 4: Configure Hardware")
        print()

        device = self._input_default("Device [910B]: ", "910B")

        num_devices = self._input_int("Number of devices [8]: ", 8)

        return HardwareSchema(
            device_preset=device,
            num_devices=num_devices,
        )

    def _configure_params(self, workload_name: str) -> Dict[str, Any]:
        """Configure workload parameters.

        Args:
            workload_name: Selected workload name

        Returns:
            Params dict
        """
        print()
        print("Step 5: Configure Workload Parameters")
        print()

        schema = self.loader.get_workload_schema(workload_name)
        params = {}

        for param_name, param_schema in schema.parameters.items():
            default = param_schema.default
            label = param_schema.label

            value = self._input_default(f"{label} [{default}]: ", default)
            params[param_name] = value

        return params

    def _configure_strategy(self) -> StrategySchema:
        """Configure parallel strategy.

        Returns:
            StrategySchema
        """
        print()
        print("Step 6: Configure Parallel Strategy")
        print()

        tp = self._input_int("TP degree [1]: ", 1)
        pp = self._input_int("PP degree [1]: ", 1)
        dp = self._input_int("DP degree [1]: ", 1)

        return StrategySchema(
            tp_degree=tp,
            pp_degree=pp,
            dp_degree=dp,
        )

    def _select_output_format(self) -> str:
        """Select output format.

        Returns:
            Output format ("json", "yaml", "table")
        """
        print()
        print("Step 7: Select Output Format")
        print()

        formats = ["json", "yaml", "table"]
        for i, f in enumerate(formats, 1):
            print(f"  {i}. {f}")

        print()

        choice = self._input_choice(f"Enter choice [1-{len(formats)}]: ", len(formats))
        if choice is None:
            return "json"

        return formats[choice - 1]

    def _print_summary(
        self,
        workload_name: str,
        model_name: str,
        hardware: HardwareSchema,
        params: Dict[str, Any],
        strategy: StrategySchema,
    ) -> None:
        """Print configuration summary."""
        print()
        print("─" * 50)
        print("Configuration Summary:")
        print("─" * 50)
        print(f"  Workload: {workload_name}")
        print(f"  Model: {model_name}")
        print(f"  Hardware: {hardware.num_devices}x {hardware.device_preset}")
        print(f"  Strategy: TP={strategy.tp_degree}, PP={strategy.pp_degree}, DP={strategy.dp_degree}")
        print()
        print("  Parameters:")
        for k, v in params.items():
            print(f"    {k}: {v}")
        print("─" * 50)

    def _confirm(self) -> bool:
        """Confirm configuration.

        Returns:
            True if confirmed, False if cancelled
        """
        print()
        answer = self._input_yes_no("Run evaluation? [Y/n]: ")
        return answer

    def _ask_run(self) -> bool:
        """Ask whether to run evaluation.

        Returns:
            True if should run
        """
        return True

    def _ask_save(self) -> bool:
        """Ask whether to save results.

        Returns:
            True if should save
        """
        print()
        answer = self._input_yes_no("Save results? [Y/n]: ")
        return answer

    def _get_save_path(self) -> Path:
        """Get save path.

        Returns:
            Save path
        """
        workload_name = self.config.get("workload_name", "evaluation")
        model_name = self.config.get("model_name", "model")

        default_path = f"results/{workload_name.replace('/', '_')}_{model_name}"
        if self.output_format == "json":
            default_path += ".json"
        elif self.output_format == "yaml":
            default_path += ".yaml"
        else:
            default_path += ".txt"

        print()
        path_str = self._input_default(f"Output file [{default_path}]: ", default_path)
        return Path(path_str)

    def _input_choice(self, prompt: str, max_choice: int) -> Optional[int]:
        """Input choice from numbered list.

        Args:
            prompt: Input prompt
            max_choice: Maximum valid choice

        Returns:
            Choice number (1-based), or None if cancelled
        """
        while True:
            try:
                value = input(prompt).strip()
                if value.lower() in ["q", "quit", "exit", "cancel"]:
                    return None

                if value == "":
                    return 1

                choice = int(value)
                if 1 <= choice <= max_choice:
                    return choice
                else:
                    print(f"Invalid choice. Please enter 1-{max_choice}.")

            except ValueError:
                print("Invalid input. Please enter a number.")
            except EOFError:
                return None

    def _input_int(self, prompt: str, default: int) -> int:
        """Input integer value.

        Args:
            prompt: Input prompt
            default: Default value

        Returns:
            Integer value
        """
        while True:
            try:
                value = input(prompt).strip()
                if value == "":
                    return default

                return int(value)

            except ValueError:
                print("Invalid input. Please enter a number.")
            except EOFError:
                return default

    def _input_default(self, prompt: str, default: Any) -> Any:
        """Input with default value.

        Args:
            prompt: Input prompt
            default: Default value

        Returns:
            Input value or default
        """
        try:
            value = input(prompt).strip()
            if value == "":
                return default

            if isinstance(default, int):
                return int(value)
            elif isinstance(default, float):
                return float(value)
            else:
                return value

        except (ValueError, EOFError):
            return default

    def _input_yes_no(self, prompt: str) -> bool:
        """Input yes/no confirmation.

        Args:
            prompt: Input prompt

        Returns:
            True for yes, False for no
        """
        while True:
            try:
                value = input(prompt).strip().lower()
                if value in ["y", "yes", "", "1"]:
                    return True
                elif value in ["n", "no", "0"]:
                    return False
                else:
                    print("Invalid input. Please enter Y or N.")

            except EOFError:
                return True