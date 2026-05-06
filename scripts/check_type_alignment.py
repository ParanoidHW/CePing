"""Type alignment checker for backend Pydantic models and frontend TypeScript types.

This script:
1. Parses backend Pydantic model field definitions
2. Parses frontend TypeScript type definitions
3. Compares field names and types
4. Reports mismatches

Usage:
    python scripts/check_type_alignment.py

Output:
    - Matching fields: ✅
    - Missing fields: ❌ (in backend but not frontend)
    - Extra fields: ⚠️ (in frontend but not backend)
    - Type mismatches: ⚠️
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class PydanticFieldParser:
    """Parse Pydantic model field definitions from Python source."""

    def parse_file(self, filepath: Path) -> Dict[str, Dict[str, str]]:
        """Parse all Pydantic models from a Python file.

        Returns:
            Dict mapping model_name -> Dict mapping field_name -> field_type
        """
        with open(filepath) as f:
            source = f.read()

        tree = ast.parse(source)
        models = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "BaseModel":
                        fields = {}
                        for item in node.body:
                            if isinstance(item, ast.AnnAssign) and item.target:
                                field_name = item.target.id if isinstance(item.target, ast.Name) else ""
                                field_type = self._parse_type(item.annotation)
                                if field_name and not field_name.startswith("_"):
                                    fields[field_name] = field_type
                        models[node.name] = fields
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name) and decorator.func.id == "BaseModel":
                            fields = {}
                            for item in node.body:
                                if isinstance(item, ast.AnnAssign) and item.target:
                                    field_name = item.target.id if isinstance(item.target, ast.Name) else ""
                                    field_type = self._parse_type(item.annotation)
                                    if field_name and not field_name.startswith("_"):
                                        fields[field_name] = field_type
                        models[node.name] = fields

        return models

    def _parse_type(self, annotation: ast.expr) -> str:
        """Parse type annotation to string."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                base = annotation.value.id
                if isinstance(annotation.slice, ast.Name):
                    return f"{base}[{annotation.slice.id}]"
                elif isinstance(annotation.slice, ast.Tuple):
                    elts = [self._parse_type(e) for e in annotation.slice.elts]
                    return f"{base}[{', '.join(elts)}]"
            return "Unknown"
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif annotation is None:
            return "Any"
        return "Unknown"


class TypeScriptTypeParser:
    """Parse TypeScript type definitions from .ts source."""

    def parse_file(self, filepath: Path) -> Dict[str, Dict[str, str]]:
        """Parse all TypeScript interfaces from a .ts file.

        Returns:
            Dict mapping interface_name -> Dict mapping field_name -> field_type
        """
        with open(filepath) as f:
            source = f.read()

        interfaces = {}
        pattern = r"export\s+interface\s+(\w+)\s*\{([^}]*)\}"
        matches = re.findall(pattern, source, re.MULTILINE)

        for interface_name, body in matches:
            fields = {}
            field_pattern = r"(\w+)(\?)?\s*:\s*([^;\n]+)"
            field_matches = re.findall(field_pattern, body)

            for field_name, optional, field_type in field_matches:
                field_type = field_type.strip()
                if optional:
                    field_type = f"Optional[{field_type}]"
                fields[field_name] = field_type

            interfaces[interface_name] = fields

        return interfaces


class TypeAlignmentChecker:
    """Check alignment between backend Pydantic models and frontend TypeScript types."""

    MAPPING = {
        ("WorkloadInfo", "WorkloadInfo"),
        ("ModelInfo", "ModelInfo"),
        ("WorkloadSchema", "WorkloadSchema"),
        ("ModelSchema", "ModelSchema"),
        ("HardwareSchema", "HardwareSchema"),
        ("StrategySchema", "StrategySchema"),
        ("EvaluationResult", "EvaluationResult"),
        ("ParamSchemaItem", "ParamSchema"),
        ("TopologyInfo", "TopologyInfo"),
    }

    TYPE_MAPPING = {
        "str": "string",
        "int": "number",
        "float": "number",
        "bool": "boolean",
        "Dict": "Record",
        "List": "Array",
        "Optional": "Optional",
        "Any": "any",
    }

    def __init__(self, backend_dir: Path, frontend_dir: Path):
        self.backend_dir = backend_dir
        self.frontend_dir = frontend_dir
        self.pydantic_parser = PydanticFieldParser()
        self.ts_parser = TypeScriptTypeParser()

    def check_alignment(self) -> List[str]:
        """Check alignment and return report."""
        reports = []

        backend_models = {}
        for py_file in self.backend_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            models = self.pydantic_parser.parse_file(py_file)
            backend_models.update(models)

        frontend_types = {}
        for ts_file in self.frontend_dir.glob("*.ts"):
            types = self.ts_parser.parse_file(ts_file)
            frontend_types.update(types)

        for backend_name, frontend_name in self.MAPPING:
            backend_fields = backend_models.get(backend_name, {})
            frontend_fields = frontend_types.get(frontend_name, {})

            if not backend_fields:
                reports.append(f"❌ Backend model '{backend_name}' not found")
                continue

            if not frontend_fields:
                reports.append(f"❌ Frontend type '{frontend_name}' not found")
                continue

            reports.append(f"\n{backend_name} <-> {frontend_name}:")
            reports.append(self._compare_fields(backend_fields, frontend_fields))

        return reports

    def _compare_fields(self, backend: Dict[str, str], frontend: Dict[str, str]) -> str:
        """Compare backend and frontend field definitions."""
        lines = []
        backend_set = set(backend.keys())
        frontend_set = set(frontend.keys())

        common = backend_set & frontend_set
        missing = backend_set - frontend_set
        extra = frontend_set - backend_set

        for field in sorted(common):
            backend_type = self._normalize_type(backend[field])
            frontend_type = frontend[field]
            if self._types_match(backend_type, frontend_type):
                lines.append(f"  ✅ {field}: {backend[field]}")
            else:
                lines.append(f"  ⚠️ {field}: backend={backend[field]}, frontend={frontend_type}")

        for field in sorted(missing):
            lines.append(f"  ❌ {field}: missing in frontend (backend has {backend[field]})")

        for field in sorted(extra):
            lines.append(f"  ⚠️ {field}: extra in frontend (frontend has {frontend[field]})")

        return "\n".join(lines)

    def _normalize_type(self, backend_type: str) -> str:
        """Normalize backend type to match frontend naming."""
        for py_type, ts_type in self.TYPE_MAPPING.items():
            backend_type = backend_type.replace(py_type, ts_type)
        return backend_type

    def _types_match(self, backend_type: str, frontend_type: str) -> bool:
        """Check if backend and frontend types match."""
        backend_type = self._normalize_type(backend_type)
        frontend_type = frontend_type.replace("Optional[", "").replace("]", "")
        backend_type = backend_type.replace("Optional[", "").replace("]", "")

        return backend_type == frontend_type


def main():
    """Main entry point."""
    backend_dir = Path("llm_perf/workload")
    frontend_dir = Path("web2/src/types")

    checker = TypeAlignmentChecker(backend_dir, frontend_dir)
    reports = checker.check_alignment()

    for report in reports:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())