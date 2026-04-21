"""ValidationError - Validation error data structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class ValidationLevel(Enum):
    """Validation error level."""
    ERROR = "error"
    WARNING = "warning"


class ValidationCategory(Enum):
    """Validation error category."""
    STRATEGY = "strategy"
    MODEL = "model"
    SEQUENCE = "sequence"
    MEMORY = "memory"
    SPECIAL = "special"


@dataclass
class ValidationError:
    """Single validation error.
    
    Attributes:
        level: Error level (error/warning)
        category: Error category
        code: Error code for programmatic handling
        message: Human-readable error message
        suggestion: Suggested fix
        details: Additional context details
    """
    level: ValidationLevel
    category: ValidationCategory
    code: str
    message: str
    suggestion: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "level": self.level.value,
            "category": self.category.value,
            "code": self.code,
            "message": self.message,
        }
        if self.suggestion:
            result["suggestion"] = self.suggestion
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class ValidationErrors:
    """Collection of validation errors.
    
    Attributes:
        errors: List of ValidationError
        warnings: List of ValidationError (level=WARNING)
    """
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def add_error(self, error: ValidationError):
        """Add a validation error."""
        if error.level == ValidationLevel.ERROR:
            self.errors.append(error)
        else:
            self.warnings.append(error)

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def merge(self, other: "ValidationErrors"):
        """Merge another ValidationErrors into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "has_errors": self.has_errors(),
            "has_warnings": self.has_warnings(),
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }

    def __bool__(self) -> bool:
        """Return True if there are any errors or warnings."""
        return self.has_errors() or self.has_warnings()

    def __repr__(self) -> str:
        return f"ValidationErrors(errors={len(self.errors)}, warnings={len(self.warnings)})"