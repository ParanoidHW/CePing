"""Base result class with common interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseResult(ABC):
    """Abstract base class for analysis results.

    Defines a minimal interface for all result types:
    - to_dict(): Convert to dictionary for serialization

    Subclasses implement specific metrics and fields.
    """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation.

        Returns:
            Dictionary with result data for serialization
        """
        pass

    def to_json(self) -> str:
        """Convert to JSON string.

        Returns:
            JSON string representation
        """
        import json
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)