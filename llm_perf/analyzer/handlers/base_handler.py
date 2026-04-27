"""Model type handler base class.

Provides abstract interface for different model types (LLM, Diffusion, Vision).
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from llm_perf.modeling import ShardedModule
from llm_perf.modeling.tensor import ShardedTensor

logger = logging.getLogger(__name__)


class BaseModelHandler(ABC):
    """Abstract base class for model type handlers.
    
    Each handler implements model-type-specific logic for:
    - Computing sequence length
    - Creating forward inputs
    - Executing forward pass
    
    This follows architecture principle: explicit type-based dispatch,
    NOT implicit hasattr checks.
    """

    @abstractmethod
    def get_seq_len(
        self,
        component: ShardedModule,
        params: Dict[str, Any],
        phase_name: str,
    ) -> int:
        """Compute sequence length for this model type.
        
        Args:
            component: ShardedModule instance
            params: Analysis parameters (batch_size, seq_len, height, width, etc.)
            phase_name: Phase name (prefill, decode, denoise, etc.)
        
        Returns:
            Sequence length for this phase
        """
        pass

    @abstractmethod
    def create_inputs(
        self,
        component: ShardedModule,
        batch_size: int,
        seq_len: int,
        params: Dict[str, Any],
    ) -> List[ShardedTensor]:
        """Create forward pass inputs for this model type.
        
        Args:
            component: ShardedModule instance
            batch_size: Batch size
            seq_len: Sequence length
            params: Additional parameters
        
        Returns:
            List of ShardedTensor inputs
        """
        pass

    def forward(
        self,
        component: ShardedModule,
        inputs: List[ShardedTensor],
    ) -> Any:
        """Execute forward pass with explicit error handling.
        
        Args:
            component: ShardedModule instance
            inputs: List of ShardedTensor inputs
        
        Returns:
            Forward pass output
        
        Raises:
            RuntimeError: If forward pass fails
        """
        try:
            return component(*inputs)
        except Exception as e:
            logger.error(
                f"[FORWARD_FAILED] {component.__class__.__name__}: {e}"
            )
            raise RuntimeError(
                f"Forward pass failed for {component.__class__.__name__}: {e}"
            )