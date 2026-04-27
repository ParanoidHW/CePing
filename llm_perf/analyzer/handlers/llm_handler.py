"""LLM (Large Language Model) handler.

Handles LLM-specific forward pass logic:
- Token-based sequence length
- Standard input: (batch_size, seq_len) or (batch_size, seq_len, hidden_size)
"""

import logging
from typing import Any, Dict, List

from llm_perf.modeling import ShardedModule
from llm_perf.modeling.tensor import ShardedTensor

from .base_handler import BaseModelHandler

logger = logging.getLogger(__name__)


class LLMHandler(BaseModelHandler):
    """Handler for LLM model types (Training, Inference)."""

    def get_seq_len(
        self,
        component: ShardedModule,
        params: Dict[str, Any],
        phase_name: str,
    ) -> int:
        """Compute sequence length for LLM.
        
        LLM sequence length logic:
        - prefill: prompt_len or seq_len
        - decode: 1 (single token generation)
        - other: seq_len or prompt_len
        """
        if phase_name == "prefill":
            return params.get("prompt_len", params.get("seq_len", 512))
        elif phase_name == "decode":
            return 1
        else:
            return params.get("seq_len", params.get("prompt_len", 512))

    def create_inputs(
        self,
        component: ShardedModule,
        batch_size: int,
        seq_len: int,
        params: Dict[str, Any],
    ) -> List[ShardedTensor]:
        """Create LLM forward inputs.
        
        LLM input types:
        - vocab_size present: token ids (batch_size, seq_len)
        - otherwise: hidden states (batch_size, seq_len, hidden_size)
        """
        hidden_size = getattr(component, "hidden_size", 4096)
        
        if hasattr(component, "vocab_size"):
            input_tensor = ShardedTensor(shape=(batch_size, seq_len))
        else:
            input_tensor = ShardedTensor(shape=(batch_size, seq_len, hidden_size))
        
        return [input_tensor]