"""Complete transformer models (backward compatibility).

DEPRECATED: Use llm_perf.modeling.models.llama and llm_perf.modeling.models.deepseek instead.
This module re-exports from the new location for backward compatibility.
"""

from llm_perf.modeling.models.llama import ShardedTransformerBlock, LlamaModel
from llm_perf.modeling.models.deepseek import ShardedMoEBlock, DeepSeekModel

__all__ = [
    "ShardedTransformerBlock",
    "LlamaModel",
    "ShardedMoEBlock",
    "DeepSeekModel",
]