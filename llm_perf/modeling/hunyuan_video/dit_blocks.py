"""HunyuanVideo DiT blocks (backward compatibility).

DEPRECATED: Use llm_perf.modeling.base.dit_blocks instead.
This module re-exports from the new location for backward compatibility.
"""

from llm_perf.modeling.base.dit_blocks import (
    ShardedMMDoubleStreamBlock,
    ShardedMMSingleStreamBlock,
)

__all__ = [
    "ShardedMMDoubleStreamBlock",
    "ShardedMMSingleStreamBlock",
]