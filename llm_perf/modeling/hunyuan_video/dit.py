"""HunyuanVideo complete DiT model (backward compatibility).

DEPRECATED: Use llm_perf.modeling.models.hunyuan_video instead.
This module re-exports from the new location for backward compatibility.
"""

from llm_perf.modeling.models.hunyuan_video import ShardedHYVideoDiT

__all__ = ["ShardedHYVideoDiT"]