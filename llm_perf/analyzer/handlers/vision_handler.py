"""Vision model handler.

Handles Vision-specific forward pass logic:
- Convolutional encoder: (batch_size, channels, num_frames, height, width)
- Convolutional decoder: (batch_size, latent_channels, latent_t, latent_h, latent_w)
"""

import logging
from typing import Any, Dict, List

from llm_perf.modeling import ShardedModule
from llm_perf.modeling.tensor import ShardedTensor

from .base_handler import BaseModelHandler

logger = logging.getLogger(__name__)


class VisionHandler(BaseModelHandler):
    """Handler for Vision model types (VAE Encoder, VAE Decoder, ResNet)."""

    def get_seq_len(
        self,
        component: ShardedModule,
        params: Dict[str, Any],
        phase_name: str,
    ) -> int:
        """Compute sequence length for Vision models.
        
        Vision models don't use traditional seq_len.
        This returns pixel count for compatibility with analysis pipeline.
        """
        num_frames = params.get("num_frames", 81)
        height = params.get("height", 720)
        width = params.get("width", 1280)
        
        return num_frames * height * width

    def create_inputs(
        self,
        component: ShardedModule,
        batch_size: int,
        seq_len: int,
        params: Dict[str, Any],
    ) -> List[ShardedTensor]:
        """Create Vision forward inputs.
        
        Vision input structure depends on component type:
        - Conv encoder: raw video/image (batch_size, channels, num_frames, height, width)
        - Conv decoder: latent (batch_size, latent_channels, latent_t, latent_h, latent_w)
        
        Input tensor shape is determined by component attributes.
        """
        num_frames = params.get("num_frames", 81)
        height = params.get("height", 720)
        width = params.get("width", 1280)
        
        in_channels = getattr(component, "in_channels", 3)
        latent_channels = getattr(component, "latent_channels", 16)
        
        if hasattr(component, "block_out_channels"):
            latent_t = (num_frames - 1) // 4 + 1
            latent_h = height // 8
            latent_w = width // 8
            input_tensor = ShardedTensor(
                shape=(batch_size, latent_channels, latent_t, latent_h, latent_w)
            )
        else:
            input_tensor = ShardedTensor(
                shape=(batch_size, in_channels, num_frames, height, width)
            )
        
        return [input_tensor]