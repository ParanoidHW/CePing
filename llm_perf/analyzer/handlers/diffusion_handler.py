"""Diffusion model handler.

Handles Diffusion-specific forward pass logic:
- Latent-based sequence length (height * width after VAE compression)
- Diffusion input: (batch_size, seq_len, latent_channels) + timestep
"""

import logging
from typing import Any, Dict, List

from llm_perf.modeling import ShardedModule
from llm_perf.modeling.tensor import ShardedTensor

from .base_handler import BaseModelHandler

logger = logging.getLogger(__name__)


class DiffusionHandler(BaseModelHandler):
    """Handler for Diffusion model types (DiT, VAE, etc.)."""

    def get_seq_len(
        self,
        component: ShardedModule,
        params: Dict[str, Any],
        phase_name: str,
    ) -> int:
        """Compute sequence length for Diffusion models.
        
        Diffusion sequence length is latent spatial size:
        - seq_len = (height // vae_compression_ratio) * (width // vae_compression_ratio)
        
        VAE compression ratio is typically 16 for video diffusion (e.g., Wan2.1).
        """
        height = params.get("height", 720)
        width = params.get("width", 1280)
        vae_compression_ratio = getattr(component, "vae_compression_ratio", 16)
        
        latent_h = height // vae_compression_ratio
        latent_w = width // vae_compression_ratio
        
        return latent_h * latent_w

    def create_inputs(
        self,
        component: ShardedModule,
        batch_size: int,
        seq_len: int,
        params: Dict[str, Any],
    ) -> List[ShardedTensor]:
        """Create Diffusion forward inputs.
        
        Diffusion input structure:
        - Standard: latent (batch_size, seq_len, latent_channels) + timestep
        - Wan-style with text/freq embedding: latent + text_embed + time_embed
        
        timestep_dim is inferred from:
        1. component.freq_dim (preferred)
        2. component.timestep_in_weight.shape[0]
        3. default 256
        """
        latent_channels = getattr(component, "latent_channels", 16)
        latent = ShardedTensor(shape=(batch_size, seq_len, latent_channels))
        
        if hasattr(component, "text_dim") and hasattr(component, "freq_dim"):
            text_dim = component.text_dim
            freq_dim = component.freq_dim
            text_embed = ShardedTensor(shape=(batch_size, 256, text_dim))
            time_embed = ShardedTensor(shape=(batch_size, freq_dim))
            return [latent, text_embed, time_embed]
        
        timestep_dim = getattr(component, "freq_dim", None)
        if timestep_dim is None and hasattr(component, "timestep_in_weight"):
            timestep_dim = component.timestep_in_weight.shape[0]
        if timestep_dim is None:
            timestep_dim = 256
        
        timestep = ShardedTensor(shape=(batch_size, timestep_dim))
        return [latent, timestep]