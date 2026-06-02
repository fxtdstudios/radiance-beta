"""Model loading, caching, and architecture detection."""
from radiance.model.vae import (
    RadianceTurboDecoder,
    RadianceFullDecoder,
    load_radiance_decoder_weights,
    decode_to_linear_realtime,
)
from radiance.model.cache import LRUCache, get_model_cache, clear_model_caches
from radiance.model.detect import (
    detect_model_type,
    LATENT_CHANNELS,
    latent_format,
    CLIP_SLOT_ORDER,
    assemble_clip_paths,
    get_clip_type_enum,
    estimate_vram_usage,
)

__all__ = [
    "RadianceTurboDecoder",
    "RadianceFullDecoder",
    "load_radiance_decoder_weights",
    "decode_to_linear_realtime",
    "LRUCache",
    "get_model_cache",
    "clear_model_caches",
    "detect_model_type",
    "LATENT_CHANNELS",
    "latent_format",
    "CLIP_SLOT_ORDER",
    "assemble_clip_paths",
    "get_clip_type_enum",
    "estimate_vram_usage",
]
