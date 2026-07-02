"""Public VAE decoder API.

The maintained implementation lives in :mod:`radiance.fast_vae`; this module
keeps the historical ``radiance.model.vae`` import path stable.
"""
from __future__ import annotations

from radiance.fast_vae import (
    RadianceTurboDecoder,
    RadianceFullDecoder,
    decode_to_linear_realtime,
    detect_rudra_model_type,
    load_radiance_decoder_weights,
    load_turbo_weights,
)

__all__ = [
    "RadianceTurboDecoder",
    "RadianceFullDecoder",
    "decode_to_linear_realtime",
    "detect_rudra_model_type",
    "load_radiance_decoder_weights",
    "load_turbo_weights",
]
