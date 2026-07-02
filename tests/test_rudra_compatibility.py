import os
import math
import torch
import pytest
import radiance.fast_vae as fast_vae_mod
from radiance.config.model_map import MODEL_VAE_CONFIG, resolve_model_vae_config
from radiance.model.vae import (
    RadianceTurboDecoder,
    RadianceFullDecoder,
    decode_to_linear_realtime,
    load_radiance_decoder_weights,
)

HAS_TORCH = isinstance(getattr(torch, "__version__", None), str)

pytestmark = pytest.mark.skipif(
    not HAS_TORCH,
    reason="RUDRA decoder compatibility tests require real torch, not the CI import stub.",
)


def test_decoder_instantiation():
    """Verify that decoders instantiate with correct channels and have RUDRA structure."""
    turbo = RadianceTurboDecoder(latent_channels=16, dr_dim=64)
    full = RadianceFullDecoder(latent_channels=16, dr_dim=64)

    assert isinstance(turbo, torch.nn.Module)
    assert isinstance(full, torch.nn.Module)
    assert hasattr(turbo, "dr_dim")
    assert hasattr(full, "dr_dim")
    assert turbo.dr_dim == 64
    assert full.dr_dim == 64

def test_rudra_forward_pass_with_conditioning():
    """Verify that RadianceTurboDecoder runs a forward pass successfully with explicit conditioning."""
    device = torch.device("cpu")
    decoder = RadianceTurboDecoder(latent_channels=16, dr_dim=64).to(device)
    
    latent = torch.randn(1, 16, 32, 32, device=device)
    dr_proj = torch.randn(1, 64, device=device)
    
    out = decoder(latent, dr_proj)
    assert out.shape == (1, 3, 256, 256)

def test_rudra_realtime_decode_with_predictor_fallback():
    """Verify that decode_to_linear_realtime automatically falls back to latent prediction when dr_proj is missing."""
    device = torch.device("cpu")
    decoder = RadianceTurboDecoder(latent_channels=16, dr_dim=64).to(device)
    
    latent = torch.randn(1, 16, 16, 16, device=device)
    
    # Run realtime decode with no dr_proj — it should auto-create predictor and decode
    out = decode_to_linear_realtime(
        latent=latent,
        decoder=decoder,
        model_type="flux",
        scale_factor=0.18215,
        tiled=False,
    )
    
    assert out.shape == (1, 128, 128, 3)
    assert hasattr(decoder, "predictor")

def test_mock_rudra_checkpoint_load(tmp_path):
    """Verify that load_radiance_decoder_weights loads a mock RUDRA checkpoint successfully."""
    # Create a mock RUDRA state dict
    decoder = RadianceTurboDecoder(latent_channels=16, dr_dim=64)
    state_dict = decoder.state_dict()
    
    ckpt_path = os.path.join(tmp_path, "rudra_test.pth")
    torch.save(state_dict, ckpt_path)
    
    # Load using our updated weight loader
    loaded = load_radiance_decoder_weights(
        model_type="flux",
        model_size="turbo",
        checkpoint_path=ckpt_path,
    )
    
    assert isinstance(loaded, RadianceTurboDecoder)
    assert loaded.dr_dim == 64


@pytest.mark.parametrize("model_type,cfg", sorted(MODEL_VAE_CONFIG.items()))
@pytest.mark.parametrize(
    "decoder_cls",
    [RadianceTurboDecoder, RadianceFullDecoder],
    ids=["turbo", "full"],
)
def test_rudra_turbo_full_forward_all_model_configs(model_type, cfg, decoder_cls):
    """Every configured VAE family must have a matching Turbo/Full RUDRA shape."""
    latent_channels = cfg.get("latent_channels", 16)
    spatial_scale = cfg.get("vae_spatial_factor", 8)
    n_upsample = max(1, int(round(math.log2(spatial_scale))))
    latent_size = 2 if spatial_scale > 8 else 4

    decoder = decoder_cls(
        latent_channels=latent_channels,
        n_upsample=n_upsample,
        dr_dim=64,
    ).eval()
    latent = torch.randn(1, latent_channels, latent_size, latent_size)
    dr_proj = torch.randn(1, 64)

    with torch.no_grad():
        raw = decoder(latent, dr_proj)
        decoded = decode_to_linear_realtime(
            latent=latent,
            decoder=decoder,
            model_type=model_type,
            precision="fp32",
            return_log_coded=True,
            tiled=False,
        )

    expected_hw = latent_size * spatial_scale
    assert raw.shape == (1, 3, expected_hw, expected_hw)
    assert decoded.shape == (1, expected_hw, expected_hw, 3)


@pytest.mark.parametrize("model_type", ["flux2-klein", "ltx-video"])
def test_rudra_tiled_decode_uses_model_spatial_scale(model_type):
    """Tiled decode must not hard-code the legacy 8x VAE scale."""
    cfg = resolve_model_vae_config(model_type)
    latent_channels = cfg["latent_channels"]
    spatial_scale = cfg["vae_spatial_factor"]
    n_upsample = max(1, int(round(math.log2(spatial_scale))))

    decoder = RadianceTurboDecoder(
        latent_channels=latent_channels,
        n_upsample=n_upsample,
        dr_dim=64,
    ).eval()
    latent = torch.randn(1, latent_channels, 2, 2)

    with torch.no_grad():
        decoded = decode_to_linear_realtime(
            latent=latent,
            decoder=decoder,
            model_type=model_type,
            precision="fp32",
            return_log_coded=True,
            tiled=True,
            tile_size=spatial_scale,
            overlap=0,
        )

    expected_hw = 2 * spatial_scale
    assert decoded.shape == (1, expected_hw, expected_hw, 3)


@pytest.mark.parametrize("model_type,cfg", sorted(MODEL_VAE_CONFIG.items()))
@pytest.mark.parametrize(
    "model_size,decoder_cls",
    [("rudra_turbo", RadianceTurboDecoder), ("rudra_full", RadianceFullDecoder)],
    ids=["turbo", "full"],
)
def test_mock_rudra_checkpoint_loads_for_all_model_configs(
    tmp_path,
    monkeypatch,
    model_type,
    cfg,
    model_size,
    decoder_cls,
):
    """The loader must accept Turbo/Full checkpoints for every configured model family."""
    latent_channels = cfg.get("latent_channels", 16)
    spatial_scale = cfg.get("vae_spatial_factor", 8)
    n_upsample = max(1, int(round(math.log2(spatial_scale))))
    source = decoder_cls(
        latent_channels=latent_channels,
        n_upsample=n_upsample,
        dr_dim=64,
    )
    state_dict = source.state_dict()
    ckpt_path = tmp_path / f"{model_type}_{model_size}.pth"
    ckpt_path.write_bytes(b"mock")

    fast_vae_mod._TRAINED_DECODER_CACHE.clear()
    monkeypatch.setattr(fast_vae_mod.torch, "load", lambda *args, **kwargs: state_dict)

    loaded = load_radiance_decoder_weights(
        model_type=model_type,
        model_size=model_size,
        checkpoint_path=str(ckpt_path),
    )

    assert isinstance(loaded, decoder_cls)
    assert loaded.latent_channels == latent_channels
    assert loaded.n_upsample == n_upsample
    assert loaded.dr_dim == 64
