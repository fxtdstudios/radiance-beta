import os
import torch
import pytest
from radiance.model.vae import (
    RadianceTurboDecoder,
    RadianceFullDecoder,
    decode_to_linear_realtime,
    load_radiance_decoder_weights,
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
