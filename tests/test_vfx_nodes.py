import pytest
import torch
import json
import os
import shutil

from radiance.nodes.vfx.masking import (
    RadianceSAMModelLoader,
    RadianceSAMGenerator,
    RadianceMultiMaskVisualPicker,
    RadianceLinearMatting,
)
from radiance.nodes.vfx.plate import (
    RadianceHDRGrainMatcher,
    RadianceSubpixelStabilizer,
)
from radiance.nodes.vfx.inpaint import (
    RadianceHDRCrop,
    RadianceHDRStitch,
    RadianceTemporalStitchStabilizer,
)
from radiance.nodes.vfx.roto import (
    RadianceVectorMaskDraw,
    RadianceVideoMaskPropagator,
)
from radiance.core.param_memory import RadianceParamHistoryTracker

def test_masking_suite():
    # 1. Test SAM loader
    loader = RadianceSAMModelLoader()
    sam_model = loader.load("sam2.1_hiera_large.pt", "cpu", False, "float32")[0]
    assert sam_model["model_name"] == "sam2.1_hiera_large.pt"
    assert sam_model["device"] == "cpu"
    
    # 2. Test SAM generator
    generator = RadianceSAMGenerator()
    image = torch.ones((2, 64, 64, 3), dtype=torch.float32)
    mask, masked_img = generator.generate(
        image=image,
        sam_model=sam_model,
        points="[[32, 32]]",
        point_labels="[1]"
    )
    assert mask.shape == (2, 64, 64)
    assert masked_img.shape == (2, 64, 64, 3)
    
    # 3. Test Picker
    picker = RadianceMultiMaskVisualPicker()
    masks_batch = torch.ones((4, 2, 64, 64), dtype=torch.float32)
    picked_mask = picker.pick(masks_batch, 1)[0]
    assert picked_mask.shape == (4, 64, 64)
    
    # 4. Test Matting (Guided Filter)
    matting = RadianceLinearMatting()
    alpha, fore = matting.apply(
        image=image,
        mask=mask,
        method="GuidedFilter",
        trimap_dilation=4,
        eps=1e-4
    )
    assert alpha.shape == (2, 64, 64)
    assert fore.shape == (2, 64, 64, 3)


def test_plate_suite():
    # 1. Test Exposure-Relative Grain Matcher
    target = torch.ones((1, 32, 32, 3), dtype=torch.float32) * 5.0  # high HDR exposure
    ref = torch.ones((1, 32, 32, 3), dtype=torch.float32) * 2.0
    
    # Add random noise to simulate grain
    ref = ref + torch.randn_like(ref) * 0.1
    
    matcher = RadianceHDRGrainMatcher()
    grained = matcher.apply(
        target=target,
        reference=ref,
        intensity=1.0,
        kernel_size=3,
        r_gain=1.0,
        g_gain=1.0,
        b_gain=1.0
    )[0]
    
    assert grained.shape == (1, 32, 32, 3)
    # Check that it didn't clip the high dynamic range output (it stays around 5)
    assert torch.max(grained) > 4.0
    
    # 2. Test Subpixel Stabilizer
    seq = torch.ones((3, 32, 32, 3), dtype=torch.float32)
    stabilizer = RadianceSubpixelStabilizer()
    stab_seq, disp = stabilizer.apply(seq, 0, 16)
    assert stab_seq.shape == (3, 32, 32, 3)
    assert disp.shape == (3, 32, 32, 2)


def test_inpainting_suite():
    image = torch.ones((2, 64, 64, 3), dtype=torch.float32)
    mask = torch.zeros((2, 64, 64), dtype=torch.float32)
    mask[:, 16:48, 16:48] = 1.0  # active region
    
    # 1. Test Crop (multiples of 16)
    cropper = RadianceHDRCrop()
    crop_img, crop_mask, stitch_data = cropper.apply(image, mask, 1.5, 16)
    
    assert crop_img.shape[0] == 2
    assert crop_img.shape[1] % 16 == 0
    assert crop_img.shape[2] % 16 == 0
    assert stitch_data["ymin"] >= 0
    
    # 2. Test Stitch
    stitcher = RadianceHDRStitch()
    stitched_img, blend_mask = stitcher.apply(
        original_image=image,
        cropped_image=crop_img,
        cropped_mask=crop_mask,
        stitcher_data=stitch_data,
        blend_mode="Linear_Laplacian",
        feather_radius=4
    )
    assert stitched_img.shape == (2, 64, 64, 3)
    assert blend_mask.shape == (2, 64, 64)
    
    # 3. Test Temporal Stabilizer
    temp_stabilizer = RadianceTemporalStitchStabilizer()
    smoothed_masks = temp_stabilizer.apply(mask, 1.5)[0]
    assert smoothed_masks.shape == (2, 64, 64)


def test_param_history_tracker(tmp_path):
    tracker = RadianceParamHistoryTracker()
    # Override database path to temporary path for testing
    test_db = os.path.join(tmp_path, "test_history.db")
    tracker.db_path = test_db
    tracker._init_db()
    
    params = {"exposure_offset": 1.5, "compression_ratio": 0.5}
    summary, diff = tracker.record("RadianceHDREncoder", json.dumps(params))
    
    assert "RadianceHDREncoder" in summary
    assert "exposure_offset" in summary
    
    # Second run with changed parameter to verify diffing
    new_params = {"exposure_offset": 2.0, "compression_ratio": 0.5}
    summary2, diff2 = tracker.record("RadianceHDREncoder", json.dumps(new_params))
    
    assert "exposure_offset" in diff2
    assert "1.5" in diff2 or "2" in diff2


def test_roto_suite():
    # 1. Test Vector Mask Draw (Polygon mode)
    drawer = RadianceVectorMaskDraw()
    points_json = "[[10, 10], [50, 10], [50, 50], [10, 50]]"
    mask = drawer.draw(64, 64, "Polygon", points_json, 1.5)[0]
    
    assert mask.shape == (1, 64, 64)
    # Check that mask has rendered active pixels inside the polygon
    assert mask[0, 30, 30].item() > 0.9
    assert mask[0, 2, 2].item() < 0.1
    
    # 2. Test Nuke-style raw format parser
    nuke_points = "{ 10.0 10.0 } { 50.0 10.0 } { 50.0 50.0 } { 10.0 50.0 }"
    mask_nuke = drawer.draw(64, 64, "Polygon", nuke_points, 1.5)[0]
    assert mask_nuke.shape == (1, 64, 64)
    assert mask_nuke[0, 30, 30].item() > 0.9
    
    # 3. Test Video Mask Propagator
    propagator = RadianceVideoMaskPropagator()
    masks_seq = torch.zeros((3, 64, 64), dtype=torch.float32)
    masks_seq[0, 10:20, 10:20] = 1.0  # reference roto frame
    
    # Flow vectors shape [3, H, W, 3] representing +2 pixels shift
    flow = torch.zeros((3, 64, 64, 3), dtype=torch.float32)
    flow[..., 0] = 2.0 # shift right
    flow[..., 1] = 0.0
    
    propagated = propagator.propagate(masks_seq, flow, "Forward")[0]
    
    assert propagated.shape == (3, 64, 64)
    # Check that frame 1 received warped mask shifted by 2 pixels
    assert propagated[1, 15, 17].item() > 0.5
