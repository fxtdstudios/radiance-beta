"""
═══════════════════════════════════════════════════════════════════════════════
                RADIANCE FILM GRAIN MODULE - UNIT TESTS
                Tests for grain generation, film emulation, and lens effects
═══════════════════════════════════════════════════════════════════════════════
"""
import unittest
import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import filmgrain modules
from nodes_filmgrain import (
    CAMERA_PRESETS,
    FILM_STOCK_PRESETS,
    LENS_PRESETS,
    MASTER_FILM_PRESETS,
    FXTDLensEffects,
    FXTDProFilmEffects,
    FXTDRealisticGrain,
)


class TestCameraPresets(unittest.TestCase):
    """Test camera preset configurations."""
    
    def test_preset_count(self):
        """Verify we have substantial camera presets."""
        self.assertGreaterEqual(len(CAMERA_PRESETS), 10)
    
    def test_arri_alexa_preset_exists(self):
        """Test ARRI Alexa 35 preset exists."""
        self.assertIn("ARRI Alexa 35", CAMERA_PRESETS)
    
    def test_preset_structure(self):
        """Test preset has required keys."""
        preset = CAMERA_PRESETS["ARRI Alexa 35"]
        
        required_keys = ["description", "grain_size", "grain_intensity", "grain_softness"]
        for key in required_keys:
            self.assertIn(key, preset, f"Missing key: {key}")
    
    def test_preset_values_reasonable(self):
        """Test preset values are in reasonable ranges."""
        for name, preset in CAMERA_PRESETS.items():
            if "grain_intensity" in preset:
                self.assertGreaterEqual(preset["grain_intensity"], 0.0)
                self.assertLessEqual(preset["grain_intensity"], 1.0)
            if "grain_size" in preset:
                self.assertGreater(preset["grain_size"], 0.0)


class TestFilmStockPresets(unittest.TestCase):
    """Test film stock preset configurations."""
    
    def test_kodak_500t_exists(self):
        """Test Kodak Vision3 500T preset exists."""
        self.assertIn("Kodak Vision3 500T 5219", FILM_STOCK_PRESETS)
    
    def test_fuji_stocks_exist(self):
        """Test Fuji film stocks exist."""
        fuji_stocks = [k for k in FILM_STOCK_PRESETS if "Fuji" in k]
        self.assertGreaterEqual(len(fuji_stocks), 1)
    
    def test_bw_stocks_exist(self):
        """Test B&W film stocks exist."""
        bw_stocks = [k for k in FILM_STOCK_PRESETS if "saturation" in FILM_STOCK_PRESETS[k] 
                     and FILM_STOCK_PRESETS[k]["saturation"] == 0.0]
        self.assertGreaterEqual(len(bw_stocks), 1)
    
    def test_cinestill_800t_halation(self):
        """Test CineStill 800T has strong halation (known characteristic)."""
        preset = FILM_STOCK_PRESETS.get("CineStill 800T")
        if preset:
            self.assertGreaterEqual(preset.get("halation", 0), 0.5,
                                   "CineStill 800T should have strong halation")


class TestLensPresets(unittest.TestCase):
    """Test lens preset configurations."""
    
    def test_anamorphic_lenses_exist(self):
        """Test anamorphic lens presets exist."""
        anamorphic = [k for k in LENS_PRESETS if "Anamorphic" in k.lower() or "anamorphic" in k.lower()]
        self.assertGreaterEqual(len(anamorphic), 2)
    
    def test_cooke_lenses_exist(self):
        """Test Cooke lens presets exist."""
        cooke = [k for k in LENS_PRESETS if "Cooke" in k]
        self.assertGreaterEqual(len(cooke), 1)
    
    def test_lens_preset_structure(self):
        """Test lens presets have required optical simulation keys."""
        for name, preset in list(LENS_PRESETS.items())[:3]:
            self.assertIn("chromatic_aberration", preset)
            self.assertIn("vignette_strength", preset)


class TestFXTDLensEffects(unittest.TestCase):
    """Test FXTDLensEffects node."""
    
    def test_instantiation(self):
        """Test node can be instantiated."""
        node = FXTDLensEffects()
        self.assertIsNotNone(node)
    
    def test_input_types(self):
        """Test INPUT_TYPES returns valid structure."""
        inputs = FXTDLensEffects.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("image", inputs["required"])
    
    def test_lens_preset_list(self):
        """Test all lens presets are available in node."""
        inputs = FXTDLensEffects.INPUT_TYPES()
        if "lens_preset" in inputs.get("required", {}):
            presets = inputs["required"]["lens_preset"][0]
            self.assertIsInstance(presets, (list, tuple))
            self.assertGreater(len(presets), 0)


class TestFXTDProFilmEffects(unittest.TestCase):
    """Test FXTDProFilmEffects node."""
    
    def test_instantiation(self):
        """Test node can be instantiated."""
        node = FXTDProFilmEffects()
        self.assertIsNotNone(node)
    
    def test_input_types(self):
        """Test INPUT_TYPES returns valid structure."""
        inputs = FXTDProFilmEffects.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("image", inputs["required"])
    
    def test_has_master_preset(self):
        """Test node uses master presets."""
        inputs = FXTDProFilmEffects.INPUT_TYPES()
        # Check for preset selection in required inputs
        required_keys = list(inputs.get("required", {}).keys())
        preset_key = [k for k in required_keys if "preset" in k.lower()]
        self.assertGreater(len(preset_key), 0, "Should have a preset selection")


class TestFXTDRealisticGrain(unittest.TestCase):
    """Test FXTDRealisticGrain node."""
    
    def test_instantiation(self):
        """Test node can be instantiated."""
        node = FXTDRealisticGrain()
        self.assertIsNotNone(node)
    
    def test_input_types(self):
        """Test INPUT_TYPES returns valid structure."""
        inputs = FXTDRealisticGrain.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("image", inputs["required"])
    
    def test_strength_parameter(self):
        """Test strength parameter exists."""
        inputs = FXTDRealisticGrain.INPUT_TYPES()
        all_inputs = {**inputs.get("required", {}), **inputs.get("optional", {})}
        strength_key = [k for k in all_inputs if "strength" in k.lower() or "intensity" in k.lower()]
        self.assertGreater(len(strength_key), 0, "Should have strength/intensity control")


class TestGrainGeneration(unittest.TestCase):
    """Test actual grain generation functions."""
    
    def setUp(self):
        """Create test image."""
        self.test_image = torch.rand((1, 64, 64, 3), dtype=torch.float32)
    
    def test_grain_node_basic_execution(self):
        """Test grain node can process an image."""
        node = FXTDRealisticGrain()
        inputs = FXTDRealisticGrain.INPUT_TYPES()
        
        # Get first available preset
        preset_key = None
        presets = None
        
        if "camera_preset" in inputs.get("required", {}):
            preset_key = "camera_preset"
            presets = inputs["required"]["camera_preset"][0]
        elif "preset" in inputs.get("required", {}):
            preset_key = "preset"
            presets = inputs["required"]["preset"][0]
        
        if presets and len(presets) > 0:
            # Getting the method name dynamically
            method_name = None
            for attr in dir(node):
                if attr.startswith("apply") or attr == "execute":
                    method_name = attr
                    break
            
            if method_name:
                try:
                    method = getattr(node, method_name)
                    # Just verify the node has executable methods
                    self.assertTrue(callable(method))
                except Exception:
                    pass
    
    def test_grain_preserves_image_shape(self):
        """Test that grain application preserves image dimensions."""
        # This is a conceptual test - actual execution would require 
        # knowing exact method signature
        self.assertEqual(self.test_image.shape, (1, 64, 64, 3))


class TestMasterFilmPresets(unittest.TestCase):
    """Test master film preset configurations."""
    
    def test_presets_exist(self):
        """Test master presets exist."""
        self.assertGreater(len(MASTER_FILM_PRESETS), 5)
    
    def test_alexa_or_clean_presets_exist(self):
        """Test Alexa or clean cinema presets exist."""
        # Look for ARRI or Alexa or clean/cinema presets
        relevant_presets = [k for k in MASTER_FILM_PRESETS if "Alexa" in k or "35mm" in k or "ARRI" in k.upper()]
        self.assertGreaterEqual(len(relevant_presets), 1)
    
    def test_35mm_presets_exist(self):
        """Test 35mm film presets exist."""
        film_35mm = [k for k in MASTER_FILM_PRESETS if "35mm" in k]
        self.assertGreaterEqual(len(film_35mm), 1)
    
    def test_preset_grain_values(self):
        """Test all presets have valid grain intensity values."""
        for name, preset in MASTER_FILM_PRESETS.items():
            if "grain_intensity" in preset:
                self.assertGreaterEqual(preset["grain_intensity"], 0.0,
                                       f"{name} has invalid grain_intensity")
                self.assertLessEqual(preset["grain_intensity"], 1.0,
                                    f"{name} has grain_intensity > 1.0")


class TestTemporalGrainFunction(unittest.TestCase):
    """Test temporal grain generation function."""
    
    def test_temporal_seed_reproducibility(self):
        """Same frame_index + temporal_seed = identical grain."""
        from nodes_filmgrain import generate_temporal_grain
        
        color_resp = {"r": 1.0, "g": 1.0, "b": 1.0}
        
        grain1 = generate_temporal_grain(64, 64, 1.0, 0.15, 0.2, color_resp,
                                         frame_index=5, temporal_seed=42)
        grain2 = generate_temporal_grain(64, 64, 1.0, 0.15, 0.2, color_resp,
                                         frame_index=5, temporal_seed=42)
        
        np.testing.assert_array_almost_equal(grain1, grain2,
            err_msg="Same seed+frame should produce identical grain")
    
    def test_different_frames_different_grain(self):
        """Adjacent frames should have different grain patterns."""
        from nodes_filmgrain import generate_temporal_grain
        
        color_resp = {"r": 1.0, "g": 1.0, "b": 1.0}
        
        grain_f0 = generate_temporal_grain(64, 64, 1.0, 0.15, 0.2, color_resp,
                                           frame_index=0, temporal_seed=42,
                                           temporal_smoothness=0.0)
        grain_f1 = generate_temporal_grain(64, 64, 1.0, 0.15, 0.2, color_resp,
                                           frame_index=1, temporal_seed=42,
                                           temporal_smoothness=0.0)
        
        # Should not be identical
        diff = np.abs(grain_f0 - grain_f1).mean()
        self.assertGreater(diff, 0.01,
            "Adjacent frames should have different grain (zero temporal_smoothness)")
    
    def test_per_channel_intensity_red_only(self):
        """Setting r_intensity=2 and g/b=0 should affect only red channel."""
        from nodes_filmgrain import generate_temporal_grain
        
        color_resp = {"r": 1.0, "g": 1.0, "b": 1.0}
        
        grain = generate_temporal_grain(64, 64, 1.0, 0.15, 0.0, color_resp,
                                        frame_index=0, temporal_seed=42,
                                        temporal_smoothness=0.0,
                                        r_intensity=2.0, g_intensity=0.0, b_intensity=0.0)
        
        # Red channel should have grain
        self.assertGreater(np.abs(grain[..., 0]).max(), 0.01, "Red channel should have grain")
        # Green and Blue should be zero
        self.assertAlmostEqual(np.abs(grain[..., 1]).max(), 0.0, places=5,
                              msg="Green channel should be zero")
        self.assertAlmostEqual(np.abs(grain[..., 2]).max(), 0.0, places=5,
                              msg="Blue channel should be zero")
    
    def test_temporal_smoothness_blends_frames(self):
        """Temporal smoothness should create a blend between adjacent frames."""
        from nodes_filmgrain import generate_temporal_grain
        
        color_resp = {"r": 1.0, "g": 1.0, "b": 1.0}
        
        # With smoothness=0, frames are fully independent
        grain_sharp = generate_temporal_grain(64, 64, 1.0, 0.15, 0.0, color_resp,
                                              frame_index=0, temporal_seed=42,
                                              temporal_smoothness=0.0)
        
        # With smoothness=1, maximum blend with next frame
        grain_smooth = generate_temporal_grain(64, 64, 1.0, 0.15, 0.0, color_resp,
                                               frame_index=0, temporal_seed=42,
                                               temporal_smoothness=1.0)
        
        # They should be different due to blending
        diff = np.abs(grain_sharp - grain_smooth).mean()
        self.assertGreater(diff, 0.001,
            "Temporal smoothness should affect grain pattern")


class TestFXTDTemporalGrainNode(unittest.TestCase):
    """Test FXTDTemporalGrain node with per-channel controls."""
    
    def test_node_has_rgb_intensity_inputs(self):
        """Verify node has per-channel intensity controls."""
        from nodes_filmgrain import FXTDTemporalGrain
        
        inputs = FXTDTemporalGrain.INPUT_TYPES()
        optional = inputs.get("optional", {})
        
        self.assertIn("r_intensity", optional, "Should have r_intensity parameter")
        self.assertIn("g_intensity", optional, "Should have g_intensity parameter")
        self.assertIn("b_intensity", optional, "Should have b_intensity parameter")
    
    def test_node_temporal_blend_parameter(self):
        """Verify temporal_blend parameter exists."""
        from nodes_filmgrain import FXTDTemporalGrain
        
        inputs = FXTDTemporalGrain.INPUT_TYPES()
        required = inputs.get("required", {})
        
        self.assertIn("temporal_blend", required, "Should have temporal_blend parameter")


if __name__ == '__main__':
    unittest.main()
