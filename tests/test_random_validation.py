"""
Randomized Testing Suite for Radiance Nodes
Generates random test cases to validate node robustness.
"""

import torch
import numpy as np
import random
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class RandomNodeTester:
    """Generate random test cases for all nodes."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        
    def random_image_tensor(self, batch=1, height=None, width=None, channels=3):
        """Generate random image tensor."""
        h = height or random.randint(64, 512)
        w = width or random.randint(64, 512)
        return torch.rand(batch, h, w, channels, dtype=torch.float32)
    
    def random_hdr_tensor(self, batch=1, height=128, width=128):
        """Generate random HDR tensor with values > 1.0."""
        img = torch.rand(batch, height, width, 3, dtype=torch.float32) * 10.0
        return img
    
    def test_color_nodes(self):
        """Test color grading nodes."""
        print("\\nTesting Color Grading Nodes...")
        print("-" * 60)
        
        try:
            from hdr.color import RadianceColorCorrect32
            node = RadianceColorCorrect32()
            
            # Random test cases
            for i in range(5):
                img = self.random_image_tensor()
                exposure = random.uniform(-3.0, 3.0)
                gamma = random.uniform(0.5, 2.5)
                saturation = random.uniform(0.0, 2.0)
                
                try:
                    result = node.correct(img, exposure=exposure, gamma=gamma, saturation=saturation)
                    assert result[0].shape == img.shape, "Output shape mismatch"
                    self.passed += 1
                    print(f"  ✓ Test {i+1}: ColorCorrect (exp={exposure:.2f}, gamma={gamma:.2f})")
                except Exception as e:
                    self.failed += 1
                    self.errors.append(f"ColorCorrect test {i+1}: {e}")
                    print(f"  ✗ Test {i+1} FAILED: {e}")
                    
        except ImportError as e:
            print(f"  ⚠ Skipped: {e}")
    
    def test_hdr_nodes(self):
        """Test HDR tone mapping nodes."""
        print("\\nTesting HDR Tone Mapping Nodes...")
        print("-" * 60)
        
        try:
            from hdr.tonemap import RadianceHDRTonemap
            node = RadianceHDRTonemap()
            
            for i in range(5):
                img = self.random_hdr_tensor()
                operator = random.choice(["Reinhard", "Filmic", "ACES", "AgX"])
                white_point = random.uniform(1.0, 10.0)
                
                try:
                    result = node.tonemap(img, operator=operator, white_point=white_point)
                    assert result[0].shape == img.shape
                    assert torch.all(result[0] >= 0.0) and torch.all(result[0] <= 1.0), "Values outside [0,1]"
                    self.passed += 1
                    print(f"  ✓ Test {i+1}: HDRTonemap ({operator}, wp={white_point:.2f})")
                except Exception as e:
                    self.failed += 1
                    self.errors.append(f"HDRTonemap test {i+1}: {e}")
                    print(f"  ✗ Test {i+1} FAILED: {e}")
                    
        except ImportError as e:
            print(f"  ⚠ Skipped: {e}")
    
    def test_image_processing(self):
        """Test image processing nodes."""
        print("\\nTesting Image Processing Nodes...")
        print("-" * 60)
        
        try:
            from image.upscale import RadianceUpscale
            node = RadianceUpscale()
            
            for i in range(3):
                img = self.random_image_tensor(height=128, width=128)
                scale = random.choice([2.0, 4.0])
                method = random.choice(["bicubic", "lanczos", "nearest"])
                
                try:
                    result = node.upscale(img, scale_factor=scale, method=method)
                    expected_h = int(128 * scale)
                    expected_w = int(128 * scale)
                    assert result[0].shape[1] == expected_h, f"Height mismatch: {result[0].shape}"
                    assert result[0].shape[2] == expected_w, f"Width mismatch: {result[0].shape}"
                    self.passed += 1
                    print(f"  ✓ Test {i+1}: Upscale ({scale}x, {method})")
                except Exception as e:
                    self.failed += 1
                    self.errors.append(f"Upscale test {i+1}: {e}")
                    print(f"  ✗ Test {i+1} FAILED: {e}")
                    
        except ImportError as e:
            print(f"  ⚠ Skipped: {e}")
    
    def test_film_grain(self):
        """Test film grain nodes."""
        print("\\nTesting Film Grain Nodes...")
        print("-" * 60)
        
        try:
            from film.grain import RadianceFilmGrainPro
            node = RadianceFilmGrainPro()
            
            for i in range(3):
                img = self.random_image_tensor()
                intensity = random.uniform(0.0, 0.5)
                size = random.uniform(0.5, 2.0)
                
                try:
                    result = node.apply_grain(img, intensity=intensity, size=size, seed=random.randint(0, 999999))
                    assert result[0].shape == img.shape
                    self.passed += 1
                    print(f"  ✓ Test {i+1}: FilmGrain (intensity={intensity:.2f}, size={size:.2f})")
                except Exception as e:
                    self.failed += 1
                    self.errors.append(f"FilmGrain test {i+1}: {e}")
                    print(f"  ✗ Test {i+1} FAILED: {e}")
                    
        except ImportError as e:
            print(f"  ⚠ Skipped: {e}")
    
    def run_all_tests(self):
        """Run all randomized tests."""
        print("=" * 80)
        print("RADIANCE RANDOMIZED NODE TESTING")
        print("=" * 80)
        
        self.test_color_nodes()
        self.test_hdr_nodes()
        self.test_image_processing()
        self.test_film_grain()
        
        # Summary
        print("\\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Passed:  {self.passed}")
        print(f"Failed:  {self.failed}")
        print(f"Total:   {self.passed + self.failed}")
        
        if self.failed > 0:
            print(f"\\nSuccess Rate: {100 * self.passed / (self.passed + self.failed):.1f}%")
            print("\\nFailed Tests:")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("\\n✅ ALL TESTS PASSED!")
        
        return self.failed == 0


if __name__ == "__main__":
    tester = RandomNodeTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
