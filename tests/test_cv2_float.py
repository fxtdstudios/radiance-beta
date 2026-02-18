import cv2
import numpy as np

try:
    # Create a dummy float32 image
    img = np.random.rand(100, 100, 3).astype(np.float32)
    
    # Try denoising
    # Note: fastNlMeansDenoising runs on grayscale (single channel) or fastNlMeansDenoisingColored for color
    # But usually these expect uint8. Let's verify float32 support.
    
    # Attempt 1: fastNlMeansDenoising (works on 8-bit usually, let's see)
    # OpenCV docs say: "The function fastNlMeansDenoising denoising the image using Non-local Means Denoising algorithm...
    # Input image array. Should be of type CV_8U" for some versions.
    
    # Let's try bilateralFilter which often supports float32
    denoised_bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    print(f"Bilateral OK: {denoised_bilateral.dtype}")
    
    # Check simple Gaussian
    denoised_gauss = cv2.GaussianBlur(img, (5,5), 0)
    print(f"Gaussian OK: {denoised_gauss.dtype}")

except Exception as e:
    print(f"Error: {e}")
