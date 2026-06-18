# test_blur_simulation.py
import cv2
import numpy as np
import rasterio
from pathlib import Path
from src.filters.blur_filter import BlurFilter

def apply_motion_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Simulate satellite motion blur."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

# Load a real B04 band
scene = r"data\extracted\S2A_MSIL1C_20260406T100051_N0512_R122_T32SPF_20260406T151531.SAFE"
granule = list(Path(scene).rglob("GRANULE"))[0]
b04 = list(granule.rglob("*_B04.jp2"))[0]

with rasterio.open(b04) as src:
    data = src.read(1)
    
    # Create motion-blurred version
    blurred = apply_motion_blur(data, kernel_size=21)
    
    # Test filter on both
    bf = BlurFilter(min_variance=15.0)
    
    # Manual check
    def check_variance(img):
        scaled = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        return cv2.Laplacian(scaled, cv2.CV_64F).var()
    
    sharp_var = check_variance(data)
    blurred_var = check_variance(blurred)
    
    print(f"Sharp variance: {sharp_var:.1f}")
    print(f"Blurred variance: {blurred_var:.1f}")
    print(f"BlurFilter would reject blurred: {blurred_var < 100.0}")