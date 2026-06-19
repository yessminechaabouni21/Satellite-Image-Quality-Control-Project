import numpy as np
import rasterio
from pathlib import Path
from .base import BaseFilter, FilterResult


class HazeFilter(BaseFilter):
    """
    Detects haze using Dark Object Subtraction (DOS) approach.
    Haze elevates the minimum reflectance across all bands uniformly.
    Normal scenes: dark objects (shadows, water) have near-zero reflectance.
    Hazy scenes: even dark objects have elevated reflectance.
    """
    
    def __init__(self, 
                 max_dark_object_reflectance: float = 0.03,  # 3% is typical haze floor
                 min_valid_pixels: float = 0.1):  # At least 10% of scene must be dark
        super().__init__()
        self.max_dark_object_reflectance = max_dark_object_reflectance
        self.min_valid_pixels = min_valid_pixels
    
    def apply(self, scene_path: str) -> FilterResult:
        granule = list(Path(scene_path).rglob("GRANULE"))[0]
        b02 = list(granule.rglob("*_B02.jp2"))
        b04 = list(granule.rglob("*_B04.jp2"))
        
        if not b02 or not b04:
            return FilterResult(passed=False, reason="B02 or B04 not found")
        
        with rasterio.open(b02[0]) as src_b, rasterio.open(b04[0]) as src_r:
            # Read at 1/8 resolution for speed
            blue = src_b.read(1, out_shape=(src_b.height//8, src_b.width//8)).astype(np.float32) / 10000
            red = src_r.read(1, out_shape=(src_r.height//8, src_r.width//8)).astype(np.float32) / 10000
            
            # Mask nodata/blackfill
            valid = (blue > 0.001) & (red > 0.001)
            blue = blue[valid]
            red = red[valid]
            
            if len(blue) < 1000:
                return FilterResult(passed=False, reason="Too few valid pixels")
            
            # Dark object: 1st percentile (shadows, water bodies)
            dark_blue = np.percentile(blue, 1)
            dark_red = np.percentile(red, 1)
            
            # Haze elevates BOTH bands at dark end
            # Normal: dark objects ~0.00-0.01
            # Hazy: dark objects > 0.03
            dark_object_avg = (dark_blue + dark_red) / 2
            
            passed = dark_object_avg < self.max_dark_object_reflectance
            
            return FilterResult(
                passed=passed,
                reason=None if passed else f"Dark object reflectance: {dark_object_avg:.4f}",
                metrics={
                    "dark_blue_p1": float(dark_blue),
                    "dark_red_p1": float(dark_red),
                    "dark_object_avg": float(dark_object_avg),
                    "threshold": self.max_dark_object_reflectance
                }
            )