import numpy as np
import rasterio
import cv2
from pathlib import Path
from .base import BaseFilter, FilterResult


class BlurFilter(BaseFilter):
    """
    Detects blurry scenes using Laplacian variance at native resolution.
    """
    
    def __init__(self, min_variance: float = 15.0):  # Lowered from 100
        super().__init__()
        self.min_variance = min_variance
    
    def apply(self, scene_path: str) -> FilterResult:
        granule = list(Path(scene_path).rglob("GRANULE"))[0]
        band_files = list(granule.rglob("*_B04.jp2"))
        
        if not band_files:
            return FilterResult(passed=False, reason="B04 not found")
        
        with rasterio.open(band_files[0]) as src:
            # Read at 1/2 resolution (better balance of speed vs accuracy)
            h, w = src.height // 2, src.width // 2
            data = src.read(1, out_shape=(h, w))
            
            # Scale to 0-255 for OpenCV
            dmin, dmax = data.min(), data.max()
            if dmax == dmin:
                return FilterResult(passed=False, reason="Uniform image (dead band)")
            
            scaled = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
            
            # Laplacian variance
            laplacian = cv2.Laplacian(scaled, cv2.CV_64F)
            variance = laplacian.var()
            
            passed = variance >= self.min_variance
            
            return FilterResult(
                passed=passed,
                reason=None if passed else f"Too blurry (variance: {variance:.1f} < {self.min_variance})",
                metrics={
                    "laplacian_variance": float(variance),
                    "threshold": self.min_variance,
                    "resolution_checked": f"{h}x{w}"
                }
            )