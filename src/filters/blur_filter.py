# src/filters/blur_filter.py
import numpy as np
import rasterio
import cv2
from pathlib import Path
from .base import BaseFilter, FilterResult


class BlurFilter(BaseFilter):
    """
    Detects blurry or out-of-focus scenes using Laplacian variance.
    Low variance = blurry / uniform (bad)
    High variance = sharp / textured (good)
    """
    
    def __init__(self, min_variance: float = 100.0):
        super().__init__()
        self.min_variance = min_variance
    
    def apply(self, scene_path: str) -> FilterResult:
        granule = list(Path(scene_path).rglob("GRANULE"))[0]
        band_files = list(granule.rglob("*_B04.jp2"))
        
        if not band_files:
            return FilterResult(passed=False, reason="B04 not found")
        
        with rasterio.open(band_files[0]) as src:
            # Read at reduced resolution for speed (1/4 scale)
            data = src.read(1, out_shape=(src.height // 4, src.width // 4))
            
            # Scale to 0-255 for OpenCV
            scaled = ((data - data.min()) / (data.max() - data.min() + 1e-8) * 255).astype(np.uint8)
            
            # Laplacian variance (sharpness metric)
            laplacian = cv2.Laplacian(scaled, cv2.CV_64F)
            variance = laplacian.var()
            
            passed = variance >= self.min_variance
            
            return FilterResult(
                passed=passed,
                reason=None if passed else f"Scene too blurry (variance: {variance:.1f})",
                metrics={
                    "laplacian_variance": float(variance),
                    "threshold": self.min_variance,
                    "resolution_checked": f"{src.height // 4}x{src.width // 4}"
                }
            )