# src/filters/haze_filter.py — CORRECTED
import numpy as np
import rasterio
from pathlib import Path
from .base import BaseFilter, FilterResult

class HazeFilter(BaseFilter):
    """
    Detects haze using Haze Optimized Transformation (HOT).
    HOT = B02 - 0.5 * B04 - 0.08 (empirical offset)
    Works better for mixed land-water scenes than simple B02/B04 ratio.
    """
    
    def __init__(self, max_hot_score: float = 0.15, max_haze_pixels_ratio: float = 0.3):
        super().__init__()
        self.max_hot_score = max_hot_score
        self.max_haze_pixels_ratio = max_haze_pixels_ratio
    
    def apply(self, scene_path: str) -> FilterResult:
        granule = list(Path(scene_path).rglob("GRANULE"))[0]
        b02 = list(granule.rglob("*_B02.jp2"))
        b04 = list(granule.rglob("*_B04.jp2"))
        
        if not b02 or not b04:
            return FilterResult(passed=False, reason="B02 or B04 not found")
        
        with rasterio.open(b02[0]) as src_b, rasterio.open(b04[0]) as src_r:
            # Read at reduced resolution
            blue = src_b.read(1, out_shape=(src_b.height//4, src_b.width//4)).astype(np.float32) / 10000
            red = src_r.read(1, out_shape=(src_r.height//4, src_r.width//4)).astype(np.float32) / 10000
            
            # HOT (Haze Optimized Transformation)
            # Higher HOT = more haze
            hot = blue - 0.5 * red - 0.08
            
            # Only consider positive HOT values (haze increases blue)
            hot_positive = hot[hot > 0]
            
            if len(hot_positive) == 0:
                # No haze detected at all
                return FilterResult(
                    passed=True,
                    reason=None,
                    metrics={
                        "hot_mean": float(np.mean(hot)),
                        "hot_max": float(np.max(hot)),
                        "haze_pixels_ratio": 0.0,
                        "threshold": self.max_hot_score
                    }
                )
            
            # Haze pixels: HOT above threshold
            haze_pixels = np.sum(hot > self.max_hot_score)
            haze_pixels_ratio = haze_pixels / hot.size
            
            passed = haze_pixels_ratio < self.max_haze_pixels_ratio
            
            return FilterResult(
                passed=passed,
                reason=None if passed else f"Haze pixels: {haze_pixels_ratio:.1%} (threshold: {self.max_haze_pixels_ratio:.0%})",
                metrics={
                    "hot_mean": float(np.mean(hot_positive)),
                    "hot_max": float(np.max(hot)),
                    "haze_pixels_ratio": float(haze_pixels_ratio),
                    "haze_pixels_threshold": self.max_haze_pixels_ratio,
                    "hot_threshold": self.max_hot_score
                }
            )