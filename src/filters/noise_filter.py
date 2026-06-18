# src/filters/noise_filter.py
import numpy as np
import rasterio
from pathlib import Path
from scipy import ndimage
from .base import BaseFilter, FilterResult


class NoiseFilter(BaseFilter):
    """
    Fast generalized noise detection using vectorized operations.
    """
    
    def __init__(self, 
                 max_noise_uniformity: float = 0.7,
                 max_dead_pixel_ratio: float = 0.001):
        super().__init__()
        self.max_noise_uniformity = max_noise_uniformity
        self.max_dead_pixel_ratio = max_dead_pixel_ratio
    
    def apply(self, scene_path: str) -> FilterResult:
        granule = list(Path(scene_path).rglob("GRANULE"))[0]
        band_files = list(granule.rglob("*_B04.jp2"))
        
        if not band_files:
            return FilterResult(passed=False, reason="B04 not found")
        
        with rasterio.open(band_files[0]) as src:
            # Read small patch for speed (center only)
            h, w = src.height, src.width
            window = rasterio.windows.Window(w//4, h//4, w//2, h//2)
            data = src.read(1, window=window).astype(np.float32)
            
            # Remove nodata
            if src.nodata is not None:
                data = data[data != src.nodata]
            
            if data.size < 1000:
                return FilterResult(passed=False, reason="Too few pixels")
            
            # FAST: Vectorized local std using uniform filter
            # E[X^2] - E[X]^2 = Var(X)
            local_mean = ndimage.uniform_filter(data, size=3)
            local_mean_sq = ndimage.uniform_filter(data**2, size=3)
            local_var = local_mean_sq - local_mean**2
            local_std = np.sqrt(np.clip(local_var, 0, None))
            
            # Coefficient of variation of local std
            valid_std = local_std[local_std > 0]
            if len(valid_std) < 100:
                cv_of_std = 0
            else:
                cv_of_std = np.std(valid_std) / (np.mean(valid_std) + 1e-8)
            
            noise_uniformity = 1.0 / (1.0 + cv_of_std)
            
            # Dead pixels
            dead_ratio = (np.sum(data == data.min()) + np.sum(data == data.max())) / data.size
            
            # Decision
            issues = []
            if noise_uniformity > self.max_noise_uniformity:
                issues.append(f"Noise uniformity: {noise_uniformity:.3f}")
            if dead_ratio > self.max_dead_pixel_ratio:
                issues.append(f"Dead pixels: {dead_ratio:.3%}")
            
            passed = len(issues) == 0
            
            return FilterResult(
                passed=passed,
                reason="; ".join(issues) if issues else None,
                metrics={
                    "noise_uniformity": float(noise_uniformity),
                    "threshold": self.max_noise_uniformity,
                    "dead_pixel_ratio": float(dead_ratio),
                    "cv_of_std": float(cv_of_std)
                }
            )