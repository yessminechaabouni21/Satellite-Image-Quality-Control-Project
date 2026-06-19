# src/filters/noise_filter.py — SIMPLIFIED, no dead pixel false positives

import numpy as np
import rasterio
from pathlib import Path
from scipy import ndimage
from .base import BaseFilter, FilterResult


class NoiseFilter(BaseFilter):
    """
    Detects excessive sensor noise using SNR from smooth patches.
    Dead pixel detection removed — causes false positives on natural scenes.
    """
    
    def __init__(self, max_noise_std_ratio: float = 0.03):
        super().__init__()
        self.max_noise_std_ratio = max_noise_std_ratio
    
    def apply(self, scene_path: str) -> FilterResult:
        granule = list(Path(scene_path).rglob("GRANULE"))[0]
        band_files = list(granule.rglob("*_B04.jp2"))
        
        if not band_files:
            return FilterResult(passed=False, reason="B04 not found")
        
        with rasterio.open(band_files[0]) as src:
            h, w = src.height, src.width
            patches = []
            positions = [
                (h//4, w//4), (h//4, 3*w//4),
                (3*h//4, w//4), (3*h//4, 3*w//4),
                (h//2, w//2),
            ]
            
            for cy, cx in positions:
                window = rasterio.windows.Window(cx - 50, cy - 50, 100, 100)
                patch = src.read(1, window=window).astype(np.float32)
                if src.nodata is not None:
                    patch = patch[patch != src.nodata]
                patch = patch[patch > 0]  # Exclude blackfill
                if patch.size > 500:
                    patches.append(patch)
            
            if not patches:
                return FilterResult(passed=False, reason="No valid patches")
            
            all_pixels = np.concatenate(patches)
            
            # Noise estimation from smoothest patch
            noise_estimates = []
            for patch in patches:
                if patch.size < 100:
                    continue
                local_mean = ndimage.uniform_filter(patch, size=3)
                local_mean_sq = ndimage.uniform_filter(patch**2, size=3)
                local_var = np.clip(local_mean_sq - local_mean**2, 0, None)
                noise_estimates.append(np.percentile(local_var[local_var > 0], 5))
            
            noise_var = np.min(noise_estimates) if noise_estimates else 0
            noise_std = np.sqrt(noise_var)
            signal = np.median(all_pixels)
            noise_ratio = noise_std / (signal + 1e-8)
            
            passed = noise_ratio <= self.max_noise_std_ratio
            print(f"  NoiseFilter: noise_std_ratio={noise_ratio:.4f}, threshold={self.max_noise_std_ratio}")
            return FilterResult(
                passed=passed,
                reason=None if passed else f"Noise ratio: {noise_ratio:.4f}",
                metrics={
                    "noise_std_ratio": float(noise_ratio),
                    "noise_std": float(noise_std),
                    "signal_median": float(signal),
                    "threshold": self.max_noise_std_ratio
                }
            )