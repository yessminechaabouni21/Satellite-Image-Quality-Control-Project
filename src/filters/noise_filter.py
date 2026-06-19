import numpy as np
import rasterio
from pathlib import Path
from scipy import ndimage
from .base import BaseFilter, FilterResult


class NoiseFilter(BaseFilter):
    """
    Detects excessive sensor noise and dead/stuck pixels.
    """
    
    def __init__(self, 
                 max_noise_std_ratio: float = 0.03,
                 max_dead_pixel_ratio: float = 0.001):
        super().__init__()
        self.max_noise_std_ratio = max_noise_std_ratio
        self.max_dead_pixel_ratio = max_dead_pixel_ratio
    
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
                # ALSO exclude blackfill (0) which is common at edges
                patch = patch[patch > 0]
                if patch.size > 500:
                    patches.append(patch)
            
            if not patches:
                return FilterResult(passed=False, reason="No valid patches")
            
            all_pixels = np.concatenate(patches)
            
            if all_pixels.size < 1000:
                return FilterResult(passed=False, reason="Too few pixels")
            
            issues = []
            metrics = {}
            
            # Noise estimation
            noise_estimates = []
            for patch in patches:
                if patch.size < 100:
                    continue
                local_mean = ndimage.uniform_filter(patch, size=3)
                local_mean_sq = ndimage.uniform_filter(patch**2, size=3)
                local_var = np.clip(local_mean_sq - local_mean**2, 0, None)
                valid_var = local_var[local_var > 0]
                if valid_var.size > 0:
                    noise_estimates.append(np.percentile(valid_var, 5))
                else:
                    noise_estimates.append(0.0)

            noise_var = np.min(noise_estimates) if noise_estimates else 0
            noise_std = np.sqrt(noise_var)
            signal = np.median(all_pixels)
            noise_ratio = noise_std / (signal + 1e-8)
            
            if noise_ratio > self.max_noise_std_ratio:
                issues.append(f"Noise ratio: {noise_ratio:.4f}")
            metrics["noise_std_ratio"] = float(noise_ratio)
            
            # Dead pixels: ONLY count if value repeats suspiciously often
            # AND is not the minimum (which could be valid dark pixels)
            unique, counts = np.unique(all_pixels, return_counts=True)
            max_count = np.max(counts)
            dead_ratio = max_count / len(all_pixels)
            dead_value = unique[np.argmax(counts)]
            
            # Only flag if >0.1% pixels have EXACTLY same value (stuck sensor)
            if dead_ratio > self.max_dead_pixel_ratio and dead_value > 0:
                issues.append(f"Stuck pixels at {dead_value}: {dead_ratio:.3%}")
            metrics["dead_pixel_ratio"] = float(dead_ratio)
            metrics["most_common_value"] = float(dead_value)
            
            passed = len(issues) == 0
            
            return FilterResult(
                passed=passed,
                reason="; ".join(issues) if issues else None,
                metrics=metrics
            )