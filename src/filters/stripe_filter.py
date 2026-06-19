import numpy as np
import rasterio
from pathlib import Path
from .base import BaseFilter, FilterResult


class StripeFilter(BaseFilter):
    """
    Detects periodic row/column stripes by analyzing row/column mean profiles.
    Stripes create periodic spikes in the mean profile that natural variation doesn't.
    """
    
    def __init__(self, 
                 max_periodic_power_ratio: float = 0.3,
                 min_stripe_period: int = 5,
                 max_stripe_period: int = 50):
        super().__init__()
        self.max_periodic_power_ratio = max_periodic_power_ratio
        self.min_stripe_period = min_stripe_period
        self.max_stripe_period = max_stripe_period
    
    def apply(self, scene_path: str) -> FilterResult:
        granule = list(Path(scene_path).rglob("GRANULE"))[0]
        band_files = list(granule.rglob("*_B04.jp2"))
        
        if not band_files:
            return FilterResult(passed=False, reason="B04 not found")
        
        with rasterio.open(band_files[0]) as src:
            # Read center strip (full width, manageable height)
            h, w = src.height, src.width
            window = rasterio.windows.Window(0, h//4, min(w, 2000), h//2)
            data = src.read(1, window=window).astype(np.float32)
            
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            
            # Compute row means (horizontal stripes affect row means)
            row_means = np.nanmean(data, axis=1)
            
            # Detrend (remove slow variations)
            detrended = row_means - np.convolve(row_means, np.ones(50)/50, mode='same')
            
            # FFT to find periodic components
            fft = np.fft.rfft(detrended)
            power = np.abs(fft)**2
            
            # Exclude DC
            power[0] = 0
            
            total_power = np.sum(power)
            if total_power < 1e-10:
                return FilterResult(passed=True, reason=None, metrics={"periodic_ratio": 0.0})
            
            # Find dominant periodic component
            peak_power = np.max(power[1:])  # Exclude DC
            peak_idx = np.argmax(power[1:]) + 1
            
            # Convert to period in pixels
            freqs = np.fft.rfftfreq(len(detrended))
            peak_period = 1.0 / freqs[peak_idx] if freqs[peak_idx] > 0 else float('inf')
            
            periodic_ratio = peak_power / total_power
            
            # Check if peak is in expected stripe period range
            is_stripe = (
                periodic_ratio > self.max_periodic_power_ratio and
                self.min_stripe_period <= peak_period <= self.max_stripe_period
            )
            
            passed = not is_stripe
            
            return FilterResult(
                passed=passed,
                reason=None if passed else (
                    f"Stripes detected: period={peak_period:.1f}px, "
                    f"power_ratio={periodic_ratio:.3f}"
                ),
                metrics={
                    "periodic_power_ratio": float(periodic_ratio),
                    "peak_period_pixels": float(peak_period),
                    "threshold": self.max_periodic_power_ratio,
                    "detrended_std": float(np.std(detrended))
                }
            )