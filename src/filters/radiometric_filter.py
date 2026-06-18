# src/filters/radiometric_filter.py
import numpy as np
import rasterio
from pathlib import Path
from .base import BaseFilter, FilterResult

class RadiometricResolutionFilter(BaseFilter):
    """Verify 12-bit radiometric resolution."""
    
    def __init__(self, expected_bits: int = 12):
        super().__init__()
        self.expected_bits = expected_bits
    
    def apply(self, scene_path: str) -> FilterResult:
        granule = list(Path(scene_path).rglob("GRANULE"))[0]
        b04 = list(granule.rglob("*_B04.jp2"))
        
        with rasterio.open(b04[0]) as src:
            data = src.read(1)
            unique_count = len(np.unique(data))
            bits_used = np.log2(unique_count + 1)
            
            passed = bits_used >= self.expected_bits - 1  # Allow some margin
            
            return FilterResult(
                passed=passed,
                reason=None if passed else f"Only {bits_used:.1f} bits used (expected {self.expected_bits})",
                metrics={
                    "bits_used": float(bits_used),
                    "expected_bits": self.expected_bits,
                    "unique_values": int(unique_count)
                }
            )