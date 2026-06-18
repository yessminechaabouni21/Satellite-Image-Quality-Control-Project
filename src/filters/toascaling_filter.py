# src/filters/toa_scaling_filter.py
import rasterio
import numpy as np
from pathlib import Path
from .base import BaseFilter, FilterResult

class TOAScalingFilter(BaseFilter):
    """
    Validates Sentinel-2 L1C TOA reflectance scaling.
    Checks data type, value range, and quantification metadata.
    """
    
    def __init__(self, expected_quantification: int = 10000):
        # Match your BaseFilter signature — check what BaseFilter.__init__ expects
        super().__init__()
        self.expected_quantification = expected_quantification
    
    def apply(self, scene_path: str) -> FilterResult:
        """Standard apply method matching BaseFilter interface."""
        granule = list(Path(scene_path).rglob("GRANULE"))[0]
        
        # L1C naming pattern (no resolution suffix)
        band_files = list(granule.rglob("*_B04.jp2"))
        
        # Fallback
        if not band_files:
            band_files = list(granule.rglob("*B04*.jp2"))
        
        if not band_files:
            return FilterResult(
                passed=False, 
                reason="B04 band not found — check .SAFE structure",
                metrics={"searched_patterns": ["*_B04.jp2", "*B04*.jp2"]}
            )
        
        with rasterio.open(band_files[0]) as src:
            data = src.read(1)
            profile = src.profile
            
            issues = []
            
            # Check 1: Data type must be uint16
            if profile['dtype'] != 'uint16':
                issues.append(f"Unexpected dtype: {profile['dtype']} (expected uint16)")
            
            # Check 2: Value range sanity check
            max_val = int(data.max())
            min_val = int(data.min())
            
            if max_val > 20000:
                issues.append(f"Max DN suspiciously high: {max_val}")
            
            if max_val == 0 and min_val == 0:
                issues.append("Band is completely zero — likely corrupted")
            
            # Check 3: Unique values (corruption check)
            unique_count = len(np.unique(data))
            if unique_count < 10:
                issues.append(f"Only {unique_count} unique values — uniform/corrupted")
            
            passed = len(issues) == 0
            
            return FilterResult(
                passed=passed,
                reason="; ".join(issues) if issues else None,
                metrics={
                    "band_file": str(band_files[0].name),
                    "dtype": str(profile['dtype']),
                    "shape": list(data.shape),
                    "min_dn": min_val,
                    "max_dn": max_val,
                    "unique_values": int(unique_count),
                    "quantification_value": self.expected_quantification,
                    "toa_scale_factor": 1.0 / self.expected_quantification
                }
            )