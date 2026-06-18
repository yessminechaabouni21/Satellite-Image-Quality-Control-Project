# src/filters/missing_bands_filter.py
import numpy as np
import rasterio
from pathlib import Path
from .base import BaseFilter, FilterResult


class MissingBandsFilter(BaseFilter):
    """
    Ensures all required bands are present and valid.
    Excludes QA/mask files (MSK_ prefix).
    """
    
    def __init__(self, required_bands: list = None):
        super().__init__()
        self.required_bands = required_bands or ["B02", "B03", "B04", "B08", "B11"]
    
    def apply(self, scene_path: str) -> FilterResult:
        granule = list(Path(scene_path).rglob("GRANULE"))[0]
        
        missing = []
        found = []
        
        for band in self.required_bands:
            # STRICT patterns — exclude MSK_ QA files
            patterns = [
                f"*{band}.jp2",           # B04.jp2
                f"*{band}_*.jp2",         # B04_10m.jp2
                f"T*{band}*.jp2",         # T32SPF_..._B04.jp2
                f"L*{band}*.jp2"          # L1C_..._B04.jp2
            ]
            
            band_files = []
            for pattern in patterns:
                candidates = list(granule.rglob(pattern))
                # Filter out MSK_ (QA masks) and other non-data files
                real_bands = [c for c in candidates if not c.name.startswith("MSK_")]
                if real_bands:
                    band_files = real_bands
                    break
            
            if not band_files:
                missing.append(band)
                print(f"  [MissingBandsFilter] MISSING: {band}")
            else:
                found.append(band)
                print(f"  [MissingBandsFilter] FOUND: {band} at {band_files[0].name}")
        
        passed = len(missing) == 0
        
        return FilterResult(
            passed=passed,
            reason=None if passed else f"Missing bands: {', '.join(missing)}",
            metrics={
                "required_bands": self.required_bands,
                "found_bands": found,
                "missing_bands": missing,
                "found_count": len(found),
                "required_count": len(self.required_bands)
            }
        )