# src/filters/nodata_filter.py
import numpy as np
import rasterio
from pathlib import Path
from scipy import ndimage
from .base import BaseFilter, FilterResult


class NoDataFilter(BaseFilter):
    """
    Detects corrupted or incomplete scenes.
    Ignores expected nodata borders, flags unexpected nodata in valid region.
    """
    
    def __init__(self, 
                 max_unexpected_nodata_ratio: float = 0.05,  # 5% unexpected
                 min_unique_values: int = 100):
        super().__init__()
        self.max_unexpected_nodata_ratio = max_unexpected_nodata_ratio
        self.min_unique_values = min_unique_values
    
    def apply(self, scene_path: str) -> FilterResult:
        granule = list(Path(scene_path).rglob("GRANULE"))[0]
        
        band_files = list(granule.rglob("*_B04.jp2"))
        if not band_files:
            band_files = list(granule.rglob("*_B08.jp2"))
        
        if not band_files:
            return FilterResult(passed=False, reason="No B04 or B08 found")
        
        with rasterio.open(band_files[0]) as src:
            data = src.read(1)
            total_pixels = data.size
            
            # CHECK 1: All zeros
            if data.max() == 0:
                return FilterResult(
                    passed=False, reason="Completely black",
                    metrics={"corruption": "all_zeros"}
                )
            
            # CHECK 2: Flat
            unique_count = len(np.unique(data))
            if unique_count < self.min_unique_values:
                return FilterResult(
                    passed=False, reason=f"Flat: {unique_count} unique values",
                    metrics={"corruption": "flat", "unique_values": unique_count}
                )
            
            # CHECK 3: Nodata detection
            if src.nodata is not None:
                nodata_mask = data == src.nodata
            else:
                # Implicit nodata: zeros that form large contiguous blocks (edges)
                zero_mask = data == 0
                
                # Label connected regions of zeros
                labeled, num_features = ndimage.label(zero_mask)
                
                # Find largest zero region (expected to be border)
                region_sizes = ndimage.sum(zero_mask, labeled, index=range(1, num_features + 1))
                
                if len(region_sizes) > 0:
                    largest_region_size = region_sizes.max()
                    largest_region_ratio = largest_region_size / total_pixels
                    
                    # If largest zero block is >40% of image, it's likely a border
                    # Remove it from "unexpected" count
                    if largest_region_ratio > 0.4:
                        # Largest region is expected border — remove it
                        largest_label = np.argmax(region_sizes) + 1
                        expected_nodata_mask = (labeled == largest_label)
                        unexpected_nodata_mask = zero_mask & ~expected_nodata_mask
                    else:
                        # No large border — all zeros are unexpected
                        unexpected_nodata_mask = zero_mask
                        expected_nodata_mask = np.zeros_like(zero_mask)
                else:
                    unexpected_nodata_mask = np.zeros_like(zero_mask, dtype=bool)
                    expected_nodata_mask = np.zeros_like(zero_mask, dtype=bool)
                
                nodata_mask = unexpected_nodata_mask
            
            # Unexpected nodata in valid region
            unexpected_nodata_ratio = nodata_mask.sum() / total_pixels
            
            # CHECK 4: Saturated
            max_possible = 65535 if data.dtype == np.uint16 else data.max()
            saturated_ratio = (data == max_possible).sum() / total_pixels
            if saturated_ratio > 0.5:
                return FilterResult(
                    passed=False, reason=f"Mostly saturated: {saturated_ratio:.1%}",
                    metrics={"corruption": "saturated"}
                )
            
            # DECISION
            passed = unexpected_nodata_ratio <= self.max_unexpected_nodata_ratio
            
            return FilterResult(
                passed=passed,
                reason=None if passed else f"Unexpected nodata: {unexpected_nodata_ratio:.1%}",
                metrics={
                    "unexpected_nodata_ratio": float(unexpected_nodata_ratio),
                    "threshold": self.max_unexpected_nodata_ratio,
                    "total_nodata_ratio": float((data == 0).sum() / total_pixels),
                    "unique_values": int(unique_count),
                    "saturated_ratio": float(saturated_ratio)
                }
            )