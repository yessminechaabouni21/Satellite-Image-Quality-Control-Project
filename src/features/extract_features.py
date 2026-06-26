# src/features/extractor.py
"""
Feature extraction for ML models.
Extracts the same features used during training from a .SAFE scene.
"""

import numpy as np
from pathlib import Path
import rasterio

from src.filters.metadata_filter import MetadataFilter
from src.filters.noData_filter import NoDataFilter
from src.filters.toascaling_filter import TOAScalingFilter
from src.filters.blur_filter import BlurFilter
from src.filters.noise_filter import NoiseFilter
from src.filters.stripe_filter import StripeFilter
from src.filters.missing_bands_filter import MissingBandsFilter


def extract_scene_features(scene_path: str) -> dict:
    """
    Extract features from a single scene for ML scoring.
    
    This runs the same filters as Stage 1 but collects ALL metrics
    (not just pass/fail) to build the feature vector for Stage 2.
    """
    scene_path = Path(scene_path)
    features = {}
    
    # Initialize filters
    filters = [
        MetadataFilter(max_cloud=60.0),
        MissingBandsFilter(),
        TOAScalingFilter(),
        NoDataFilter(max_unexpected_nodata_ratio=0.05, min_unique_values=100),
        BlurFilter(min_variance=15.0),
        StripeFilter(max_periodic_power_ratio=0.3),
        NoiseFilter(max_noise_std_ratio=0.15),
    ]
    
    # Run each filter and collect metrics
    for f in filters:
        try:
            res = f.run(scene_path)
            if hasattr(res, 'metrics') and res.metrics:
                for k, v in res.metrics.items():
                    if hasattr(v, 'item'):
                        v = float(v.item())
                    elif isinstance(v, np.ndarray):
                        v = float(v)
                    features[f"{f.name}__{k}"] = v
            
            features[f"{f.name}__passed"] = 1.0 if res.passed else 0.0
            
        except Exception:
            features[f"{f.name}__passed"] = 0.0
            features[f"{f.name}__error"] = 1.0
    
    # Add derived features
    try:
        features.update(_extract_derived_features(scene_path))
    except Exception:
        pass
    
    return features


def _extract_derived_features(scene_path: Path) -> dict:
    """Extract derived statistical features from B04 band."""
    derived = {}
    
    # Find B04 file
    b04_files = list(scene_path.rglob("*B04*.jp2"))
    if not b04_files:
        b04_files = list(scene_path.rglob("*B04*.tif"))
    
    if b04_files:
        with rasterio.open(b04_files[0]) as src:
            data = src.read(1).astype(np.float32)
            derived['dn_p05'] = float(np.percentile(data, 5))
            derived['dn_p25'] = float(np.percentile(data, 25))
            derived['dn_p75'] = float(np.percentile(data, 75))
            derived['dn_p95'] = float(np.percentile(data, 95))
            derived['dn_range'] = float(np.max(data) - np.min(data))
            derived['dn_iqr'] = float(np.percentile(data, 75) - np.percentile(data, 25))
    
    # Inter-band ratio
    b08_files = list(scene_path.rglob("*B08*.jp2"))
    if b04_files and b08_files:
        try:
            with rasterio.open(b04_files[0]) as red, rasterio.open(b08_files[0]) as nir:
                red_data = red.read(1).astype(np.float32)
                nir_data = nir.read(1).astype(np.float32)
                mask = red_data > 0
                ratio = np.mean(nir_data[mask] / red_data[mask])
                derived['inter_band_ratio_nir_red'] = float(ratio)
        except Exception:
            derived['inter_band_ratio_nir_red'] = 0.0
    
    return derived