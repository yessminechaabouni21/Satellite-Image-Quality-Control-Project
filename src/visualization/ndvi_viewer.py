# src/visualization/ndvi_viewer.py
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path


def view_ndvi(scene_path: str):
    """Compute and visualize NDVI."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    b04 = list(granule.rglob("*_B04.jp2"))[0]
    b08 = list(granule.rglob("*_B08.jp2"))[0]
    
    with rasterio.open(b04) as src_r, rasterio.open(b08) as src_nir:
        red = src_r.read(1).astype(np.float32) / 10000
        nir = src_nir.read(1).astype(np.float32) / 10000
        
        ndvi = (nir - red) / (nir + red + 1e-8)
        
        # Compute statistics
        ndvi_min = float(np.nanmin(ndvi))
        ndvi_max = float(np.nanmax(ndvi))
        ndvi_mean = float(np.nanmean(ndvi))
        ndvi_std = float(np.nanstd(ndvi))
        
        print(f"\n{'='*50}")
        print(f"NDVI Statistics for: {Path(scene_path).name}")
        print(f"{'='*50}")
        print(f"  Min:   {ndvi_min:.4f}")
        print(f"  Max:   {ndvi_max:.4f}")
        print(f"  Mean:  {ndvi_mean:.4f}")
        print(f"  Std:   {ndvi_std:.4f}")
        print(f"{'='*50}\n")
        
        plt.figure(figsize=(12, 12))
        plt.imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        plt.title(f"NDVI (TOA)\nMin: {ndvi_min:.3f} | Max: {ndvi_max:.3f} | Mean: {ndvi_mean:.3f}")
        plt.colorbar(label="NDVI")
        plt.show()
        
        return {
            "ndvi_min": ndvi_min,
            "ndvi_max": ndvi_max,
            "ndvi_mean": ndvi_mean,
            "ndvi_std": ndvi_std
        }


if __name__ == "__main__":
    import sys
    
    scene_path = sys.argv[1] if len(sys.argv) > 1 else \
        r"data\extracted\S2A_MSIL1C_20260406T100051_N0512_R122_T32SPF_20260406T151531.SAFE"
    
    view_ndvi(scene_path)