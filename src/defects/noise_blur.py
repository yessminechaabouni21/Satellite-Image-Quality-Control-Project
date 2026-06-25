# src/defects/noise_blur.py
import cv2
import numpy as np
import rasterio
from pathlib import Path
import shutil


def _fast_copy_scene(base_scene, output_dir, prefix):
    """Copy only GRANULE folder, convert all JP2 bands to GeoTIFF for fast write."""
    base = Path(base_scene)
    new_scene = Path(output_dir) / f"{prefix}_{base.name}"
    
    if new_scene.exists():
        shutil.rmtree(new_scene)
    
    new_scene.mkdir(parents=True)
    granule_src = list(base.rglob("GRANULE"))[0]
    granule_dst = new_scene / "GRANULE"
    granule_dst.mkdir()
    
    # Convert ALL bands to GeoTIFF (fast read/write vs JP2)
    for band_file in granule_src.rglob("*.jp2"):
        with rasterio.open(band_file) as src:
            profile = src.profile.copy()
            profile.update(driver='GTiff', compress='lzw', dtype='uint16', count=1)
            data = src.read(1)
            
            dst_path = granule_dst / band_file.with_suffix('.tif').name
            with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(data, 1)
    
    # Copy manifest files
    for f in ["manifest.safe", "MTD_MSIL1C.xml"]:
        src = base / f
        if src.exists():
            shutil.copy2(src, new_scene / f)
    
    return new_scene


def inject_noise_fast(scene_path, output_dir, band="B04", sigma=25):
    """Add Gaussian noise — fast version with TIF bands."""
    new_scene = _fast_copy_scene(scene_path, output_dir, f"NOISE_{sigma}")
    new_granule = list(new_scene.rglob("GRANULE"))[0]
    
    # Find TIF band (converted from JP2)
    new_band = list(new_granule.rglob(f"*{band}.tif"))[0]
    
    with rasterio.open(new_band) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    
    noise = np.random.normal(0, sigma, data.shape)
    noisy = np.clip(data + noise, 0, 65535).astype(np.uint16)
    
    with rasterio.open(new_band, 'w', **profile) as dst:
        dst.write(noisy, 1)
    
    return str(new_scene)


def inject_blur_fast(scene_path, output_dir, band="B04", kernel=15):
    """Add Gaussian blur — fast version with TIF bands."""
    new_scene = _fast_copy_scene(scene_path, output_dir, f"BLUR_{kernel}")
    new_granule = list(new_scene.rglob("GRANULE"))[0]
    new_band = list(new_granule.rglob(f"*{band}.tif"))[0]
    
    with rasterio.open(new_band) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    
    blurred = cv2.GaussianBlur(data, (kernel, kernel), 0)
    blurred = np.clip(blurred, 0, 65535).astype(np.uint16)
    
    with rasterio.open(new_band, 'w', **profile) as dst:
        dst.write(blurred, 1)
    
    return str(new_scene)


def inject_stripes_fast(scene_path, output_dir, band="B04", intensity=1000):
    """Add periodic stripes — fast version with TIF bands."""
    new_scene = _fast_copy_scene(scene_path, output_dir, f"STRIPE_{intensity}")
    new_granule = list(new_scene.rglob("GRANULE"))[0]
    new_band = list(new_granule.rglob(f"*{band}.tif"))[0]
    
    with rasterio.open(new_band) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    
    h, w = data.shape
    for i in range(0, h, 10):
        data[i:i+2, :] += intensity
    
    data = np.clip(data, 0, 65535).astype(np.uint16)
    
    with rasterio.open(new_band, 'w', **profile) as dst:
        dst.write(data, 1)
    
    return str(new_scene)


# Keep old functions for backward compatibility
inject_noise = inject_noise_fast
inject_blur = inject_blur_fast
inject_stripes = inject_stripes_fast