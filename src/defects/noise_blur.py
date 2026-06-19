# src/defects/noise_blur.py
import cv2
import numpy as np
import rasterio
from pathlib import Path
import shutil


def _copy_scene(scene_path, output_dir, prefix):
    """Helper: copy scene and return new scene path."""
    new_scene = Path(output_dir) / f"{prefix}_{Path(scene_path).name}"
    if new_scene.exists():
        shutil.rmtree(new_scene)
    shutil.copytree(scene_path, new_scene)
    return new_scene


def inject_noise(scene_path, output_dir, band="B04", sigma=25):
    """Add Gaussian noise to simulate sensor degradation."""
    new_scene = _copy_scene(scene_path, output_dir, f"NOISE_{sigma}")
    new_granule = list(new_scene.rglob("GRANULE"))[0]
    new_band = list(new_granule.rglob(f"*{band}.jp2"))[0]
    
    # Read, modify, overwrite
    with rasterio.open(new_band) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile
    
    noise = np.random.normal(0, sigma, data.shape)
    noisy = np.clip(data + noise, 0, 65535).astype(np.uint16)
    
    # Overwrite — update profile to ensure correct dtype
    profile.update(dtype='uint16', count=1)
    with rasterio.open(new_band, 'w', **profile) as dst:
        dst.write(noisy, 1)
    
    return str(new_scene)


def inject_blur(scene_path, output_dir, band="B04", kernel=15):
    """Add Gaussian blur to simulate motion or defocus."""
    new_scene = _copy_scene(scene_path, output_dir, f"BLUR_{kernel}")
    new_granule = list(new_scene.rglob("GRANULE"))[0]
    new_band = list(new_granule.rglob(f"*{band}.jp2"))[0]
    
    with rasterio.open(new_band) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile
    
    # Blur at native resolution (float32, NOT 8-bit)
    blurred = cv2.GaussianBlur(data, (kernel, kernel), 0)
    blurred = np.clip(blurred, 0, 65535).astype(np.uint16)
    
    profile.update(dtype='uint16', count=1)
    with rasterio.open(new_band, 'w', **profile) as dst:
        dst.write(blurred, 1)
    
    return str(new_scene)


def inject_stripes(scene_path, output_dir, band="B04", intensity=1000):
    """Add periodic stripes to simulate readout interference."""
    new_scene = _copy_scene(scene_path, output_dir, f"STRIPE_{intensity}")
    new_granule = list(new_scene.rglob("GRANULE"))[0]
    new_band = list(new_granule.rglob(f"*{band}.jp2"))[0]
    
    with rasterio.open(new_band) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile
    
    h, w = data.shape
    for i in range(0, h, 10):
        data[i:i+2, :] += intensity
    
    data = np.clip(data, 0, 65535).astype(np.uint16)
    
    profile.update(dtype='uint16', count=1)
    with rasterio.open(new_band, 'w', **profile) as dst:
        dst.write(data, 1)
    
    return str(new_scene)