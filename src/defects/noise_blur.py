# src/defects/noise_blur.py
import cv2
import numpy as np
import rasterio
from pathlib import Path
import shutil


def inject_noise(scene_path, output_dir, band="B04", sigma=25):
    """Add Gaussian noise to simulate sensor degradation."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    
    new_scene = Path(output_dir) / f"NOISE_{sigma}_{Path(scene_path).name}"
    if new_scene.exists():
        shutil.rmtree(new_scene)
    shutil.copytree(scene_path, new_scene)
    
    new_granule = list(new_scene.rglob("GRANULE"))[0]
    new_band = list(new_granule.rglob(f"*{band}.jp2"))[0]
    
    with rasterio.open(new_band, 'r+') as src:
        data = src.read(1).astype(np.float32)
        noise = np.random.normal(0, sigma, data.shape)
        noisy = np.clip(data + noise, 0, 65535).astype(np.uint16)
        src.write(noisy, 1)
    
    return str(new_scene)


def inject_blur(scene_path, output_dir, band="B04", kernel=15):
    """Add Gaussian blur to simulate motion or defocus."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    
    new_scene = Path(output_dir) / f"BLUR_{kernel}_{Path(scene_path).name}"
    if new_scene.exists():
        shutil.rmtree(new_scene)
    shutil.copytree(scene_path, new_scene)
    
    new_granule = list(new_scene.rglob("GRANULE"))[0]
    new_band = list(new_granule.rglob(f"*{band}.jp2"))[0]
    
    with rasterio.open(new_band, 'r+') as src:
        data = src.read(1)
        # Scale to 0-255 for OpenCV, blur, scale back
        scaled = ((data - data.min()) / (data.max() - data.min() + 1e-8) * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(scaled, (kernel, kernel), 0)
        back = (blurred / 255 * (data.max() - data.min()) + data.min()).astype(np.uint16)
        src.write(back, 1)
    
    return str(new_scene)


def inject_stripes(scene_path, output_dir, band="B04", intensity=1000):
    """Add periodic stripes to simulate readout interference."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    
    new_scene = Path(output_dir) / f"STRIPE_{intensity}_{Path(scene_path).name}"
    if new_scene.exists():
        shutil.rmtree(new_scene)
    shutil.copytree(scene_path, new_scene)
    
    new_granule = list(new_scene.rglob("GRANULE"))[0]
    new_band = list(new_granule.rglob(f"*{band}.jp2"))[0]
    
    with rasterio.open(new_band, 'r+') as src:
        data = src.read(1).astype(np.float32)
        h, w = data.shape
        
        # Add horizontal stripes every 10 rows
        for i in range(0, h, 10):
            data[i:i+2, :] += intensity
        
        data = np.clip(data, 0, 65535).astype(np.uint16)
        src.write(data, 1)
    
    return str(new_scene)