# src/defects/corruption.py
import numpy as np
import rasterio
from pathlib import Path
import shutil


def inject_zero_band(scene_path, output_dir, band="B04", percentage=0.5):
    """Zero out percentage of a band to simulate corruption."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    band_file = list(granule.rglob(f"*{band}.jp2"))[0]
    
    # Copy scene to output
    new_scene = Path(output_dir) / f"ZERO_{percentage}_{Path(scene_path).name}"
    if new_scene.exists():
        shutil.rmtree(new_scene)
    shutil.copytree(scene_path, new_scene)
    
    # Find new band file
    new_granule = list(new_scene.rglob("GRANULE"))[0]
    new_band = list(new_granule.rglob(f"*{band}.jp2"))[0]
    
    with rasterio.open(new_band, 'r+') as src:
        data = src.read(1)
        # Zero out random block
        h, w = data.shape
        start_h = np.random.randint(0, max(1, h//2))
        start_w = np.random.randint(0, max(1, w//2))
        end_h = min(start_h + int(h * percentage), h)
        end_w = min(start_w + int(w * percentage), w)
        data[start_h:end_h, start_w:end_w] = 0
        src.write(data, 1)
    
    return str(new_scene)


def inject_flat_band(scene_path, output_dir, band="B04", value=5000):
    """Make entire band flat (single value)."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    
    new_scene = Path(output_dir) / f"FLAT_{value}_{Path(scene_path).name}"
    if new_scene.exists():
        shutil.rmtree(new_scene)
    shutil.copytree(scene_path, new_scene)
    
    new_granule = list(new_scene.rglob("GRANULE"))[0]
    new_band = list(new_granule.rglob(f"*{band}.jp2"))[0]
    
    with rasterio.open(new_band, 'r+') as src:
        data = src.read(1)
        data[:] = value  # ALL pixels = same value
        src.write(data, 1)
    
    return str(new_scene)


def inject_swapped_bands(scene_path, output_dir):
    """Swap B04 and B08 to simulate processing error."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    
    new_scene = Path(output_dir) / f"SWAPPED_{Path(scene_path).name}"
    if new_scene.exists():
        shutil.rmtree(new_scene)
    shutil.copytree(scene_path, new_scene)
    
    new_granule = list(new_scene.rglob("GRANULE"))[0]
    b04_file = list(new_granule.rglob("*_B04.jp2"))[0]
    b08_file = list(new_granule.rglob("*_B08.jp2"))[0]
    
    # Swap by renaming
    temp_b04 = b04_file.with_suffix('.tmp')
    b04_file.rename(temp_b04)
    b08_file.rename(b04_file)
    temp_b04.rename(b08_file)
    
    return str(new_scene)


def inject_missing_band(scene_path, output_dir, band="B11"):
    """Delete a band file."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    
    new_scene = Path(output_dir) / f"MISSING_{band}_{Path(scene_path).name}"
    if new_scene.exists():
        shutil.rmtree(new_scene)
    shutil.copytree(scene_path, new_scene)
    
    new_granule = list(new_scene.rglob("GRANULE"))[0]
    band_file = list(new_granule.rglob(f"*{band}.jp2"))[0]
    
    band_file.unlink()  # Delete the file
    
    return str(new_scene)