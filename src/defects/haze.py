# src/defects/cloud_haze.py
import numpy as np
import rasterio
from pathlib import Path
import shutil


def inject_haze(scene_path, output_dir, strength=0.3):
    """Add atmospheric haze (increases B02 disproportionately)."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    
    new_scene = Path(output_dir) / f"HAZE_{strength}_{Path(scene_path).name}"
    if new_scene.exists():
        shutil.rmtree(new_scene)
    shutil.copytree(scene_path, new_scene)
    
    new_granule = list(new_scene.rglob("GRANULE"))[0]
    b02_file = list(new_granule.rglob("*_B02.jp2"))
    
    if b02_file:
        with rasterio.open(b02_file[0], 'r+') as src:
            data = src.read(1).astype(np.float32)
            # Haze increases blue channel
            haze = data * (1 + strength)
            src.write(np.clip(haze, 0, 65535).astype(np.uint16), 1)
    
    return str(new_scene)