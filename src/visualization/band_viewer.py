# src/visualization/band_viewer.py
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path


def view_band(scene_path: str, band: str = "B04"):
    """View a single band as grayscale."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    band_file = list(granule.rglob(f"*{band}.jp2"))[0]
    
    with rasterio.open(band_file) as src:
        data = src.read(1)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(data, cmap='gray', vmin=0, vmax=8000)
        plt.title(f"{band} — Min: {data.min()}, Max: {data.max()}")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    import sys
    
    scene_path = sys.argv[1] if len(sys.argv) > 1 else \
        r"data\extracted\S2A_MSIL1C_20260406T100051_N0512_R122_T32SPF_20260406T151531.SAFE"
    
    view_band(scene_path, "B04")
    view_band(scene_path, "B08")