# src/visualization/rgb_toa.py
import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt


def load_rgb_toa(scene_path):
    """Load RGB image and return numpy array."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]

    b02 = list(granule.rglob("*_B02.jp2"))[0]
    b03 = list(granule.rglob("*_B03.jp2"))[0]
    b04 = list(granule.rglob("*_B04.jp2"))[0]

    with rasterio.open(b04) as src_r, \
         rasterio.open(b03) as src_g, \
         rasterio.open(b02) as src_b:

        r = src_r.read(1).astype(np.float32) / 10000.0
        g = src_g.read(1).astype(np.float32) / 10000.0
        b = src_b.read(1).astype(np.float32) / 10000.0

        rgb = np.dstack([r, g, b])

        # Stretch contrast
        valid = rgb[rgb > 0]
        if len(valid) > 0:
            p2, p98 = np.percentile(valid, (2, 98))
            rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)

        return rgb


def create_rgb_toa(scene_path: str, output_path: str = None):
    """Create and display RGB TOA image."""
    rgb = load_rgb_toa(scene_path)

    plt.figure(figsize=(12, 12))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title(f"RGB TOA: {Path(scene_path).name[:40]}")
    plt.show()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(output_path, rgb)
        print(f"Saved: {output_path}")

    return rgb


def create_rgb_with_nodata_mask(scene_path: str, output_path: str = None):
    """RGB with red overlay on nodata regions."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    b04 = list(granule.rglob("*_B04.jp2"))[0]
    
    with rasterio.open(b04) as src:
        data = src.read(1)
        nodata = src.nodata
        
        # Get nodata mask
        if nodata is not None:
            nodata_mask = data == nodata
        else:
            nodata_mask = data == 0
        
        print(f"\n{'='*50}")
        print(f"NODATA ANALYSIS: {Path(scene_path).name[:40]}")
        print(f"{'='*50}")
        print(f"Nodata value: {nodata}")
        print(f"Nodata pixels: {nodata_mask.sum():,} ({nodata_mask.mean():.1%})")
        print(f"Valid pixels: {(~nodata_mask).sum():,} ({(~nodata_mask).mean():.1%})")
        print(f"{'='*50}\n")
    
    # Load RGB
    rgb = load_rgb_toa(scene_path)
    
    # Overlay red on nodata
    rgb[nodata_mask] = [1.0, 0.0, 0.0]  # Red
    
    plt.figure(figsize=(12, 12))
    plt.imshow(rgb)
    plt.title(f"RGB TOA (Red = Nodata)\n{Path(scene_path).name[:40]}")
    plt.axis("off")
    plt.show()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(output_path, rgb)
        print(f"Saved: {output_path}")
    
    return rgb


if __name__ == "__main__":
    # Scene that was rejected by NoDataFilter
    scene = r"data\extracted\S2A_MSIL1C_20260409T101051_N0512_R022_T32SPF_20260409T152519.SAFE"
    output = r"outputs\rgb_toa_nodata.png"

    # Choose which function to run:
    # create_rgb_toa(scene, output)           # Normal RGB
    create_rgb_with_nodata_mask(scene, output)  # RGB with nodata overlay