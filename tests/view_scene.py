# view_scene.py
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

scene = r"data\extracted\S2A_MSIL1C_20260406T100051_N0512_R122_T32SPF_20260406T151531.SAFE"
granule = list(Path(scene).rglob("GRANULE"))[0]
b04 = list(granule.rglob("*_B04.jp2"))[0]

with rasterio.open(b04) as src:
    data = src.read(1)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap='gray', vmin=0, vmax=8000)
    plt.title("B04 (Red band) — S2A_20260406")
    plt.colorbar(label="DN")
    plt.show()