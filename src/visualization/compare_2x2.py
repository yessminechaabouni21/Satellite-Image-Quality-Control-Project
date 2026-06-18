# src/visualization/compare_2x2.py
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pathlib import Path
from src.visualization.select_diverse_subset import select_diverse_subset


def load_rgb(scene_path: str):
    """Load RGB array without displaying."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    b02 = list(granule.rglob("*_B02.jp2"))[0]
    b03 = list(granule.rglob("*_B03.jp2"))[0]
    b04 = list(granule.rglob("*_B04.jp2"))[0]
    
    with rasterio.open(b04) as src_r, \
         rasterio.open(b03) as src_g, \
         rasterio.open(b02) as src_b:
        
        r = src_r.read(1).astype(np.float32) / 10000
        g = src_g.read(1).astype(np.float32) / 10000
        b = src_b.read(1).astype(np.float32) / 10000
        
        rgb = np.dstack([r, g, b])
        rgb = np.clip(rgb, 0, 0.3) / 0.3
        return rgb


def create_2x2_figure(scenes: list, output_path: str = "reports/visuals/qc_2x2_comparison.png"):
    """Create 2×2 figure with auto-selected diverse scenes."""
    
    # Select subset
    accepted, rejected = select_diverse_subset(scenes, n_accepted=2, n_rejected=2)
    selected = accepted + rejected
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    positions = [
        ("Top-Left", "Most Diverse Clear"),
        ("Top-Right", "Different Clear Scene"),
        ("Bottom-Left", "Most Cloudy (Rejected)"),
        ("Bottom-Right", "Near-Threshold (Rejected)")
    ]
    
    for idx, (scene_info, (pos, reason)) in enumerate(zip(selected, positions)):
        ax = axes[idx]
        rgb = load_rgb(scene_info["path"])
        ax.imshow(rgb)
        
        status = "✅ ACCEPTED" if scene_info["accepted"] else "❌ REJECTED"
        name = Path(scene_info["path"]).stem[:25]
        cloud = scene_info.get("cloud_cover", "N/A")
        
        title = f"{pos}: {reason}\n{status} | {name}\nCloud: {cloud:.1f}%"
        color = "green" if scene_info["accepted"] else "red"
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        ax.axis("off")
    
    plt.suptitle("EO QC Pipeline: Diverse Scene Selection (2×2)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path("reports/visuals").mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n💾 Saved: {output_path}")
    plt.show()
    
    return selected


if __name__ == "__main__":
    # Your actual scenes data — replace with your real paths
    scenes = [
        {"path": r"data\extracted\S2A_MSIL1C_20260406T100051_N0512_R122_T32SPF_20260406T151531.SAFE", "accepted": True, "cloud_cover": 0.0},
        {"path": r"data\extracted\S2A_MSIL1C_20260409T101051_N0512_R022_T32SPF_20260409T152519.SAFE", "accepted": True, "cloud_cover": 2.5},
        {"path": r"data\extracted\S2A_MSIL1C_20260426T100051_N0512_R122_T32SPF_20260426T151525.SAFE", "accepted": True, "cloud_cover": 0.8},
        {"path": r"data\extracted\S2C_MSIL1C_20260524T100021_N0512_R122_T32SPF_20260524T133919.SAFE", "accepted": True, "cloud_cover": 0.5},
        {"path": r"data\extracted\S2C_MSIL1C_20260613T100031_N0512_R122_T32SPF_20260613T133922.SAFE", "accepted": True, "cloud_cover": 0.03},
        {"path": r"data\extracted\S2A_MSIL1C_20260506T100041_N0512_R122_T32SPF_20260506T151521.SAFE", "accepted": False, "cloud_cover": 99.8, "failed_filter": "MetadataFilter"},
        {"path": r"data\extracted\S2A_MSIL1C_20260526T100041_N0512_R122_T32SPF_20260526T151555.SAFE", "accepted": False, "cloud_cover": 49.0, "failed_filter": "MetadataFilter"},
        {"path": r"data\extracted\S2B_MSIL1C_20260412T101019_N0512_R022_T32SPF_20260412T134431.SAFE", "accepted": False, "cloud_cover": 99.3, "failed_filter": "MetadataFilter"},
        {"path": r"data\extracted\S2B_MSIL1C_20260509T100019_N0512_R122_T32SPF_20260509T133418.SAFE", "accepted": False, "cloud_cover": 98.5, "failed_filter": "MetadataFilter"},
        {"path": r"data\extracted\S2B_MSIL1C_20260608T100029_N0512_R122_T32SPF_20260608T134034.SAFE", "accepted": False, "cloud_cover": 92.0, "failed_filter": "MetadataFilter"},
    ]
    
    selected = create_2x2_figure(scenes)
    print(f"\nSelected {len(selected)} scenes for visualization")