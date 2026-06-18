# src/visualization/compare_all_scenes.py
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import rasterio
import pandas as pd


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


def load_scenes_from_report(report_path: str = "reports/pipeline_report.csv"):
    """Load scene data from pipeline report."""
    df = pd.read_csv(report_path)
    
    scenes = []
    for _, row in df.iterrows():
        scene_name = row["scene"]
        scene_path = f"data/extracted/{scene_name}"
        
        scenes.append({
            "path": scene_path,
            "accepted": bool(row["accepted"]),
            "cloud_cover": row["cloud_cover"] if pd.notna(row["cloud_cover"]) else None,
            "failed_filter": row["failed_filter"] if pd.notna(row["failed_filter"]) else None,
            "failure_reason": row["failure_reason"] if pd.notna(row["failure_reason"]) else None
        })
    
    return scenes


def create_comparison_figure(scenes_data: list, output_path: str = "reports/visuals/qc_comparison.png"):
    """Create figure with all scenes from pipeline report."""
    n = len(scenes_data)
    cols = min(n, 5)  # Max 5 columns
    rows = (n + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    for idx, scene_info in enumerate(scenes_data):
        ax = axes[idx] if n > 1 else axes[0]
        
        # Load and display
        try:
            rgb = load_rgb(scene_info["path"])
            ax.imshow(rgb)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\n{str(e)[:50]}", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Title with status and failed filter
        status = "✅ ACCEPTED" if scene_info["accepted"] else "❌ REJECTED"
        name = Path(scene_info["path"]).stem[:20]
        cloud = scene_info.get("cloud_cover", "N/A")
        failed = scene_info.get("failed_filter", "")
        
        # Build title
        title_lines = [f"{status}", f"{name}"]
        if isinstance(cloud, float):
            title_lines.append(f"Cloud: {cloud:.1f}%")
        if failed:
            title_lines.append(f"Filter: {failed}")
        
        title = "\n".join(title_lines)
        color = "green" if scene_info["accepted"] else "red"
        ax.set_title(title, fontsize=8, color=color, fontweight='bold')
        ax.axis("off")
    
    # Hide unused subplots
    total_axes = len(axes) if isinstance(axes, np.ndarray) else 1
    for idx in range(n, total_axes):
        axes[idx].axis("off")
    
    plt.suptitle("EO QC Pipeline Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    # Load from pipeline report instead of hardcoding
    scenes = load_scenes_from_report()
    print(f"Loaded {len(scenes)} scenes from pipeline report")
    
    create_comparison_figure(scenes, "reports/visuals/all_scenes_qc_comparison.png")