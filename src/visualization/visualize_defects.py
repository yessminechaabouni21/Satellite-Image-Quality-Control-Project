# src/visualization/visualize_defects.py
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import numpy as np
import pandas as pd


def load_rgb_fast(scene_path: str, scale=4):
    """Load RGB at reduced resolution for speed."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    
    b02 = list(granule.rglob("*_B02.jp2"))[0]
    b03 = list(granule.rglob("*_B03.jp2"))[0]
    b04 = list(granule.rglob("*_B04.jp2"))[0]
    
    with rasterio.open(b04) as src_r, \
         rasterio.open(b03) as src_g, \
         rasterio.open(b02) as src_b:
        
        # Read at 1/4 resolution (16x faster)
        h, w = src_r.height // scale, src_r.width // scale
        r = src_r.read(1, out_shape=(h, w)).astype(np.float32) / 10000
        g = src_g.read(1, out_shape=(h, w)).astype(np.float32) / 10000
        b = src_b.read(1, out_shape=(h, w)).astype(np.float32) / 10000
    
    rgb = np.dstack([r, g, b])
    rgb = np.clip(rgb, 0, 0.3) / 0.3
    return rgb


def visualize_defects(defect_dir: str = "data/defective", report_path: str = "reports/defect_injection_results.csv"):
    """Visualize defective scenes quickly."""
    
    results = pd.read_csv(report_path)
    scene_paths = sorted(Path(defect_dir).glob("*.SAFE"))
    
    if len(scene_paths) == 0:
        print("No defective scenes found!")
        return
    
    n = len(scene_paths)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    for idx, scene_path in enumerate(scene_paths):
        ax = axes[idx] if n > 1 else axes[0]
        
        print(f"Loading {scene_path.name}...")
        
        try:
            rgb = load_rgb_fast(str(scene_path), scale=8)  # Even smaller: 1/8
            ax.imshow(rgb)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error", ha='center', va='center', transform=ax.transAxes)
        
        # Get result
        scene_name = scene_path.name
        # Match by prefix (e.g., "NOISE_50_" -> "NOISE")
        prefix = scene_name.split('_')[0]
        result_row = results[results['defect'].str.startswith(prefix)]
        
        if len(result_row) > 0:
            caught = result_row.iloc[0]['caught']
            filter_name = result_row.iloc[0]['failed_filter']
            status = "✅" if caught else "❌"
            color = "green" if caught else "red"
        else:
            status = "?"
            filter_name = "N/A"
            color = "gray"
        
        name = scene_name[:20]
        title = f"{status} {name}\n{filter_name}"
        ax.set_title(title, fontsize=8, color=color)
        ax.axis('off')
    
    # Hide unused
    for idx in range(n, len(axes) if isinstance(axes, np.ndarray) else 1):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    output = "reports/visuals/defect_visualization.png"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=100, bbox_inches='tight')
    print(f"\nSaved: {output}")
    plt.show()


if __name__ == "__main__":
    visualize_defects()