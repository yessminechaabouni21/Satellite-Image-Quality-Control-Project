# src/visualization/visualize_defects.py
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import numpy as np
import pandas as pd


def load_rgb_crop(scene_path: str, center_y: int = 5490, center_x: int = 5490, 
                  crop_size: int = 100):
    """Load RGB at FULL RESOLUTION from a centered crop."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    
    b02 = list(granule.rglob("*_B02.jp2"))[0]
    b03 = list(granule.rglob("*_B03.jp2"))[0]
    b04 = list(granule.rglob("*_B04.jp2"))[0]
    
    half = crop_size // 2
    
    with rasterio.open(b04) as src_r, \
         rasterio.open(b03) as src_g, \
         rasterio.open(b02) as src_b:
        
        window = rasterio.windows.Window(
            col_off=center_x - half, 
            row_off=center_y - half,
            width=crop_size, 
            height=crop_size
        )
        
        r = src_r.read(1, window=window).astype(np.float32) / 10000
        g = src_g.read(1, window=window).astype(np.float32) / 10000
        b = src_b.read(1, window=window).astype(np.float32) / 10000
    
    rgb = np.dstack([r, g, b])
    rgb = np.clip(rgb, 0, 0.3) / 0.3
    return rgb


def create_figure(scene_batch, results, cols=3, crop_size=100):
    """Create a single figure with up to (cols * rows) images."""
    n = len(scene_batch)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Handle single subplot case
    if n == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    for idx, scene_path in enumerate(scene_batch):
        ax = axes[idx]
        
        try:
            rgb = load_rgb_crop(str(scene_path), crop_size=crop_size)
            ax.imshow(rgb, interpolation='nearest')
            
            # Pixel grid
            ax.set_xticks(np.arange(-0.5, crop_size, 10), minor=True)
            ax.set_yticks(np.arange(-0.5, crop_size, 10), minor=True)
            ax.grid(which='minor', color='white', alpha=0.3, linewidth=0.5)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error", ha='center', va='center', transform=ax.transAxes)
        
        # Match defect to results
        scene_name = scene_path.name
        prefix = scene_name.split('_')[0]
        result_row = results[results['defect'].str.startswith(prefix)]
        
        if len(result_row) > 0:
            caught = result_row.iloc[0]['caught']
            filter_name = result_row.iloc[0]['failed_filter']
            status = "CAUGHT" if caught else "MISSED"
            color = "green" if caught else "red"
        else:
            status = "?"
            filter_name = "N/A"
            color = "gray"
        
        name = scene_name[:25]
        title = f"{status}: {name}\n({filter_name})"
        ax.set_title(title, fontsize=8, color=color)
        ax.set_xlabel("Pixels", fontsize=7)
        ax.set_ylabel("Pixels", fontsize=7)
    
    # Hide unused subplots
    total = len(axes)
    for idx in range(n, total):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_defects(defect_dir: str = "data/defective", 
                      report_path: str = "reports/defect_injection_results.csv",
                      cols: int = 3,
                      max_per_figure: int = 9):
    """
    Visualize all defective scenes. Automatically splits into multiple figures
    if there are more than max_per_figure scenes.
    """
    results = pd.read_csv(report_path)
    scene_paths = sorted(Path(defect_dir).glob("*.SAFE"))
    
    if len(scene_paths) == 0:
        print("No defective scenes found!")
        return
    
    n = len(scene_paths)
    print(f"Found {n} defective scenes.")
    
    # Split into batches
    batches = [
        scene_paths[i:i + max_per_figure] 
        for i in range(0, n, max_per_figure)
    ]
    
    saved_paths = []
    
    for fig_idx, batch in enumerate(batches):
        print(f"Creating figure {fig_idx + 1}/{len(batches)} with {len(batch)} images...")
        
        fig = create_figure(batch, results, cols=cols)
        
        # Save each figure with numbered suffix
        if len(batches) == 1:
            output = "reports/visuals/defect_visualization.png"
        else:
            output = f"reports/visuals/defect_visualization_{fig_idx + 1:02d}.png"
        
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches='tight')
        saved_paths.append(output)
        print(f"  Saved: {output}")
        
        plt.show()
        plt.close(fig)  # Free memory
    
    print(f"\nDone! Saved {len(saved_paths)} figure(s).")
    return saved_paths


if __name__ == "__main__":
    visualize_defects()