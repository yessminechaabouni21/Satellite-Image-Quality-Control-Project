# src/visualization/select_diverse_subset.py
import numpy as np
import rasterio
from pathlib import Path
from typing import List, Dict, Tuple


def load_band_stats(scene_path: str) -> Dict[str, float]:
    """Quickly compute spectral statistics for diversity scoring."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    
    # Use B04 (Red) as proxy for scene diversity
    b04_files = list(granule.rglob("*_B04.jp2"))
    if not b04_files:
        return {"diversity_score": 0.0, "mean": 0.0, "std": 0.0}
    
    with rasterio.open(b04_files[0]) as src:
        data = src.read(1).astype(np.float32) / 10000
        
        # Sample every 10th pixel for speed (10% of data)
        sampled = data[::10, ::10]
        
        return {
            "diversity_score": float(np.std(sampled)),  # Higher std = more diverse
            "mean": float(np.mean(sampled)),
            "std": float(np.std(sampled)),
            "cloud_cover": None  # Will be filled from metadata
        }


def select_diverse_subset(
    scenes: List[Dict],
    n_accepted: int = 2,
    n_rejected: int = 2
) -> Tuple[List[Dict], List[Dict]]:
    """
    Select most visually diverse scenes from accepted and rejected.
    
    Strategy:
    - Accepted: Pick highest diversity (most interesting clear scenes)
    - Rejected: Pick most diverse rejection reasons (cloud % spread)
    """
    
    # Split by status
    accepted = [s for s in scenes if s.get("accepted", False)]
    rejected = [s for s in scenes if not s.get("accepted", False)]
    
    # Score accepted scenes by spectral diversity
    for scene in accepted:
        stats = load_band_stats(scene["path"])
        scene["diversity_score"] = stats["diversity_score"]
        scene["mean_reflectance"] = stats["mean"]
    
    # Sort accepted by diversity (highest first = most interesting)
    accepted_sorted = sorted(accepted, key=lambda x: x.get("diversity_score", 0), reverse=True)
    
    # For rejected, pick extremes to show range
    # Sort by cloud cover to get spread
    rejected_sorted = sorted(rejected, key=lambda x: x.get("cloud_cover", 0), reverse=True)
    
    # Select:
    # - Accepted: highest diversity + one moderate (for comparison)
    selected_accepted = []
    if len(accepted_sorted) >= 1:
        selected_accepted.append(accepted_sorted[0])  # Most diverse
    if len(accepted_sorted) >= 2:
        # Pick one with different mean reflectance (e.g., darker vs brighter)
        remaining = [s for s in accepted_sorted[1:] if abs(s.get("mean_reflectance", 0) - accepted_sorted[0].get("mean_reflectance", 0)) > 0.05]
        if remaining:
            selected_accepted.append(remaining[0])
        else:
            selected_accepted.append(accepted_sorted[1])
    
    # - Rejected: highest cloud + lowest cloud (to show threshold effect)
    selected_rejected = []
    if len(rejected_sorted) >= 1:
        selected_rejected.append(rejected_sorted[0])  # Most cloudy
    if len(rejected_sorted) >= 2:
        # Find one closest to threshold (e.g., ~20-30% cloud)
        near_threshold = sorted(
            rejected_sorted[1:],
            key=lambda x: abs(x.get("cloud_cover", 0) - 25.0)
        )
        if near_threshold:
            selected_rejected.append(near_threshold[0])
        else:
            selected_rejected.append(rejected_sorted[-1])  # Least cloudy rejected
    
    return selected_accepted[:n_accepted], selected_rejected[:n_rejected]


def print_selection(accepted: List[Dict], rejected: List[Dict]):
    """Pretty print the selected scenes."""
    print(f"\n{'='*60}")
    print("SELECTED 2×2 SUBSET")
    print(f"{'='*60}\n")
    
    print("✅ ACCEPTED (Clear scenes):")
    for i, s in enumerate(accepted, 1):
        name = Path(s["path"]).stem[:30]
        print(f"  {i}. {name}")
        print(f"     Cloud: {s.get('cloud_cover', 'N/A'):.1f}% | Diversity: {s.get('diversity_score', 0):.4f}")
        print()
    
    print("❌ REJECTED (Cloudy scenes):")
    for i, s in enumerate(rejected, 1):
        name = Path(s["path"]).stem[:30]
        print(f"  {i}. {name}")
        print(f"     Cloud: {s.get('cloud_cover', 'N/A'):.1f}% | Filter: {s.get('failed_filter', 'N/A')}")
        print()
    
    print(f"{'='*60}")


# Example usage
if __name__ == "__main__":
    # Your 10 scenes
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
    
    acc, rej = select_diverse_subset(scenes)
    print_selection(acc, rej)