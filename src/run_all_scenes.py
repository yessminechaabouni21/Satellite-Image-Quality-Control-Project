

from pathlib import Path
import pandas as pd

# These imports work when run as module (python -m src.run_all_scenes)
from src.pipeline.orchestrator import Pipeline
from src.filters.metadata_filter import MetadataFilter
from src.filters.noData_filter import NoDataFilter
from src.filters.toascaling_filter import TOAScalingFilter
from src.filters.blur_filter import BlurFilter
from src.filters.noise_filter import NoiseFilter
from src.filters.missing_bands_filter import MissingBandsFilter
from src.filters.haze_filter import HazeFilter
from src.filters.duplicate_filter import DuplicateFilter
from src.filters.radiometric_filter import RadiometricResolutionFilter



def run_all_scenes(extracted_dir: str = "data/extracted"):
    """Process all .SAFE scenes and return results as a DataFrame."""
    
    # Find all scenes
    scene_paths = sorted(Path(extracted_dir).glob("*.SAFE"))
    print(f"Found {len(scene_paths)} scenes to process\n")
    
    # Initialize pipeline
    pipeline = Pipeline([
        DuplicateFilter(),
        MetadataFilter(max_cloud=60.0),
        MissingBandsFilter(),
        RadiometricResolutionFilter(),  
        TOAScalingFilter(),
        NoDataFilter(max_unexpected_nodata_ratio=0.05, min_unique_values=100),
        HazeFilter(max_hot_score=0.15, max_haze_pixels_ratio=0.3), 
        BlurFilter(min_variance=15.0),
        NoiseFilter(max_noise_uniformity=0.7, max_dead_pixel_ratio=0.001)
    ])
    # Collect results
    all_results = []
    
    for scene_path in scene_paths:
        scene_name = scene_path.name
        print(f"Processing: {scene_name} ...", end=" ")
        
        result = pipeline.run(str(scene_path))
        
        # Flatten results into a single row
        row = {
            "scene": scene_name,
            "accepted": result["accepted"],
            "failed_filter": None,
            "failure_reason": None,
            "cloud_cover": None,
            "nodata_ratio": None,
            "unique_values": None,
            "max_dn": None,
            "toa_dtype": None
        }
        
        # Extract metrics from each filter
        for filter_name, filter_result in result["results"].items():
            metrics = filter_result.get("metrics", {})
            
            if filter_name == "MetadataFilter":
                row["cloud_cover"] = metrics.get("cloud_cover")
            
            elif filter_name == "NoDataFilter":
                row["nodata_ratio"] = metrics.get("nodata_ratio")
                row["unique_values"] = metrics.get("unique_values")
            
            elif filter_name == "TOAScalingFilter":
                row["max_dn"] = metrics.get("max_dn")
                row["toa_dtype"] = metrics.get("dtype")
            
            # Track first failure
            if not filter_result["passed"] and row["failed_filter"] is None:
                row["failed_filter"] = filter_name
                row["failure_reason"] = filter_result.get("reason")
        
        all_results.append(row)
        status = "✅ ACCEPTED" if result["accepted"] else f"❌ REJECTED ({row['failed_filter']})"
        print(status)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns for readability
    cols = [
        "scene", "accepted", "failed_filter", "failure_reason",
        "cloud_cover", "nodata_ratio", "unique_values", "max_dn", "toa_dtype"
    ]
    df = df[cols]
    
    return df

def main():
    # Run pipeline on all scenes
    df = run_all_scenes()
    
    # Display table
    print(f"\n{'='*80}")
    print("PIPELINE RESULTS SUMMARY")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))
    
    # Summary stats
    total = len(df)
    accepted = df["accepted"].sum()
    rejected = total - accepted
    
    print(f"\n{'='*80}")
    print(f"Total scenes: {total}")
    print(f"Accepted: {accepted} ({accepted/total*100:.1f}%)")
    print(f"Rejected: {rejected} ({rejected/total*100:.1f}%)")
    
    if rejected > 0:
        print(f"\nRejection breakdown:")
        print(df[df["accepted"] == False]["failed_filter"].value_counts().to_string())
    
    # Save to CSV
    output_path = "reports/pipeline_report.csv"
    Path("reports").mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")

if __name__ == "__main__":
    main()
