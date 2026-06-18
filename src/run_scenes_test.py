# src/run_all_scenes.py
from pathlib import Path
import pandas as pd
import argparse
from src.pipeline.orchestrator import Pipeline
from src.filters.metadata_filter import MetadataFilter
from src.filters.noData_filter import NoDataFilter
from src.filters.toascaling_filter import TOAScalingFilter
from src.filters.blur_filter import BlurFilter
from src.filters.noise_filter import NoiseFilter
from src.filters.missing_bands_filter import MissingBandsFilter


def run_all_scenes(extracted_dir: str = "data/extracted", output_csv: str = "reports/pipeline_report.csv"):
    """Process all .SAFE scenes in directory and save report."""
    scene_paths = sorted(Path(extracted_dir).glob("*.SAFE"))
    print(f"Found {len(scene_paths)} scenes in {extracted_dir}\n")
    
    if len(scene_paths) == 0:
        print(f"WARNING: No .SAFE scenes found in {extracted_dir}")
        return pd.DataFrame()
    
    pipeline = Pipeline([
        MetadataFilter(max_cloud=20.0),
        MissingBandsFilter(required_bands=["B02", "B03", "B04", "B08", "B11"]),
        
        NoDataFilter(max_unexpected_nodata_ratio=0.05, min_unique_values=100),
        BlurFilter(min_variance=15.0),
        NoiseFilter(max_noise_uniformity=0.7, max_dead_pixel_ratio=0.001),
        TOAScalingFilter()
    ])
    
    all_results = []
    
    for scene_path in scene_paths:
        scene_name = scene_path.name
        print(f"Processing: {scene_name} ...", end=" ")
        
        result = pipeline.run(str(scene_path))
        
        # Flatten results
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
        
        for filter_name, filter_result in result["results"].items():
            metrics = filter_result.get("metrics", {})
            
            if filter_name == "MetadataFilter":
                row["cloud_cover"] = metrics.get("cloud_cover")
            elif filter_name == "NoDataFilter":
                row["nodata_ratio"] = metrics.get("unexpected_nodata_ratio")
                row["unique_values"] = metrics.get("unique_values")
            elif filter_name == "TOAScalingFilter":
                row["max_dn"] = metrics.get("max_dn")
                row["toa_dtype"] = metrics.get("dtype")
            
            if not filter_result["passed"] and row["failed_filter"] is None:
                row["failed_filter"] = filter_name
                row["failure_reason"] = filter_result.get("reason")
        
        all_results.append(row)
        status = "✅ ACCEPTED" if result["accepted"] else f"❌ REJECTED ({row['failed_filter']})"
        print(status)
    
    df = pd.DataFrame(all_results)
    
    # Reorder columns
    cols = [
        "scene", "accepted", "failed_filter", "failure_reason",
        "cloud_cover", "nodata_ratio", "unique_values", "max_dn", "toa_dtype"
    ]
    df = df[cols] if all(c in df.columns for c in cols) else df
    
    # Summary
    total = len(df)
    accepted = df["accepted"].sum()
    rejected = total - accepted
    
    print(f"\n{'='*60}")
    print(f"PIPELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Directory: {extracted_dir}")
    print(f"Total scenes: {total}")
    print(f"Accepted: {accepted} ({accepted/total*100:.1f}%)")
    print(f"Rejected: {rejected} ({rejected/total*100:.1f}%)")
    
    if rejected > 0:
        print(f"\nRejection breakdown:")
        print(df[df["accepted"] == False]["failed_filter"].value_counts().to_string())
    
    # Save
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Saved to: {output_csv}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EO QC Pipeline")
    parser.add_argument("--dir", type=str, default="data/extracted", help="Directory with .SAFE scenes")
    parser.add_argument("--output", type=str, default="reports/pipeline_report.csv", help="Output CSV path")
    args = parser.parse_args()
    
    run_all_scenes(extracted_dir=args.dir, output_csv=args.output)