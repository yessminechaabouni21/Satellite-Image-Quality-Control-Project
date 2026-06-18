# src/defects/test_all.py
from pathlib import Path
from src.defects.corruption import inject_zero_band, inject_flat_band, inject_missing_band
from src.defects.noise_blur import inject_noise, inject_blur, inject_stripes
from src.defects.haze import inject_haze
from src.pipeline.orchestrator import Pipeline
from src.filters.metadata_filter import MetadataFilter
from src.filters.noData_filter import NoDataFilter
from src.filters.toascaling_filter import TOAScalingFilter
from src.filters.blur_filter import BlurFilter
from src.filters.noise_filter import NoiseFilter
from src.filters.missing_bands_filter import MissingBandsFilter
import pandas as pd


def run_all_defect_tests():
    base = r"data\extracted\S2A_MSIL1C_20260406T100051_N0512_R122_T32SPF_20260406T151531.SAFE"
    output_dir = "data/defective"
    Path(output_dir).mkdir(exist_ok=True)
    
    pipeline = Pipeline([
        MetadataFilter(max_cloud=20.0),
        MissingBandsFilter(),
        TOAScalingFilter(),
        NoDataFilter(max_unexpected_nodata_ratio=0.05, min_unique_values=100),
        BlurFilter(min_variance=15.0),
        NoiseFilter(max_noise_uniformity=0.7, max_dead_pixel_ratio=0.001)
    ])
    
    # Generate all defect types
    defects = [
        ("CORRUPTION_zero_50", lambda: inject_zero_band(base, output_dir, "B04", 0.5)),
        ("CORRUPTION_flat", lambda: inject_flat_band(base, output_dir, "B04", 5000)),
        ("CORRUPTION_missing_B11", lambda: inject_missing_band(base, output_dir, "B11")),
        ("NOISE_weak", lambda: inject_noise(base, output_dir, "B04", 10)),
        ("NOISE_strong", lambda: inject_noise(base, output_dir, "B04", 50)),
        ("BLUR_mild", lambda: inject_blur(base, output_dir, "B04", 7)),
        ("BLUR_severe", lambda: inject_blur(base, output_dir, "B04", 21)),
        ("STRIPE_light", lambda: inject_stripes(base, output_dir, "B04", 500)),
        ("STRIPE_heavy", lambda: inject_stripes(base, output_dir, "B04", 2000)),
        ("HAZE_light", lambda: inject_haze(base, output_dir, 0.2)),
        ("HAZE_heavy", lambda: inject_haze(base, output_dir, 0.5)),
    ]
    
    results = []
    
    for defect_name, defect_func in defects:
        print(f"\nGenerating: {defect_name}")
        try:
            scene = defect_func()
            print(f"  Testing: {Path(scene).name}")
            
            result = pipeline.run(scene)
            
            caught = not result["accepted"]
            failed_filter = next((k for k,v in result["results"].items() if not v["passed"]), None)
            
            results.append({
                "defect": defect_name,
                "caught": caught,
                "failed_filter": failed_filter,
                "status": "✅ CAUGHT" if caught else "❌ MISSED"
            })
            
            print(f"  {results[-1]['status']} by {failed_filter}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "defect": defect_name,
                "caught": False,
                "failed_filter": "ERROR",
                "status": f"ERROR: {e}"
            })
    
    # Summary
    df = pd.DataFrame(results)
    total = len(df)
    caught = df["caught"].sum()
    
    print(f"\n{'='*60}")
    print(f"DEFECT INJECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total defects: {total}")
    print(f"Caught: {caught} ({caught/total*100:.1f}%)")
    print(f"Missed: {total - caught}")
    print(f"\nMissed defects:")
    print(df[~df["caught"]][["defect", "failed_filter"]].to_string())
    
    df.to_csv("reports/defect_injection_results.csv", index=False)
    print(f"\n💾 Saved: reports/defect_injection_results.csv")
    
    return df


if __name__ == "__main__":
    run_all_defect_tests()