# src/defects/test_defect.py
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
from src.filters.stripe_filter import StripeFilter
import pandas as pd
from src.database import init_db, log_defect


def run_all_defect_tests():
    base = r"data\extracted\S2A_MSIL1C_20260406T100051_N0512_R122_T32SPF_20260406T151531.SAFE"
    output_dir = "data/defective"
    Path(output_dir).mkdir(exist_ok=True)

    # SAME PIPELINE as run_all_scenes.py — the working one
    pipeline = Pipeline([
        MetadataFilter(max_cloud=60.0),           # Match run_all_scenes
        MissingBandsFilter(),
        TOAScalingFilter(),
        NoDataFilter(max_unexpected_nodata_ratio=0.05, min_unique_values=100),
        BlurFilter(min_variance=15.0),            # Fixed version (handles uniform)
        StripeFilter(max_periodic_power_ratio=0.3),
        NoiseFilter(max_noise_std_ratio=0.03),    # FIXED: new parameter, no dead pixels
    ])
    conn = init_db()
    
    defects = [
        ("CORRUPTION_zero_50", lambda: inject_zero_band(base, output_dir, "B04", 0.5)),
        ("CORRUPTION_flat", lambda: inject_flat_band(base, output_dir, "B04", 5000)),
        ("CORRUPTION_missing_B11", lambda: inject_missing_band(base, output_dir, "B11")),
        # Renamed: these are actually "moderate" not "weak"
        ("NOISE_moderate", lambda: inject_noise(base, output_dir, "B04", 500)),    # Was weak=100
        ("NOISE_severe", lambda: inject_noise(base, output_dir, "B04", 2000)),      # Was strong=500
        ("BLUR_moderate", lambda: inject_blur(base, output_dir, "B04", 15)),       # Was mild=7
        ("BLUR_severe", lambda: inject_blur(base, output_dir, "B04", 51)),         # Was severe=21
        ("STRIPE_light", lambda: inject_stripes(base, output_dir, "B04", 500)),
        ("STRIPE_heavy", lambda: inject_stripes(base, output_dir, "B04", 2000)),
        # Haze:  add HazeFilter if needed
]
    
    results = []
    
    for defect_name, defect_func in defects:
        print(f"\nGenerating: {defect_name}")
        try:
            scene = defect_func()
            print(f"  Testing: {Path(scene).name}")
            
            result = pipeline.run(scene)
            log_defect(conn, scene, defect_name, result)
            
            caught = not result["accepted"]
            failed_filter = next((k for k, v in result["results"].items() if not v["passed"]), None)
            
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
    
    if total - caught > 0:
        print(f"\nMissed/Error defects:")
        print(df[~df["caught"]][["defect", "failed_filter", "status"]].to_string())
    
    df.to_csv("reports/defect_injection_results.csv", index=False)
    print(f"\n💾 Saved: reports/defect_injection_results.csv")
    
    return df


if __name__ == "__main__":
    run_all_defect_tests()