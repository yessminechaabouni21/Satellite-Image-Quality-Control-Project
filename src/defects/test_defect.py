# src/defects/test_defect.py
# Defect-injection harness:
#   * loops over ALL clean .SAFE scenes in data/extracted/
#   * injects every defect type
#   * sweeps multiple severities for parametric defects (noise/blur/stripe)
#   * logs each run to SQLite with base_scene / defect_family / severity
#   * uses fast copy (only GRANULE folder, not full .SAFE)

from pathlib import Path
import shutil
import pandas as pd

from src.defects.corruption import (
    inject_zero_band, inject_flat_band, inject_missing_band,
)
from src.defects.noise_blur import (
    inject_noise_fast, inject_blur_fast, inject_stripes_fast
)
from src.pipeline.orchestrator import Pipeline
from src.filters.metadata_filter import MetadataFilter
from src.filters.noData_filter import NoDataFilter
from src.filters.toascaling_filter import TOAScalingFilter
from src.filters.blur_filter import BlurFilter
from src.filters.noise_filter import NoiseFilter
from src.filters.missing_bands_filter import MissingBandsFilter
from src.filters.stripe_filter import StripeFilter
from src.database import init_db, log_defect


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
EXTRACTED_DIR = "data/extracted"
OUTPUT_DIR = "data/defective"
BAND = "B04"

# Severity grids
NOISE_SIGMAS   = [10, 50, 100, 250, 500, 1000]
BLUR_KERNELS   = [3, 7, 15, 21, 51]
STRIPE_INTENS  = [100, 500, 1000, 2000]

# Delete each injected .SAFE after testing + logging
CLEANUP_AFTER_TEST = False


def build_pipeline():
    """Identical filter stack / params to run_all_scenes.py."""
    return Pipeline([
        MetadataFilter(max_cloud=60.0),
        MissingBandsFilter(),
        TOAScalingFilter(),
        NoDataFilter(max_unexpected_nodata_ratio=0.05, min_unique_values=100),
        BlurFilter(min_variance=15.0),
        StripeFilter(max_periodic_power_ratio=0.3),
        NoiseFilter(max_noise_std_ratio=0.15),
    ])


def build_defect_jobs(base_scene_path):
    """
    Return list of (defect_type, defect_family, severity, inject_callable).
    severity is None for categorical defects.
    """
    base = str(base_scene_path)
    jobs = []

    # --- Categorical corruption defects ---
    jobs.append(("CORRUPTION_zero_50", "CORRUPTION", None,
                 lambda: inject_zero_band(base, OUTPUT_DIR, BAND, 0.5)))
    jobs.append(("CORRUPTION_flat", "CORRUPTION", None,
                 lambda: inject_flat_band(base, OUTPUT_DIR, BAND, 5000)))
    jobs.append(("CORRUPTION_missing_B11", "CORRUPTION", None,
                 lambda: inject_missing_band(base, OUTPUT_DIR, "B11")))

    # --- Parametric: NOISE sweep ---
    for sigma in NOISE_SIGMAS:
        jobs.append((f"NOISE_{sigma}", "NOISE", float(sigma),
                     lambda s=sigma: inject_noise_fast(base, OUTPUT_DIR, BAND, s)))

    # --- Parametric: BLUR sweep ---
    for k in BLUR_KERNELS:
        jobs.append((f"BLUR_{k}", "BLUR", float(k),
                     lambda kk=k: inject_blur_fast(base, OUTPUT_DIR, BAND, kk)))

    # --- Parametric: STRIPE sweep ---
    for intensity in STRIPE_INTENS:
        jobs.append((f"STRIPE_{intensity}", "STRIPE", float(intensity),
                     lambda i=intensity: inject_stripes_fast(base, OUTPUT_DIR, BAND, i)))

    return jobs


def _safe_cleanup(scene_path):
    """Delete injected scene dir only if under OUTPUT_DIR."""
    try:
        p = Path(scene_path).resolve()
        out = Path(OUTPUT_DIR).resolve()
        if out in p.parents and p.exists():
            shutil.rmtree(p)
    except Exception as e:
        print(f"  (cleanup skipped: {e})")


def run_all_defect_tests():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    base_scenes = sorted(Path(EXTRACTED_DIR).glob("*.SAFE"))
    if not base_scenes:
        print(f"No .SAFE scenes found in {EXTRACTED_DIR}")
        return pd.DataFrame()

    print(f"Found {len(base_scenes)} base scenes in {EXTRACTED_DIR}")
    print(f"Severities -> noise:{NOISE_SIGMAS} blur:{BLUR_KERNELS} stripe:{STRIPE_INTENS}")
    print(f"Total runs: ~{len(base_scenes) * (3 + len(NOISE_SIGMAS) + len(BLUR_KERNELS) + len(STRIPE_INTENS))}")

    pipeline = build_pipeline()
    conn = init_db()

    results = []

    for si, base_scene in enumerate(base_scenes, 1):
        base_name = base_scene.name
        jobs = build_defect_jobs(base_scene)
        print(f"\n{'#'*70}")
        print(f"# BASE SCENE {si}/{len(base_scenes)}: {base_name}")
        print(f"# {len(jobs)} defect variants")
        print(f"{'#'*70}")

        for defect_type, family, severity, inject in jobs:
            sev_lbl = "-" if severity is None else f"{severity:g}"
            print(f"\n[{family:10s} sev={sev_lbl:>6s}] {defect_type}")

            scene = None
            try:
                scene = inject()
                print(f"  injected: {Path(scene).name}")

                result = pipeline.run(scene)
                log_defect(conn, scene, defect_type, result,
                           base_scene=base_name,
                           severity=severity,
                           defect_family=family)

                caught = not result["accepted"]
                failed_filter = next(
                    (k for k, v in result["results"].items() if not v["passed"]),
                    None,
                )

                results.append({
                    "base_scene": base_name,
                    "defect": defect_type,
                    "family": family,
                    "severity": severity,
                    "caught": caught,
                    "failed_filter": failed_filter,
                    "status": "CAUGHT" if caught else "MISSED",
                })
                print(f"  {'CAUGHT' if caught else 'MISSED'} by {failed_filter}")

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "base_scene": base_name,
                    "defect": defect_type,
                    "family": family,
                    "severity": severity,
                    "caught": False,
                    "failed_filter": "ERROR",
                    "status": f"ERROR: {e}",
                })
            finally:
                if CLEANUP_AFTER_TEST and scene is not None:
                    _safe_cleanup(scene)

    conn.close()

    df = pd.DataFrame(results)
    _print_summary(df)

    Path("reports").mkdir(exist_ok=True)
    out_csv = "reports/defect_injection_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    return df


def _print_summary(df):
    if df.empty:
        return

    total = len(df)
    caught = int(df["caught"].sum())

    print(f"\n{'='*70}")
    print("DEFECT INJECTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total runs:   {total}")
    print(f"Caught:       {caught} ({caught/total*100:.1f}%)")
    print(f"Missed/Error: {total - caught}")

    # Detection rate vs severity
    print(f"\n{'='*70}")
    print("DETECTION RATE BY FAMILY x SEVERITY")
    print(f"{'='*70}")
    param = df[df["severity"].notna()]
    if not param.empty:
        tbl = (param.groupby(["family", "severity"])["caught"]
               .agg(n="count", caught="sum")
               .reset_index())
        tbl["rate"] = (100.0 * tbl["caught"] / tbl["n"]).round(0)
        for fam in ["NOISE", "BLUR", "STRIPE"]:
            sub = tbl[tbl["family"] == fam]
            if sub.empty:
                continue
            print(f"\n{fam}:")
            for _, r in sub.iterrows():
                bar = "#" * int(r["rate"] / 10)
                print(f"  sev={r['severity']:>7g} | {int(r['caught'])}/{int(r['n'])} "
                      f"| {r['rate']:5.0f}% {bar}")

    # Categorical defects
    cat = df[df["severity"].isna()]
    if not cat.empty:
        print(f"\n{'='*70}")
        print("CATEGORICAL DEFECTS (catch rate over base scenes)")
        print(f"{'='*70}")
        tbl = (cat.groupby("defect")["caught"]
               .agg(n="count", caught="sum").reset_index())
        for _, r in tbl.iterrows():
            rate = 100.0 * r["caught"] / r["n"]
            print(f"  {r['defect']:24s} | {int(r['caught'])}/{int(r['n'])} | {rate:5.0f}%")

    # Per-base-scene breakdown
    print(f"\n{'='*70}")
    print("PER BASE SCENE")
    print(f"{'='*70}")
    for base, grp in df.groupby("base_scene"):
        caught = int(grp["caught"].sum())
        print(f"  {base[:40]:40s} | {caught}/{len(grp)} caught")


if __name__ == "__main__":
    run_all_defect_tests()