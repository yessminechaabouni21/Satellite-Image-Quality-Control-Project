# src/real_world/confusion_matrix.py
#
# Runs your rule-based pipeline on every .SAFE in data/esa_reference/,
# joins the result against the ESA ground-truth labels in esa_reference table,
# and prints a confusion matrix + per-filter breakdown.
#
# Run from repo root:
#   python -m src.real_world.confusion_matrix
#
import sqlite3
from pathlib import Path

import pandas as pd

from src.pipeline.orchestrator import Pipeline
from src.filters.metadata_filter import MetadataFilter
from src.filters.noData_filter import NoDataFilter
from src.filters.toascaling_filter import TOAScalingFilter
from src.filters.blur_filter import BlurFilter
from src.filters.noise_filter import NoiseFilter
from src.filters.missing_bands_filter import MissingBandsFilter
from src.filters.stripe_filter import StripeFilter

DB_PATH      = "reports/eo_qc.db"
SCENE_DIR    = "data/esa_reference"
REPORT_DIR   = Path("reports")


# ── pipeline (same stack as run_esa_scenes.py) ────────────────────────────────
def build_pipeline():
    return Pipeline([
        MetadataFilter(max_cloud=60.0),
        MissingBandsFilter(),
        TOAScalingFilter(),
        NoDataFilter(max_unexpected_nodata_ratio=0.05, min_unique_values=100),
        BlurFilter(min_variance=15.0),
        StripeFilter(max_periodic_power_ratio=0.3),
        NoiseFilter(max_noise_std_ratio=0.15),
    ])


# ── run pipeline on all scenes ────────────────────────────────────────────────
def run_pipeline(scene_dir):
    pipeline = build_pipeline()
    scenes   = sorted(Path(scene_dir).glob("*.SAFE"))
    print(f"Found {len(scenes)} .SAFE scenes in {scene_dir}\n")

    rows = []
    for i, scene in enumerate(scenes, 1):
        print(f"[{i}/{len(scenes)}] {scene.name[:65]}", end=" ... ")
        result = pipeline.run(str(scene))

        # collect all metrics from every filter that ran
        metrics = {}
        for fname, fres in result["results"].items():
            for k, v in (fres.get("metrics") or {}).items():
                metrics[f"{fname}__{k}"] = v

        failed_filter  = result.get("failed_filter")
        failure_reason = result.get("failure_reason")

        print("PASS" if result["accepted"] else f"FAIL ({failed_filter})")
        rows.append({
            "scene_name":      scene.name,
            "pipeline_passed": result["accepted"],
            "failed_filter":   failed_filter,
            "failure_reason":  failure_reason,
            **metrics,
        })

    return pd.DataFrame(rows)


# ── load ESA ground truth from DB ─────────────────────────────────────────────
def load_esa_labels(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT scene_name,
               esa_flag,
               failed_indicator,
               cloud_cover_pct,
               degraded_msi_pct,
               nodata_pct,
               saturated_defective_pct
        FROM   esa_reference
    """, conn)
    conn.close()
    df["esa_defective"] = (df["esa_flag"] == "FAILED")
    return df


# ── confusion matrix ──────────────────────────────────────────────────────────
def confusion_matrix(merged):
    # pipeline_rejected = True means pipeline said scene is BAD
    merged["pipeline_rejected"] = ~merged["pipeline_passed"]

    tp = int(( merged["esa_defective"] &  merged["pipeline_rejected"]).sum())
    fn = int(( merged["esa_defective"] & ~merged["pipeline_rejected"]).sum())
    fp = int((~merged["esa_defective"] &  merged["pipeline_rejected"]).sum())
    tn = int((~merged["esa_defective"] & ~merged["pipeline_rejected"]).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    fpr       = fp / (fp + tn) if (fp + tn) else 0.0   # false-positive rate

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr,
    }


# ── print helpers ─────────────────────────────────────────────────────────────
def print_matrix(cm, n_esa_scenes):
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX  (pipeline vs ESA ground truth)")
    print(f"{'='*60}")
    print(f"  Scenes evaluated : {n_esa_scenes}")
    print()
    print(f"                      ESA FAILED   ESA PASSED")
    print(f"  Pipeline REJECTED      {cm['tp']:^6}       {cm['fp']:^6}   "
          f"← True Pos / False Pos")
    print(f"  Pipeline ACCEPTED      {cm['fn']:^6}       {cm['tn']:^6}   "
          f"← False Neg / True Neg")
    print()
    print(f"  Precision : {cm['precision']:.2f}   "
          f"(of scenes rejected, how many were truly defective)")
    print(f"  Recall    : {cm['recall']:.2f}   "
          f"(of ESA-FAILED scenes, how many did we catch)")
    print(f"  F1        : {cm['f1']:.2f}")
    print(f"  False-positive rate: {cm['fpr']:.2f}   "
          f"(fraction of clean scenes wrongly rejected)")


def print_per_indicator(merged):
    print(f"\n{'='*60}")
    print("BREAKDOWN BY ESA FAILED INDICATOR")
    print(f"{'='*60}")
    failed_esa = merged[merged["esa_defective"]]
    if failed_esa.empty:
        print("  No ESA-FAILED scenes in merged set.")
        return
    for ind in sorted(failed_esa["failed_indicator"].dropna().unique()):
        sub   = failed_esa[failed_esa["failed_indicator"] == ind]
        caught = sub["pipeline_rejected"].sum()
        print(f"  {ind:25s} : {caught}/{len(sub)} caught  "
              f"(by: {', '.join(sub[sub['pipeline_rejected']]['failed_filter'].dropna().unique())})")


def print_fp_detail(merged):
    fp_scenes = merged[(~merged["esa_defective"]) & merged["pipeline_rejected"]]
    if fp_scenes.empty:
        print("\n  No false positives — pipeline never wrongly rejected a clean scene.")
        return
    print(f"\n{'='*60}")
    print(f"FALSE POSITIVES ({len(fp_scenes)} clean scenes wrongly rejected)")
    print(f"{'='*60}")
    for _, r in fp_scenes.iterrows():
        print(f"  {r['scene_name'][:55]}")
        print(f"    filter: {r['failed_filter']}  reason: {r['failure_reason']}")


def print_fn_detail(merged):
    fn_scenes = merged[merged["esa_defective"] & ~merged["pipeline_rejected"]]
    if fn_scenes.empty:
        print("\n  No false negatives — pipeline caught all ESA-FAILED scenes.")
        return
    print(f"\n{'='*60}")
    print(f"FALSE NEGATIVES ({len(fn_scenes)} defective scenes the pipeline MISSED)")
    print(f"{'='*60}")
    for _, r in fn_scenes.iterrows():
        print(f"  {r['scene_name'][:55]}")
        print(f"    ESA indicator: {r['failed_indicator']}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if not Path(DB_PATH).exists():
        raise SystemExit(f"DB not found: {DB_PATH}")

    # 1. Run pipeline on every downloaded scene
    pipeline_df = run_pipeline(SCENE_DIR)

    # 2. Load ESA labels
    esa_df = load_esa_labels(DB_PATH)

    # 3. Merge — inner join on scene_name (only scenes we actually downloaded)
    merged = pipeline_df.merge(esa_df, on="scene_name", how="inner")
    print(f"\nMatched {len(merged)} scenes with ESA labels "
          f"({len(pipeline_df) - len(merged)} scenes not in DB, skipped)")

    if merged.empty:
        raise SystemExit("No scenes matched ESA labels. "
                         "Check that scene names in data/esa_reference/ "
                         "match those in the esa_reference table.")

    # 4. Confusion matrix
    cm = confusion_matrix(merged)
    print_matrix(cm, len(merged))
    print_per_indicator(merged)
    print_fn_detail(merged)
    print_fp_detail(merged)

    # 5. Save full results
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "confusion_matrix_results.csv"

    save_cols = ["scene_name", "esa_flag", "failed_indicator",
                 "esa_defective", "pipeline_passed", "pipeline_rejected",
                 "failed_filter", "failure_reason", "cloud_cover_pct"]
    merged[save_cols].to_csv(out, index=False)
    print(f"\n{'='*60}")
    print(f"Saved full results to: {out}")

    # 6. Summary for thesis
    print(f"\n{'='*60}")
    print("THESIS SUMMARY")
    print(f"{'='*60}")
    print(f"Rule-based pipeline evaluated on {len(merged)} real Sentinel-2 scenes "
          f"(tile T32SPF, 2024-2026):")
    print(f"  - {cm['tp']} / {cm['tp']+cm['fn']} ESA-FAILED scenes correctly rejected "
          f"(recall = {cm['recall']:.0%})")
    print(f"  - {cm['tn']} / {cm['tn']+cm['fp']} ESA-PASSED scenes correctly accepted "
          f"(specificity = {cm['tn']/(cm['tn']+cm['fp']):.0%})")
    print(f"  - False-positive rate: {cm['fpr']:.0%}")
    print(f"  - F1 score: {cm['f1']:.2f}")


if __name__ == "__main__":
    main()