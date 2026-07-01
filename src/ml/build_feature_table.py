# src/ml/build_feature_table.py
#
# Runs your pipeline on every available scene (clean, ESA-reference, synthetic)
# and collects ALL filter metrics into a single labeled feature table.
# Output: reports/ml_features.csv  (one row per scene run)
#
# Label assignment:
#   0 = clean  (ESA-PASSED or your original 10 clean scenes)
#   1 = defective  (ESA-FAILED or synthetic injections)
#
# Run from repo root:
#   python -m src.ml.build_feature_table
#
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.pipeline.orchestrator import Pipeline
from src.filters.metadata_filter import MetadataFilter
from src.filters.noData_filter import NoDataFilter
from src.filters.toascaling_filter import TOAScalingFilter
from src.filters.blur_filter import BlurFilter
from src.filters.noise_filter import NoiseFilter
from src.filters.missing_bands_filter import MissingBandsFilter
from src.filters.stripe_filter import StripeFilter

DB_PATH     = "reports/eo_qc.db"
REPORT_DIR  = Path("reports")
OUT_CSV     = REPORT_DIR / "ml_features.csv"

# Scene source directories
SCENE_DIRS = {
    "clean":      "data/extracted",        # your 10 original clean scenes
    "esa_ref":    "data/esa_reference",    # ESA scenes (PASSED + FAILED)
    "synthetic":  "data/defective",        # injected defects (if kept on disk)
}


# ── pipeline ──────────────────────────────────────────────────────────────────
def build_pipeline():
    return Pipeline([
        MetadataFilter(max_cloud=60.0),
        MissingBandsFilter(),
        TOAScalingFilter(tolerance=200),   # uses pct_above logic now
        NoDataFilter(max_unexpected_nodata_ratio=0.05, min_unique_values=100),
        BlurFilter(min_variance=6.827),
        StripeFilter(max_periodic_power_ratio=0.3),
        NoiseFilter(max_noise_std_ratio=0.15),   # calibrated value
    ])


# ── extra band-level features not in existing filters ─────────────────────────
def compute_extra_features(scene_path: Path) -> dict:
    """
    Additional statistical features computed directly from B04.
    Returns empty dict on any error so it never breaks the pipeline run.
    """
    out = {
        "dn_p05": None, "dn_p25": None, "dn_p75": None, "dn_p95": None,
        "dn_range": None, "dn_iqr": None, "dn_skew": None,
        "inter_band_ratio_nir_red": None,
    }
    try:
        b04 = list(scene_path.rglob("*_B04.jp2")) or \
              list(scene_path.rglob("*B04*.jp2"))
        if not b04:
            return out
        with rasterio.open(b04[0]) as src:
            arr = src.read(1).astype(np.float32)

        valid = arr[arr > 0]
        if valid.size < 100:
            return out

        p05, p25, p75, p95 = np.percentile(valid, [5, 25, 75, 95])
        out["dn_p05"]   = float(p05)
        out["dn_p25"]   = float(p25)
        out["dn_p75"]   = float(p75)
        out["dn_p95"]   = float(p95)
        out["dn_range"] = float(p95 - p05)
        out["dn_iqr"]   = float(p75 - p25)

        mean = valid.mean()
        std  = valid.std()
        if std > 0:
            out["dn_skew"] = float(
                np.mean(((valid - mean) / std) ** 3))

        # NIR / Red ratio (proxy for NDVI direction)
        b08 = list(scene_path.rglob("*_B08.jp2")) or \
              list(scene_path.rglob("*B08*.jp2"))
        if b08:
            with rasterio.open(b08[0]) as src:
                nir = src.read(1).astype(np.float32)
            nir_valid = nir[nir > 0]
            if nir_valid.size > 100 and mean > 0:
                out["inter_band_ratio_nir_red"] = float(
                    nir_valid.mean() / mean)
    except Exception:
        pass
    return out


# ── ESA label lookup ──────────────────────────────────────────────────────────
def load_esa_labels(db_path):
    """Returns dict: scene_name -> (esa_flag, failed_indicator)"""
    if not Path(db_path).exists():
        return {}
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT scene_name, esa_flag, failed_indicator FROM esa_reference"
    ).fetchall()
    conn.close()
    return {r[0]: (r[1], r[2]) for r in rows}


# ── synthetic label lookup (from defects table) ───────────────────────────────
def load_synthetic_labels(db_path):
    """Returns dict: scene_name -> (defect_type, defect_family, severity)"""
    if not Path(db_path).exists():
        return {}
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT scene_name, defect_type, defect_family, severity, base_scene "
            "FROM defects"
        ).fetchall()
    except Exception:
        rows = []
    conn.close()
    return {r[0]: {"defect_type": r[1], "defect_family": r[2],
                   "severity": r[3], "base_scene": r[4]} for r in rows}


# ── assign label ──────────────────────────────────────────────────────────────
def assign_label(scene_name, source, esa_labels, synth_labels):
    """
    Returns (label, base_scene, defect_type, defect_family, severity)
    label: 0=clean, 1=defective
    """
    if source == "synthetic":
        info = synth_labels.get(scene_name, {})
        return (1, info.get("base_scene", scene_name),
                info.get("defect_type"), info.get("defect_family"),
                info.get("severity"))

    if source in ("clean", "esa_ref"):
        esa = esa_labels.get(scene_name)
        if esa:
            flag, indicator = esa
            label = 1 if flag == "FAILED" else 0
            return (label, scene_name, indicator, None, None)
        # clean dir scenes not in ESA table → treat as clean
        if source == "clean":
            return (0, scene_name, None, None, None)

    return (None, scene_name, None, None, None)   # unknown — will be dropped


# ── process one scene ─────────────────────────────────────────────────────────
def process_scene(scene_path, pipeline, source, esa_labels, synth_labels):
    name = scene_path.name
    result = pipeline.run(str(scene_path))

    # Flatten all filter metrics
    flat = {}
    for fname, fres in result["results"].items():
        for k, v in (fres.get("metrics") or {}).items():
            flat[f"{fname}__{k}"] = v

    # Extra band features
    flat.update(compute_extra_features(scene_path))

    label, base_scene, defect_type, defect_family, severity = \
        assign_label(name, source, esa_labels, synth_labels)

    return {
        "scene_name":    name,
        "source":        source,
        "base_scene":    base_scene,
        "label":         label,          # 0=clean, 1=defective
        "defect_type":   defect_type,
        "defect_family": defect_family,
        "severity":      severity,
        "pipeline_passed": result["accepted"],
        "failed_filter": result.get("failed_filter"),
        **flat,
    }


# ── core feature columns (guaranteed to exist after imputation) ───────────────
CORE_FEATURES = [
    "MetadataFilter__cloud_cover",
    "TOAScalingFilter__max_dn",
    "TOAScalingFilter__min_dn",
    "TOAScalingFilter__unique_values",
    "NoDataFilter__unexpected_nodata_ratio",
    "BlurFilter__laplacian_variance",
    "StripeFilter__periodic_power_ratio",
    "NoiseFilter__noise_std_ratio",
    "dn_p05", "dn_p25", "dn_p75", "dn_p95",
    "dn_range", "dn_iqr", "dn_skew",
    "inter_band_ratio_nir_red",
    "NoDataFilter__total_nodata_ratio",    # total zeros including swath edge
    "NoDataFilter__saturated_ratio",  
]


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    REPORT_DIR.mkdir(exist_ok=True)
    pipeline     = build_pipeline()
    esa_labels   = load_esa_labels(DB_PATH)
    synth_labels = load_synthetic_labels(DB_PATH)

    print(f"ESA labels loaded  : {len(esa_labels)} scenes")
    print(f"Synthetic labels   : {len(synth_labels)} scenes\n")

    all_rows = []
    for source, dir_path in SCENE_DIRS.items():
        p = Path(dir_path)
        if not p.exists():
            print(f"  [{source}] directory not found: {dir_path} — skipping")
            continue
        scenes = sorted(p.glob("*.SAFE"))
        print(f"[{source}] {len(scenes)} scenes in {dir_path}")
        for i, scene in enumerate(scenes, 1):
            print(f"  [{i}/{len(scenes)}] {scene.name[:60]}", end=" ... ")
            try:
                row = process_scene(scene, pipeline, source,
                                    esa_labels, synth_labels)
                all_rows.append(row)
                lbl = row["label"]
                print(f"label={lbl}  "
                      f"{'PASS' if row['pipeline_passed'] else 'FAIL'}")
            except Exception as e:
                print(f"ERROR: {e}")

    df = pd.DataFrame(all_rows)

    # Drop rows with unknown label
    before = len(df)
    df = df.dropna(subset=["label"])
    dropped = before - len(df)
    if dropped:
        print(f"\nDropped {dropped} rows with unknown label")

    df["label"] = df["label"].astype(int)

    # Ensure core feature columns exist (fill missing with NaN)
    for col in CORE_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    df.to_csv(OUT_CSV, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"FEATURE TABLE SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows  : {len(df)}")
    print(f"Clean (0)   : {(df['label']==0).sum()}")
    print(f"Defective(1): {(df['label']==1).sum()}")
    print(f"Sources     : {df['source'].value_counts().to_dict()}")
    print(f"Features    : {len([c for c in df.columns if '__' in c or c.startswith('dn_')])}")
    print(f"\nSaved: {OUT_CSV}")
    print("\nNext: python -m src.ml.train_isolation_forest")
    print("      python -m src.ml.train_random_forest")


if __name__ == "__main__":
    main()