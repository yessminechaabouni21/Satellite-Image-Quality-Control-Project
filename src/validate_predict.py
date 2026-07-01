# src/validate_predict.py
#
# Final sanity check: runs the actual predict() deliverable against every
# downloaded ESA scene with known ground truth, and reports a confusion
# matrix. This proves the final wrapped function (rule-based + radiometric
# composite) behaves consistently with the standalone evaluation you already
# did in final_comparison.py — it's the same logic, but exercised through
# the single callable interface someone else would actually use.
#
# Run from repo root:
#   python -m src.validate_predict
#
import json
from pathlib import Path

import pandas as pd

from src.predict import predict

FEATURES_CSV = "reports/ml_features.csv"
SCENE_DIR    = "data/esa_reference"

OUT_CSV      = "reports/predict_validation.csv"


def main():
    if not Path(FEATURES_CSV).exists():
        raise SystemExit(f"{FEATURES_CSV} not found.")

    df  = pd.read_csv(FEATURES_CSV)
    esa = df[df["source"] == "esa_ref"].dropna(subset=["label"]).copy()
    esa["label"] = esa["label"].astype(int)

    print(f"Found {len(esa)} ESA scenes with ground truth in {FEATURES_CSV}\n")

    rows = []
    skipped = 0

    for i, (_, row) in enumerate(esa.iterrows(), 1):
        scene_path = Path(SCENE_DIR) / row["scene_name"]
        if not scene_path.exists():
            skipped += 1
            continue

        print(f"[{i}/{len(esa)}] {row['scene_name'][:55]}", end=" ... ")
        try:
            result = predict(str(scene_path))
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        esa_failed       = bool(row["label"] == 1)
        predicted_reject = not result["accepted"]
        correct = esa_failed == predicted_reject

        print(f"{'REJECT' if predicted_reject else 'ACCEPT':7} "
              f"({'correct' if correct else 'WRONG'})")

        rows.append({
            "scene_name": row["scene_name"],
            "esa_flag": "FAILED" if esa_failed else "PASSED",
            "defect_type": row.get("defect_type"),
            "predicted": "REJECTED" if predicted_reject else "ACCEPTED",
            "correct": correct,
            "reason": result["reason"],
            "rule_based_accepted": result["rule_based"]["accepted"] if result["rule_based"] else None,
            "composite_z_score": result["radiometric_composite"]["z_score"] if result["radiometric_composite"] else None,
            "composite_flagged": result["radiometric_composite"]["flagged"] if result["radiometric_composite"] else None,
        })

    if skipped:
        print(f"\n{skipped} scenes skipped (not found in {SCENE_DIR} — "
              f"only scenes downloaded to disk can be validated)")

    if not rows:
        raise SystemExit(
            f"\nNo scenes found in {SCENE_DIR}. "
            "Download scenes first (see download_failed_scenes.py / "
            "download_passed_sample.py).")

    out = pd.DataFrame(rows)

    # ── confusion matrix ──────────────────────────────────────────────
    tp = int(((out["esa_flag"] == "FAILED") & (out["predicted"] == "REJECTED")).sum())
    fn = int(((out["esa_flag"] == "FAILED") & (out["predicted"] == "ACCEPTED")).sum())
    fp = int(((out["esa_flag"] == "PASSED") & (out["predicted"] == "REJECTED")).sum())
    tn = int(((out["esa_flag"] == "PASSED") & (out["predicted"] == "ACCEPTED")).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    fpr       = fp / (fp + tn) if (fp + tn) else 0

    print(f"\n{'='*60}")
    print(f"FINAL DELIVERABLE VALIDATION — predict() on {len(out)} real scenes")
    print(f"{'='*60}")
    print(f"                    ESA FAILED   ESA PASSED")
    print(f"  predict() REJECT      {tp:^6}       {fp:^6}")
    print(f"  predict() ACCEPT      {fn:^6}       {tn:^6}")
    print()
    print(f"  Precision: {precision:.2f}   Recall: {recall:.2f}   "
          f"F1: {f1:.2f}   FPR: {fpr:.2f}")

    # Breakdown of which layer caught what
    print(f"\n{'='*60}")
    print("WHICH LAYER CAUGHT EACH ESA-FAILED SCENE")
    print(f"{'='*60}")
    failed = out[out["esa_flag"] == "FAILED"]
    for _, r in failed.iterrows():
        layer = []
        if r["rule_based_accepted"] is False:
            layer.append("rule-based")
        if r["composite_flagged"]:
            layer.append("composite")
        layer_str = " + ".join(layer) if layer else "MISSED by both"
        print(f"  {r['scene_name'][:50]:<50} {r['defect_type'] or '-':<20} {layer_str}")

    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()