# src/ml/radiometric_composite.py
#
# Builds and evaluates a minimal, fully transparent 2-feature composite score
# using ONLY the one statistically validated signal from diagnose_separation.py:
# BlurFilter__dn_range and TOAScalingFilter__max_dn (p=0.001 and p=0.081).
#
# Unlike the 16-feature Random Forest, every number here is traceable:
# the composite is a z-score average computed from YOUR actual PASSED-scene
# mean/std (not estimated or hardcoded), so the threshold is defensible.
#
# Run from repo root (after build_feature_table.py has been run):
#   python -m src.ml.radiometric_composite
#
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

FEATURES_CSV = "reports/ml_features.csv"
REPORT_DIR   = Path("reports/ml")

# The two validated features, in order of effect size from diagnose_separation.py
COMPOSITE_FEATURES = [
    "BlurFilter__dn_range",        # p=0.001, Cohen's d=1.72 (primary signal)
    "TOAScalingFilter__max_dn",    # p=0.081, Cohen's d=0.83 (supporting signal)
]


def load_esa(csv_path):
    df = pd.read_csv(csv_path)
    esa = df[df["source"] == "esa_ref"].dropna(subset=["label"]).copy()
    esa["label"] = esa["label"].astype(int)
    for col in COMPOSITE_FEATURES:
        if col not in esa.columns:
            raise SystemExit(f"Missing required column: {col}")
        esa[col] = pd.to_numeric(esa[col], errors="coerce")
    esa = esa.dropna(subset=COMPOSITE_FEATURES)
    return esa


def check_correlation(esa):
    """
    Sanity check: if the two features are near-perfectly correlated, the
    'composite' isn't adding independent information — say so explicitly.
    """
    corr = esa[COMPOSITE_FEATURES].corr().iloc[0, 1]
    print(f"Correlation between {COMPOSITE_FEATURES[0]} and "
          f"{COMPOSITE_FEATURES[1]}: {corr:.3f}")
    if abs(corr) > 0.85:
        print("  NOTE: highly correlated — composite is effectively measuring")
        print("  ONE underlying signal (DN-distribution width/skew), not two")
        print("  independent ones. Report it as such.")
    print()
    return corr


def compute_composite(esa):
    """
    z-score average using PASSED-scene mean/std as the reference distribution
    (the 'normal' baseline), computed directly from this data — not hardcoded.
    """
    passed = esa[esa["label"] == 0]

    stats = {}
    z_components = []
    for col in COMPOSITE_FEATURES:
        mu  = passed[col].mean()
        sig = passed[col].std()
        if sig == 0 or np.isnan(sig):
            sig = 1.0
        z = (esa[col] - mu) / sig
        z_components.append(z)
        stats[col] = {"passed_mean": float(mu), "passed_std": float(sig)}

    composite = pd.concat(z_components, axis=1).mean(axis=1)
    return composite.values, stats


def evaluate(esa, score):
    y_true = esa["label"].values

    if len(np.unique(y_true)) < 2:
        print("Cannot evaluate — need both classes.")
        return None

    auc = roc_auc_score(y_true, score)

    prec, rec, thresholds = precision_recall_curve(y_true, score)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s)
    best_thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.0

    pred = (score >= best_thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    fpr       = fp / (fp + tn) if (fp + tn) else 0

    print(f"ROC-AUC: {auc:.3f}")
    print(f"Best-F1 threshold (z-score): {best_thr:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision={precision:.2f}  Recall={recall:.2f}  "
          f"F1={f1:.2f}  FPR={fpr:.2f}")

    return {
        "auc": auc, "best_threshold": best_thr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "y_pred": pred, "y_score": score,
    }


def plot(esa, score, metrics_result):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    y_true = esa["label"].values

    # ── Plot 1: scatter of the two raw features ─────────────────────
    ax = axes[0]
    passed = esa[esa["label"] == 0]
    failed = esa[esa["label"] == 1]
    ax.scatter(passed[COMPOSITE_FEATURES[0]], passed[COMPOSITE_FEATURES[1]],
               c="#2ecc71", label=f"PASSED (n={len(passed)})", alpha=0.7, s=50)
    ax.scatter(failed[COMPOSITE_FEATURES[0]], failed[COMPOSITE_FEATURES[1]],
               c="#e74c3c", label=f"FAILED (n={len(failed)})", alpha=0.9, s=70,
               marker="^")
    ax.set_xlabel(COMPOSITE_FEATURES[0])
    ax.set_ylabel(COMPOSITE_FEATURES[1])
    ax.set_title("The two validated features\n(raw values)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Plot 2: composite score distribution ─────────────────────────
    ax = axes[1]
    bins = np.linspace(score.min(), score.max(), 20)
    ax.hist(score[y_true == 0], bins=bins, alpha=0.65, color="#2ecc71",
            label=f"PASSED (n={int((y_true==0).sum())})")
    ax.hist(score[y_true == 1], bins=bins, alpha=0.65, color="#e74c3c",
            label=f"FAILED (n={int((y_true==1).sum())})")
    if metrics_result:
        ax.axvline(metrics_result["best_threshold"], color="navy", ls="--",
                   lw=2, label=f"Threshold={metrics_result['best_threshold']:.2f}")
    ax.set_xlabel("Composite z-score")
    ax.set_ylabel("Count")
    ax.set_title("Radiometric anomaly composite\n(2-feature, transparent)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Plot 3: ROC curve ──────────────────────────────────────────
    ax = axes[2]
    if metrics_result and len(np.unique(y_true)) == 2:
        fpr_c, tpr_c, _ = roc_curve(y_true, score)
        ax.plot(fpr_c, tpr_c, color="#8e44ad", lw=2.5,
                label=f"Composite (AUC={metrics_result['auc']:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC — composite vs random")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Radiometric anomaly composite (dn_range + max_dn)\n"
        "Built from the ONE statistically validated signal in the dataset",
        fontsize=12,
    )
    fig.tight_layout()
    out = REPORT_DIR / "radiometric_composite.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {out}")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(FEATURES_CSV).exists():
        raise SystemExit(
            f"{FEATURES_CSV} not found.\n"
            "Run: python -m src.ml.build_feature_table")

    esa = load_esa(FEATURES_CSV)
    n_failed = int(esa["label"].sum())
    n_passed = len(esa) - n_failed
    print(f"ESA test set: {len(esa)} scenes ({n_failed} FAILED, {n_passed} PASSED)\n")

    corr = check_correlation(esa)

    score, stats = compute_composite(esa)
    print("Composite built from (PASSED-scene reference statistics):")
    for col, s in stats.items():
        print(f"  {col}: mean={s['passed_mean']:.1f}  std={s['passed_std']:.1f}")
    print()

    result = evaluate(esa, score)
    plot(esa, score, result)

    # Per-scene breakdown for the FAILED scenes specifically
    if result:
        print(f"\n{'='*70}")
        print("PER-SCENE: ESA-FAILED scenes")
        print(f"{'='*70}")
        failed_idx = np.where(esa["label"].values == 1)[0]
        for idx in failed_idx:
            row = esa.iloc[idx]
            caught = "CATCH" if result["y_pred"][idx] == 1 else "MISS"
            dtype = row.get("defect_type", "-")
            print(f"  {row['scene_name'][:55]:<55} "
                  f"{dtype:<20} z={score[idx]:>6.2f}  {caught}")

    # Save results
    out_csv = REPORT_DIR / "radiometric_composite_results.csv"
    esa_out = esa[["scene_name", "label"] + COMPOSITE_FEATURES].copy()
    esa_out["composite_score"] = score
    if result:
        esa_out["predicted"] = result["y_pred"]
    esa_out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    if result:
        with open(REPORT_DIR / "radiometric_composite_metrics.json", "w") as f:
            json.dump({
                "auc": result["auc"], "f1": result["f1"],
                "precision": result["precision"], "recall": result["recall"],
                "fpr": result["fpr"], "best_threshold": result["best_threshold"],
                "tp": result["tp"], "fp": result["fp"],
                "fn": result["fn"], "tn": result["tn"],
                "feature_correlation": float(corr),
                "reference_stats": stats,
            }, f, indent=2)
        print(f"Saved: {REPORT_DIR}/radiometric_composite_metrics.json")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print("This composite uses ONLY 2 features validated by Mann-Whitney U test")
    print("(p<0.10) — it is intentionally simple and fully traceable, unlike the")
    print("16-feature Random Forest which trains on mostly non-significant signal.")
    if abs(corr) > 0.85:
        print("The two features are highly correlated, so this is effectively a")
        print("single radiometric-width signal expressed two ways, not two")
        print("independent detectors.")
    print("Compare this AUC/F1 directly against final_comparison.py's table.")


if __name__ == "__main__":
    main()