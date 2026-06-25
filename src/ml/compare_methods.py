# src/ml/compare_methods.py
#
# Final comparison: rule-based pipeline vs Isolation Forest vs Random Forest.
# Reads metrics saved by each training script and the confusion matrix,
# produces a unified comparison table and figure for the thesis.
#
# Run from repo root (after all three training scripts):
#   python -m src.ml.compare_methods
#
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

REPORT_DIR   = Path("reports/ml")
FEATURES_CSV = "reports/ml_features.csv"
CM_CSV       = "reports/confusion_matrix_results.csv"
DB_PATH      = "reports/eo_qc.db"

FEATURE_COLS = [
    "MetadataFilter__cloud_cover",
    "TOAScalingFilter__max_dn",
    "TOAScalingFilter__min_dn",
    "TOAScalingFilter__unique_values",
    "NoDataFilter__nodata_ratio",
    "BlurFilter__laplacian_variance",
    "StripeFilter__periodic_power_ratio",
    "NoiseFilter__noise_std_ratio",
    "dn_p05", "dn_p25", "dn_p75", "dn_p95",
    "dn_range", "dn_iqr", "dn_skew",
    "inter_band_ratio_nir_red",
]


# ── load saved metrics ────────────────────────────────────────────────────────
def load_rule_based():
    """Derive rule-based metrics from confusion_matrix_results.csv."""
    if not Path(CM_CSV).exists():
        print(f"  Warning: {CM_CSV} not found — run confusion_matrix.py first")
        return None

    df = pd.read_csv(CM_CSV)
    if "esa_defective" not in df.columns or "pipeline_passed" not in df.columns:
        return None

    df["pipeline_rejected"] = ~df["pipeline_passed"].astype(bool)
    df["esa_defective"]     = df["esa_defective"].astype(bool)

    tp = int(( df["esa_defective"] &  df["pipeline_rejected"]).sum())
    fn = int(( df["esa_defective"] & ~df["pipeline_rejected"]).sum())
    fp = int((~df["esa_defective"] &  df["pipeline_rejected"]).sum())
    tn = int((~df["esa_defective"] & ~df["pipeline_rejected"]).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    fpr       = fp / (fp + tn) if (fp + tn) else 0

    return {
        "method": "Rule-based (7 filters)",
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr,
        "roc_auc": None,   # no probability score → no AUC
    }


def load_json_metrics(path, method_name):
    if not Path(path).exists():
        print(f"  Warning: {path} not found")
        return None
    with open(path) as f:
        m = json.load(f)
    m["method"] = method_name
    return m


# ── re-score ESA scenes with saved models ────────────────────────────────────
def rescore_esa(features_csv):
    """
    Re-run IF and RF on ESA scenes to get probability scores for ROC plot.
    Returns dict of {method_name: (y_true, scores)}
    """
    results = {}
    if not Path(features_csv).exists():
        return results

    df = pd.read_csv(features_csv)
    esa = df[df["source"] == "esa_ref"].dropna(subset=["label"])
    if esa.empty:
        return results

    y = esa["label"].astype(int).values

    for method, model_path, scaler_path, cols_path in [
        ("Isolation Forest",
         REPORT_DIR / "isolation_forest.pkl",
         REPORT_DIR / "if_scaler.pkl",
         REPORT_DIR / "if_feature_cols.json"),
        ("Random Forest",
         REPORT_DIR / "random_forest.pkl",
         REPORT_DIR / "rf_scaler.pkl",
         REPORT_DIR / "rf_feature_cols.json"),
    ]:
        if not all(p.exists() for p in [model_path, scaler_path, cols_path]):
            continue

        with open(cols_path) as f:
            model_feat_cols = json.load(f)

        missing_cols = set(model_feat_cols) - set(esa.columns)
        if missing_cols:
            print(f"  Warning: {cols_path.name} contains columns not found in ESA data: {sorted(missing_cols)}")
            continue

        X_model = esa[model_feat_cols].copy()
        for col in model_feat_cols:
            median = X_model[col].median()
            if np.isnan(median):
                median = 0.0
            X_model[col] = X_model[col].fillna(median)

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        X_scaled = scaler.transform(X_model.values)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_scaled)[:, 1]
        else:
            # Isolation Forest: negate decision_function
            scores = -model.decision_function(X_scaled)
            # Normalise to [0,1] for fair comparison
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        results[method] = (y, scores)

    return results


# ── comparison table ──────────────────────────────────────────────────────────
def print_comparison_table(methods):
    print(f"\n{'='*80}")
    print("METHOD COMPARISON — ESA T32SPF test set")
    print(f"{'='*80}")
    cols = ["Method", "TP", "FP", "FN", "TN",
            "Precision", "Recall", "F1", "FPR", "ROC-AUC"]
    header = (f"{'Method':<30} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
              f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6} {'AUC':>7}")
    print(header)
    print("-" * 80)
    for m in methods:
        if m is None:
            continue
        auc = f"{m['roc_auc']:.3f}" if m.get("roc_auc") else "  N/A "
        print(f"{m['method']:<30} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4} "
              f"{m['precision']:>6.2f} {m['recall']:>6.2f} "
              f"{m['f1']:>6.2f} {m['fpr']:>6.2f} {auc:>7}")
    print(f"{'='*80}")


# ── comparison figure ─────────────────────────────────────────────────────────
def plot_comparison(methods, roc_scores):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    valid = [m for m in methods if m is not None]
    colors = ["#7f8c8d", "#27ae60", "#c0392b"][:len(valid)]

    # ── Plot 1: bar chart of key metrics ─────────────────────────────
    ax = axes[0]
    metric_names = ["Recall", "Precision", "F1", "FPR"]
    x = np.arange(len(metric_names))
    width = 0.25
    for i, (m, color) in enumerate(zip(valid, colors)):
        vals = [m["recall"], m["precision"], m["f1"], m["fpr"]]
        bars = ax.bar(x + i * width, vals, width, label=m["method"],
                      color=color, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    # Lower FPR is better — annotate
    ax.annotate("↓ lower is better", xy=(3, 0.05),
                fontsize=7, color="gray", ha="center")

    # ── Plot 2: ROC curves ────────────────────────────────────────────
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")

    roc_colors = {"Isolation Forest": "#27ae60", "Random Forest": "#c0392b"}
    for method_name, (y_true, scores) in roc_scores.items():
        if len(np.unique(y_true)) < 2:
            continue
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, lw=2.5,
                color=roc_colors.get(method_name, "gray"),
                label=f"{method_name} (AUC={auc:.2f})")

    # Rule-based operating point (single point, no threshold sweep)
    rb = next((m for m in valid if "Rule" in m["method"]), None)
    if rb:
        ax.scatter([rb["fpr"]], [rb["recall"]], s=150, zorder=6,
                   color="#7f8c8d", marker="D",
                   label=f"Rule-based (F1={rb['f1']:.2f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — ESA test set")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Plot 3: confusion matrix grid ────────────────────────────────
    ax = axes[2]
    ax.axis("off")
    table_data = [["Method", "TP", "FP", "FN", "TN", "F1"]]
    for m in valid:
        table_data.append([
            m["method"].replace(" (7 filters)", ""),
            str(m["tp"]), str(m["fp"]),
            str(m["fn"]), str(m["tn"]),
            f"{m['f1']:.2f}",
        ])
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center",
                   bbox=[0, 0.2, 1, 0.6])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    ax.set_title("Confusion Matrix Summary", pad=60)

    fig.suptitle(
        "EO Quality Control: Rule-based vs ML Methods\n"
        f"Sentinel-2 L1C Tile T32SPF",
        fontsize=13,
    )
    fig.tight_layout()
    out = REPORT_DIR / "method_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison plot: {out}")


# ── thesis summary ────────────────────────────────────────────────────────────
def print_thesis_summary(methods, roc_scores):
    print(f"\n{'='*70}")
    print("THESIS NARRATIVE SUMMARY")
    print(f"{'='*70}")
    rb  = next((m for m in methods if m and "Rule" in m["method"]), None)
    ifo = next((m for m in methods if m and "Isolation" in m["method"]), None)
    rf  = next((m for m in methods if m and "Random" in m["method"]), None)

    if rb:
        print(f"\nRule-based pipeline (7 hand-tuned filters):")
        print(f"  Recall={rb['recall']:.0%}  FPR={rb['fpr']:.0%}  F1={rb['f1']:.2f}")
        print(f"  Correctly rejected {rb['tp']}/{rb['tp']+rb['fn']} "
              f"ESA-FAILED scenes.")
        if rb["fp"] > 0:
            print(f"  Wrongly rejected {rb['fp']} clean scenes (false alarms).")

    if ifo:
        print(f"\nIsolation Forest (one-class, trained on clean scenes only):")
        print(f"  Recall={ifo['recall']:.0%}  FPR={ifo['fpr']:.0%}  "
              f"F1={ifo['f1']:.2f}  AUC={ifo.get('roc_auc',0):.2f}")
        print(f"  Requires NO labeled defective examples.")

    if rf:
        print(f"\nRandom Forest (supervised, trained on synthetic + ESA labels):")
        print(f"  Recall={rf['recall']:.0%}  FPR={rf['fpr']:.0%}  "
              f"F1={rf['f1']:.2f}  AUC={rf.get('roc_auc',0):.2f}")
        print(f"  Learned joint decision boundary across {len(FEATURE_COLS)} features.")

    print(f"\nConclusion:")
    best = max([m for m in [rb, ifo, rf] if m],
               key=lambda m: m["f1"], default=None)
    if best:
        print(f"  Best F1 achieved by: {best['method']} (F1={best['f1']:.2f})")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rb  = load_rule_based()
    ifo = load_json_metrics(REPORT_DIR / "if_metrics.json", "Isolation Forest")
    rf  = load_json_metrics(REPORT_DIR / "rf_metrics.json", "Random Forest")

    methods = [m for m in [rb, ifo, rf] if m is not None]
    if not methods:
        raise SystemExit(
            "No metrics found. Run in order:\n"
            "  1. python -m src.real_world.confusion_matrix\n"
            "  2. python -m src.ml.build_feature_table\n"
            "  3. python -m src.ml.train_isolation_forest\n"
            "  4. python -m src.ml.train_random_forest\n"
            "  5. python -m src.ml.compare_methods")

    print_comparison_table(methods)

    roc_scores = rescore_esa(FEATURES_CSV)
    plot_comparison(methods, roc_scores)
    print_thesis_summary(methods, roc_scores)

    # Save comparison CSV
    out_csv = REPORT_DIR / "method_comparison.csv"
    pd.DataFrame(methods).to_csv(out_csv, index=False)
    print(f"\nSaved comparison table: {out_csv}")


if __name__ == "__main__":
    main()