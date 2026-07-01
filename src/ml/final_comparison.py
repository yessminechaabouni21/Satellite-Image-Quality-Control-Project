# src/ml/final_comparison.py
#
# Honest, consistent comparison of three methods on the SAME 48 ESA scenes.
# All results come from ml_features.csv — no hardcoded numbers, no proxies.
#
# Rule-based  : pipeline_passed column  (actual pipeline output, already stored)
# IF          : isolation_forest.pkl    (predict on feature cols)
# RF          : random_forest.pkl       (predict on feature cols)
# Hybrid      : rule-based Stage 1 → IF Stage 2 on scenes that passed Stage 1
#
# Ground truth: label column (0 = ESA PASSED, 1 = ESA FAILED)
# Test set    : source == "esa_ref" rows only (real ESA scenes, no synthetic)
#
# Run from repo root:
#   python -m src.ml.final_comparison
#
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

FEATURES_CSV = "reports/ml_features.csv"
REPORT_DIR   = Path("reports/ml")

# Must match FEATURE_COLS used during training exactly
FEATURE_COLS = [
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
]


# ── load & prepare ────────────────────────────────────────────────────────────
def load_esa_test_set(csv_path):
    """
    Returns df, X, y for ESA scenes only.
    Rule-based predictions already stored in pipeline_passed column.
    """
    df = pd.read_csv(csv_path)

    # ESA scenes only — these are the held-out real-world test set
    esa = df[df["source"] == "esa_ref"].copy()
    esa = esa.dropna(subset=["label"])
    esa["label"] = esa["label"].astype(int)

    if esa.empty:
        raise SystemExit(
            "No ESA scenes found in feature table.\n"
            "Run: python -m src.ml.build_feature_table")

    # Feature matrix — use only cols that exist, median-impute NaNs
    feat_cols = [c for c in FEATURE_COLS if c in esa.columns]
    missing   = set(FEATURE_COLS) - set(feat_cols)
    if missing:
        print(f"  Note: {len(missing)} feature cols missing, skipping: {missing}")

    X = esa[feat_cols].copy()
    for col in feat_cols:
        X[col] = X[col].fillna(X[col].median())

    y = esa["label"].values   # ground truth: 1=FAILED, 0=PASSED

    print(f"Test set: {len(esa)} ESA scenes  "
          f"({int(y.sum())} FAILED, {int((1-y).sum())} PASSED)")

    return esa, X.values, y, feat_cols


def load_model(path):
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── metric helpers ────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, y_score=None):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    fpr       = fp / (fp + tn) if (fp + tn) else 0.0

    auc = None
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            pass

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": f1, "fpr": fpr, "auc": auc,
            "y_pred": y_pred, "y_score": y_score}


# ── evaluate each method ──────────────────────────────────────────────────────
def eval_rule_based(esa_df, y_true):
    """Read pipeline_passed column — actual pipeline decision, no proxy."""
    if "pipeline_passed" not in esa_df.columns:
        raise SystemExit(
            "pipeline_passed column missing from ml_features.csv.\n"
            "Rebuild: python -m src.ml.build_feature_table")

    # pipeline_passed=True means ACCEPTED (not defective)
    # so pipeline_rejected = NOT pipeline_passed = predicted defective
    y_pred = (~esa_df["pipeline_passed"].astype(bool)).astype(int).values
    return metrics(y_true, y_pred)


def eval_isolation_forest(X, y_true):
    model  = load_model(REPORT_DIR / "isolation_forest.pkl")
    scaler = load_model(REPORT_DIR / "if_scaler.pkl")
    if model is None:
        print("  Isolation Forest model not found — skipping")
        return None

    X_scaled = scaler.transform(X) if scaler else X
    # predict: -1=anomaly → 1=defective, +1=normal → 0=clean
    y_pred   = (model.predict(X_scaled) == -1).astype(int)
    y_score  = -model.decision_function(X_scaled)   # higher = more anomalous
    return metrics(y_true, y_pred, y_score)


def eval_random_forest(X, y_true):
    model  = load_model(REPORT_DIR / "random_forest.pkl")
    scaler = load_model(REPORT_DIR / "rf_scaler.pkl")
    if model is None:
        print("  Random Forest model not found — skipping")
        return None

    X_scaled = scaler.transform(X) if scaler else X
    y_pred   = model.predict(X_scaled)
    y_score  = model.predict_proba(X_scaled)[:, 1]
    return metrics(y_true, y_pred, y_score)


def eval_hybrid(esa_df, X, y_true, if_model, if_scaler):
    """
    Stage 1: use pipeline_passed (rule-based decision).
    Stage 2: run IF only on scenes that PASSED Stage 1.
    Final decision: REJECT if either stage rejects.
    """
    if if_model is None:
        return None

    rule_reject = (~esa_df["pipeline_passed"].astype(bool)).values

    # Stage 2: score only the scenes that passed Stage 1
    passed_mask  = ~rule_reject
    y_pred_final = rule_reject.copy().astype(int)
    y_score_final = np.where(rule_reject, 1.0, 0.0).astype(float)

    if passed_mask.sum() > 0:
        X_pass   = X[passed_mask]
        X_scaled = if_scaler.transform(X_pass) if if_scaler else X_pass
        if_pred  = (if_model.predict(X_scaled) == -1).astype(int)
        if_score = -if_model.decision_function(X_scaled)

        y_pred_final[passed_mask]  = if_pred
        y_score_final[passed_mask] = if_score

    return metrics(y_true, y_pred_final, y_score_final)


# ── print & plot ──────────────────────────────────────────────────────────────
def print_table(results):
    print(f"\n{'='*78}")
    print("FINAL COMPARISON — same 48 ESA scenes, same ground truth for all methods")
    print(f"{'='*78}")
    print(f"{'Method':<22} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
          f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6} {'AUC':>7}")
    print("-" * 78)
    for name, m in results.items():
        if m is None:
            continue
        auc = f"{m['auc']:.3f}" if m["auc"] is not None else "  N/A "
        print(f"{name:<22} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4} "
              f"{m['precision']:>6.2f} {m['recall']:>6.2f} "
              f"{m['f1']:>6.2f} {m['fpr']:>6.2f} {auc:>7}")
    print(f"{'='*78}")


def print_per_scene(esa_df, y_true, results):
    """Show per-scene decisions for the 8 FAILED scenes."""
    print(f"\n{'='*78}")
    print("PER-SCENE BREAKDOWN — ESA-FAILED scenes only")
    print(f"{'='*78}")
    failed_idx = np.where(y_true == 1)[0]
    header = f"{'Scene':<52} {'ESA indicator':<22}"
    for name in results:
        header += f" {name[:8]:>8}"
    print(header)
    print("-" * 78)
    for idx in failed_idx:
        row   = esa_df.iloc[idx]
        scene = row["scene_name"][:50]
        ind   = str(row.get("failed_indicator", "-"))[:20]
        line  = f"{scene:<52} {ind:<22}"
        for name, m in results.items():
            if m is None:
                line += f" {'N/A':>8}"
            else:
                pred = m["y_pred"][idx]
                line += f" {'CATCH' if pred==1 else 'MISS':>8}"
        print(line)


def print_fp_scenes(esa_df, y_true, results):
    """Show which clean scenes are wrongly rejected (false positives)."""
    print(f"\n{'='*78}")
    print("FALSE POSITIVES — clean ESA-PASSED scenes wrongly rejected")
    print(f"{'='*78}")
    clean_idx = np.where(y_true == 0)[0]
    any_fp = False
    for name, m in results.items():
        if m is None:
            continue
        fp_idx = [i for i in clean_idx if m["y_pred"][i] == 1]
        if fp_idx:
            any_fp = True
            print(f"\n  {name} ({len(fp_idx)} false positives):")
            for i in fp_idx:
                row = esa_df.iloc[i]
                ff  = row.get("failed_filter", "-")
                print(f"    {row['scene_name'][:55]}  "
                      f"filter={ff}")
    if not any_fp:
        print("  None — all methods reject no clean scenes.")


def plot_comparison(results, y_true):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {
        "Rule-based":       "#7f8c8d",
        "Isolation Forest": "#27ae60",
        "Random Forest":    "#c0392b",
        "Hybrid":           "#2980b9",
    }

    # ── Plot 1: key metrics bar chart ─────────────────────────────────
    ax = axes[0]
    valid   = {k: v for k, v in results.items() if v is not None}
    metrics_names = ["Recall", "Precision", "F1", "FPR"]
    x     = np.arange(len(metrics_names))
    width = 0.8 / len(valid)
    for i, (name, m) in enumerate(valid.items()):
        vals = [m["recall"], m["precision"], m["f1"], m["fpr"]]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=name, color=colors.get(name, "gray"), alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Method comparison\n(all on same 48 ESA scenes)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.annotate("↓ lower FPR is better", xy=(3.0, 0.04),
                fontsize=7, color="#e74c3c", ha="center")

    # ── Plot 2: ROC curves for methods with probability scores ─────────
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (AUC=0.50)")
    for name, m in valid.items():
        if m["y_score"] is not None and m["auc"] is not None:
            fpr_c, tpr_c, _ = roc_curve(y_true, m["y_score"])
            ax.plot(fpr_c, tpr_c, lw=2.5, color=colors.get(name, "gray"),
                    label=f"{name} (AUC={m['auc']:.2f})")
        else:
            # Rule-based: single operating point
            ax.scatter([m["fpr"]], [m["recall"]], s=160, marker="D",
                       color=colors.get(name, "gray"), zorder=6,
                       label=f"{name} (F1={m['f1']:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC curves\n(ESA test set)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Plot 3: confusion matrix grid ─────────────────────────────────
    ax = axes[2]
    ax.axis("off")
    rows = [["Method", "TP", "FP", "FN", "TN", "F1", "AUC"]]
    for name, m in valid.items():
        auc_s = f"{m['auc']:.2f}" if m["auc"] is not None else "N/A"
        rows.append([name, str(m["tp"]), str(m["fp"]),
                     str(m["fn"]), str(m["tn"]),
                     f"{m['f1']:.2f}", auc_s])
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc="center", loc="center",
                   bbox=[0, 0.15, 1, 0.7])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    ax.set_title("Confusion matrix summary", pad=55)

    fig.suptitle(
        "EO QC — Rule-based vs ML vs Hybrid\n"
        "Sentinel-2 L1C T32SPF  |  48 real ESA scenes  |  same ground truth",
        fontsize=12,
    )
    fig.tight_layout()
    out = REPORT_DIR / "final_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {out}")


def print_thesis_summary(results, n_esa, n_failed, n_passed):
    print(f"\n{'='*78}")
    print("THESIS NARRATIVE SUMMARY")
    print(f"{'='*78}")
    print(f"Evaluated on {n_esa} real Sentinel-2 L1C scenes (tile T32SPF, 2024-2026)")
    print(f"Ground truth: ESA OLQC flags  ({n_failed} FAILED, {n_passed} PASSED)\n")

    order = ["Rule-based", "Isolation Forest", "Random Forest", "Hybrid"]
    for name in order:
        m = results.get(name)
        if m is None:
            continue
        auc_s = f"  AUC={m['auc']:.2f}" if m["auc"] else ""
        print(f"{name}:")
        print(f"  Recall={m['recall']:.0%}  FPR={m['fpr']:.0%}  "
              f"F1={m['f1']:.2f}{auc_s}")
        print(f"  Caught {m['tp']}/{m['tp']+m['fn']} ESA-FAILED  |  "
              f"{m['fp']} false alarms on clean scenes")
        print()

    # Best by F1
    best_name = max(
        (k for k, v in results.items() if v),
        key=lambda k: results[k]["f1"]
    )
    best = results[best_name]
    print(f"Best F1: {best_name}  (F1={best['f1']:.2f}, "
          f"Recall={best['recall']:.0%}, FPR={best['fpr']:.0%})")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(FEATURES_CSV).exists():
        raise SystemExit(
            f"Feature table not found: {FEATURES_CSV}\n"
            "Run: python -m src.ml.build_feature_table")

    print(f"Loading: {FEATURES_CSV}")
    esa_df, X, y_true, feat_cols = load_esa_test_set(FEATURES_CSV)

    n_esa    = len(esa_df)
    n_failed = int(y_true.sum())
    n_passed = int((1 - y_true).sum())

    # Load IF model once (used by both IF eval and Hybrid eval)
    if_model  = load_model(REPORT_DIR / "isolation_forest.pkl")
    if_scaler = load_model(REPORT_DIR / "if_scaler.pkl")

    # ── evaluate all methods on identical test set ─────────────────────
    print("\nEvaluating...")
    results = {}

    print("  Rule-based  (from pipeline_passed column)")
    results["Rule-based"] = eval_rule_based(esa_df, y_true)

    print("  Isolation Forest")
    results["Isolation Forest"] = eval_isolation_forest(X, y_true)

    print("  Random Forest")
    results["Random Forest"] = eval_random_forest(X, y_true)

    print("  Hybrid  (Rule Stage 1 → IF Stage 2)")
    results["Hybrid"] = eval_hybrid(esa_df, X, y_true, if_model, if_scaler)

    # ── print results ─────────────────────────────────────────────────
    print_table(results)
    print_per_scene(esa_df, y_true, results)
    print_fp_scenes(esa_df, y_true, results)
    print_thesis_summary(results, n_esa, n_failed, n_passed)

    # ── plot ──────────────────────────────────────────────────────────
    plot_comparison(results, y_true)

    # ── save CSV ──────────────────────────────────────────────────────
    rows = []
    for name, m in results.items():
        if m is None:
            continue
        rows.append({
            "method": name,
            "tp": m["tp"], "fp": m["fp"],
            "fn": m["fn"], "tn": m["tn"],
            "precision": round(m["precision"], 3),
            "recall":    round(m["recall"],    3),
            "f1":        round(m["f1"],        3),
            "fpr":       round(m["fpr"],       3),
            "auc":       round(m["auc"], 3) if m["auc"] else None,
        })
    out_csv = REPORT_DIR / "final_comparison.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv}")

    print(f"\nNote: all four methods evaluated on the identical {n_esa} ESA scenes.")
    print("Rule-based result comes from the actual pipeline_passed column,")
    print("not from hardcoded numbers or proxies.")

# Voting ensemble: reject if ANY method flags the scene
    if all(results[m] is not None for m in ["Rule-based","Isolation Forest","Random Forest"]):
        rb_pred = results["Rule-based"]["y_pred"]
        if_pred = results["Isolation Forest"]["y_pred"]
        rf_pred = results["Random Forest"]["y_pred"]
        ensemble_pred = ((rb_pred + if_pred + rf_pred) >= 1).astype(int)  # ANY votes reject
        results["Ensemble (any)"] = metrics(y_true, ensemble_pred)
if __name__ == "__main__":
    main()