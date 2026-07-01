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
    Returns esa_df, y for ESA scenes only.
    Rule-based predictions already stored in pipeline_passed column.
    Feature selection now happens per-model in build_X_for_model(),
    using each model's own saved feature_cols.json — this avoids the
    'X has N features but scaler expects M' crash when different models
    were trained with different feature subsets (e.g. --source esa_ref runs
    that dropped a column not present in that smaller dataset).
    """
    df = pd.read_csv(csv_path)

    esa = df[df["source"] == "esa_ref"].copy()
    esa = esa.dropna(subset=["label"])
    esa["label"] = esa["label"].astype(int)

    if esa.empty:
        raise SystemExit(
            "No ESA scenes found in feature table.\n"
            "Run: python -m src.ml.build_feature_table")

    y = esa["label"].values   # ground truth: 1=FAILED, 0=PASSED

    print(f"Test set: {len(esa)} ESA scenes  "
          f"({int(y.sum())} FAILED, {int((1-y).sum())} PASSED)")

    return esa, y


def load_feature_cols(json_path, fallback_cols, esa_df):
    """Load the exact feature column list/order a model was trained with.
    Falls back to FEATURE_COLS (intersected with available columns) if the
    json file isn't present — keeps backward compatibility with older runs.
    """
    if Path(json_path).exists():
        with open(json_path) as f:
            cols = json.load(f)
        missing = [c for c in cols if c not in esa_df.columns]
        if missing:
            print(f"  Warning: {len(missing)} trained feature cols missing "
                  f"from feature table: {missing}")
        return [c for c in cols if c in esa_df.columns]
    return [c for c in fallback_cols if c in esa_df.columns]


def build_X_for_model(esa_df, feat_cols):
    """Build the feature matrix for one model using exactly its own columns."""
    X = esa_df[feat_cols].copy()
    for col in feat_cols:
        X[col] = X[col].fillna(X[col].median())
    return X.values


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


def eval_isolation_forest(esa_df, y_true, suffix=""):
    model  = load_model(REPORT_DIR / f"isolation_forest{suffix}.pkl")
    scaler = load_model(REPORT_DIR / f"if_scaler{suffix}.pkl")
    if model is None:
        print(f"  Isolation Forest model not found "
              f"(reports/ml/isolation_forest{suffix}.pkl) — skipping")
        return None

    feat_cols = load_feature_cols(
        REPORT_DIR / f"if_feature_cols{suffix}.json", FEATURE_COLS, esa_df)
    X = build_X_for_model(esa_df, feat_cols)

    X_scaled = scaler.transform(X) if scaler else X
    # predict: -1=anomaly → 1=defective, +1=normal → 0=clean
    y_pred   = (model.predict(X_scaled) == -1).astype(int)
    y_score  = -model.decision_function(X_scaled)   # higher = more anomalous
    return metrics(y_true, y_pred, y_score)


def eval_random_forest(esa_df, y_true, suffix=""):
    """
    Prefer the honest LOO out-of-fold predictions (rf_loo_predictions.csv)
    when available — these come from train_random_forest.py --source esa_ref
    mode, where each scene was scored by a model that NEVER saw it during
    training. Re-scoring the final saved random_forest.pkl directly would
    leak, because in ESA-only mode that file is retrained on ALL 48 scenes
    (including the ones being evaluated here) as its last step.

    Falls back to re-scoring the saved model only if no LOO file exists —
    safe in that case because default training mode keeps train/test disjoint
    (synthetic+clean for training, esa_ref held out for testing).
    """
    loo_path = REPORT_DIR / "rf_loo_predictions.csv"

    if loo_path.exists():
        print("  (using LOO out-of-fold predictions — no leakage)")
        loo_df = pd.read_csv(loo_path)

        merged = esa_df[["scene_name"]].merge(
            loo_df[["scene_name", "rf_loo_proba"]],
            on="scene_name", how="left"
        )
        n_missing = merged["rf_loo_proba"].isna().sum()
        if n_missing:
            print(f"  Warning: {n_missing}/{len(merged)} scenes missing from "
                  f"LOO predictions (filled with 0 — check scene_name match)")

        y_score = merged["rf_loo_proba"].fillna(0).values
        y_pred  = (y_score >= 0.5).astype(int)
        return metrics(y_true, y_pred, y_score)

    # Fallback: no LOO file — re-score the saved model directly
    print("  (no rf_loo_predictions.csv found — scoring saved model directly;"
          " only valid if train/test were disjoint)")
    model  = load_model(REPORT_DIR / f"random_forest{suffix}.pkl")
    scaler = load_model(REPORT_DIR / f"rf_scaler{suffix}.pkl")
    if model is None:
        print("  Random Forest model not found — skipping")
        return None

    feat_cols = load_feature_cols(
        REPORT_DIR / f"rf_feature_cols{suffix}.json", FEATURE_COLS, esa_df)
    X = build_X_for_model(esa_df, feat_cols)

    X_scaled = scaler.transform(X) if scaler else X
    y_pred   = model.predict(X_scaled)
    y_score  = model.predict_proba(X_scaled)[:, 1]
    return metrics(y_true, y_pred, y_score)


def eval_hybrid(esa_df, y_true, if_model, if_scaler, suffix=""):
    """
    Stage 1: use pipeline_passed (rule-based decision).
    Stage 2: run IF only on scenes that PASSED Stage 1.
    Final decision: REJECT if either stage rejects.
    """
    if if_model is None:
        return None

    feat_cols = load_feature_cols(
        REPORT_DIR / f"if_feature_cols{suffix}.json", FEATURE_COLS, esa_df)
    X = build_X_for_model(esa_df, feat_cols)

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
        ind   = str(row.get("defect_type", row.get("failed_indicator", "-")))[:20]
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["all", "validated"], default="all",
                    help="Which trained model variant to load: "
                        "'all' loads isolation_forest.pkl/random_forest.pkl "
                        "(no suffix), 'validated' loads the _validated.pkl "
                        "versions trained on only the statistically "
                        "significant features.")
    args = ap.parse_args()
    suffix = "_validated" if args.features == "validated" else ""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(FEATURES_CSV).exists():
        raise SystemExit(
            f"Feature table not found: {FEATURES_CSV}\n"
            "Run: python -m src.ml.build_feature_table")

    print(f"Loading: {FEATURES_CSV}  (model variant: {args.features})")
    esa_df, y_true = load_esa_test_set(FEATURES_CSV)

    n_esa    = len(esa_df)
    n_failed = int(y_true.sum())
    n_passed = int((1 - y_true).sum())

    # Load IF model once (used by both IF eval and Hybrid eval)
    if_model  = load_model(REPORT_DIR / f"isolation_forest{suffix}.pkl")
    if_scaler = load_model(REPORT_DIR / f"if_scaler{suffix}.pkl")

    # ── evaluate all methods on identical test set ─────────────────────
    # Each method uses its OWN trained feature columns (see *_feature_cols.json),
    # not a single shared matrix — this avoids "X has N features but scaler
    # expects M" errors when models were trained with different feature subsets.
    #
    # Random Forest prefers rf_loo_predictions.csv (honest out-of-fold scores)
    # over re-scoring the saved .pkl directly, because in --source esa_ref
    # training mode that file is retrained on ALL 48 ESA scenes as its final
    # step — re-scoring it here would be evaluating the model on data it
    # already memorized (the F1=1.00 / AUC=1.00 symptom).
    print("\nEvaluating...")
    results = {}

    print("  Rule-based  (from pipeline_passed column)")
    results["Rule-based"] = eval_rule_based(esa_df, y_true)

    print("  Isolation Forest")
    results["Isolation Forest"] = eval_isolation_forest(esa_df, y_true, suffix=suffix)

    print("  Random Forest")
    results["Random Forest"] = eval_random_forest(esa_df, y_true, suffix=suffix)

    print("  Hybrid  (Rule Stage 1 → IF Stage 2)")
    results["Hybrid"] = eval_hybrid(esa_df, y_true, if_model, if_scaler, suffix=suffix)

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


if __name__ == "__main__":
    main()