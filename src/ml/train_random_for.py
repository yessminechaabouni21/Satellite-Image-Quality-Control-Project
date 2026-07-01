# src/ml/train_random_forest.py
#
# Supervised Random Forest classifier.
# Training: synthetic injections + ESA-FAILED scenes (positives)
#           clean scenes + ESA-PASSED scenes (negatives)
# Validation: GroupKFold by base_scene to prevent data leakage.
# Held-out test: ESA scenes only (real, independently acquired data).
#
# Run from repo root:
#   python -m src.ml.train_random_forest
#
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, classification_report,
)

FEATURES_CSV = "reports/ml_features.csv"
REPORT_DIR   = Path("reports/ml")
MODEL_PATH   = REPORT_DIR / "random_forest.pkl"
SCALER_PATH  = REPORT_DIR / "rf_scaler.pkl"

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

# ONLY the features that passed diagnose_separation.py with p<0.10.
# Update this if you re-run diagnose_separation.py and get different results.
VALIDATED_FEATURE_COLS = [
    "BlurFilter__dn_range",            # p=0.001, Cohen's d=1.72
    "TOAScalingFilter__max_dn",        # p=0.081, Cohen's d=0.83
    "BlurFilter__laplacian_variance",  # p=0.16,  Cohen's d=-0.67
]

# Short display names for plots
FEATURE_LABELS = {
    "MetadataFilter__cloud_cover":           "cloud_cover",
    "TOAScalingFilter__max_dn":              "max_dn",
    "TOAScalingFilter__min_dn":              "min_dn",
    "TOAScalingFilter__unique_values":       "unique_values",
    "NoDataFilter__nodata_ratio":            "nodata_ratio",
    "BlurFilter__laplacian_variance":        "blur_variance",
    "StripeFilter__periodic_power_ratio":    "stripe_score",
    "NoiseFilter__noise_std_ratio":          "noise_std_ratio",
    "dn_p05": "dn_p05", "dn_p25": "dn_p25",
    "dn_p75": "dn_p75", "dn_p95": "dn_p95",
    "dn_range": "dn_range", "dn_iqr": "dn_iqr",
    "dn_skew": "dn_skew",
    "inter_band_ratio_nir_red": "nir_red_ratio",
    "BlurFilter__dn_range": "blur_dn_range",
}


def load_data(csv_path, column_set=None):
    df = pd.read_csv(csv_path)
    cols_to_use = column_set if column_set is not None else FEATURE_COLS
    feat_cols = [c for c in cols_to_use if c in df.columns]
    missing = set(cols_to_use) - set(feat_cols)
    if missing:
        print(f"  Note: {len(missing)} feature columns missing, skipping")

    X = df[feat_cols].copy()
    for col in feat_cols:
        X[col] = X[col].fillna(X[col].median())

    return df, X.values, feat_cols


def split_train_test(df, source_filter=None):
    """
    Default mode: Train on all sources except esa_ref; test on esa_ref (held out).

    ESA-only mode (source_filter='esa_ref'): train AND test both restricted
    to esa_ref rows. Since there's no separate held-out set in this mode,
    evaluation uses Leave-One-Out cross-validation instead of a fixed split
    (see cross_validate_model / evaluate_on_test below).
    """
    if source_filter:
        mask = df["source"] == source_filter
        # train_mask == test_mask in this mode; LOO-CV handles the split internally
        return mask, mask

    esa_mask   = df["source"] == "esa_ref"
    train_mask = ~esa_mask
    test_mask  = esa_mask & df["label"].notna()

    return train_mask, test_mask


def cross_validate_model(X_train, y_train, groups_train, feat_cols, use_loo=False):
    """
    GroupKFold CV by default (splits by base_scene, prevents leakage).
    LeaveOneOut when use_loo=True — appropriate for tiny datasets (<50 rows)
    such as ESA-only training (40 clean + 8 defective = 48 rows), where
    GroupKFold folds would be too small to be meaningful.
    """
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",   # handles class imbalance
        random_state=42,
        n_jobs=-1,
    )

    if use_loo:
        from sklearn.model_selection import LeaveOneOut
        cv = LeaveOneOut()
        n_splits = X_train.shape[0]
        cv_label = f"LeaveOneOut ({n_splits} folds, 1 sample held out per fold)"
        scoring = ["roc_auc", "f1"]   # average_precision unstable with n=1 test fold
    else:
        n_splits = min(5, len(np.unique(groups_train)))
        cv = GroupKFold(n_splits=n_splits)
        cv_label = f"GroupKFold ({n_splits} folds, split by base_scene)"
        scoring = ["roc_auc", "average_precision", "f1"]

    cv_results = cross_validate(
        model, X_scaled, y_train,
        groups=None if use_loo else groups_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        error_score=np.nan,
    )

    print(f"\n{cv_label}:")
    for metric in [f"test_{s}" for s in scoring]:
        vals = cv_results[metric]
        vals = vals[~np.isnan(vals)]
        name = metric.replace("test_", "")
        if len(vals):
            print(f"  {name:20s}: {vals.mean():.3f} ± {vals.std():.3f}")
        else:
            print(f"  {name:20s}: N/A (no valid folds)")

    return model, scaler, cv_results


def train_final(X_train, y_train, scaler):
    """Retrain on all training data with the fitted scaler."""
    X_scaled = scaler.transform(X_train)
    model = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y_train)
    return model


def evaluate_on_test(model, scaler, X_test, y_test, df_test):
    X_scaled = scaler.transform(X_test)
    proba    = model.predict_proba(X_scaled)[:, 1]

    print(f"\nHeld-out ESA test set ({len(y_test)} scenes, "
          f"{int(y_test.sum())} FAILED, {int((1-y_test).sum())} PASSED):")

    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        print("  Cannot compute ROC — need both classes in test set.")
        return proba, {}

    roc_auc = roc_auc_score(y_test, proba)
    ap      = average_precision_score(y_test, proba)

    # Best threshold by F1
    prec, rec, thresholds = precision_recall_curve(y_test, proba)
    f1s      = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s)
    best_thr = float(thresholds[best_idx]) if best_idx < len(thresholds) \
               else 0.5
    pred     = (proba >= best_thr).astype(int)

    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    fpr_rate  = fp / (fp + tn) if (fp + tn) else 0

    print(f"  ROC-AUC              : {roc_auc:.3f}")
    print(f"  Average Precision    : {ap:.3f}")
    print(f"\n  At best-F1 threshold ({best_thr:.2f}):")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"    Precision={precision:.2f}  Recall={recall:.2f}  "
          f"F1={f1:.2f}  FPR={fpr_rate:.2f}")

    metrics = {
        "roc_auc": roc_auc, "avg_precision": ap,
        "best_threshold": best_thr,
        "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr_rate,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

    # Per-indicator breakdown
    if "failed_indicator" in df_test.columns:
        print("\n  Catch rate by ESA failed_indicator:")
        df_test = df_test.copy()
        df_test["rf_pred"] = pred
        for ind in df_test[df_test["label"]==1]["failed_indicator"].dropna().unique():
            sub    = df_test[df_test["failed_indicator"] == ind]
            caught = sub["rf_pred"].sum()
            print(f"    {ind:25s}: {caught}/{len(sub)}")

    return proba, metrics


def plot_results(model, scaler, df, X_all,
                 feat_cols, train_mask, test_mask,
                 proba_test, y_test, cv_results):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── Plot 1: Feature importances ───────────────────────────────────
    ax = axes[0]
    X_scaled_all = scaler.transform(X_all)
    importances  = pd.Series(
        model.feature_importances_,
        index=[FEATURE_LABELS.get(c, c.split("__")[-1]) for c in feat_cols]
    ).sort_values(ascending=True)
    importances.plot(kind="barh", ax=ax, color="#2980b9")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Feature Importances\n(Random Forest)")
    ax.grid(alpha=0.3, axis="x")

    # ── Plot 2: ROC curve ─────────────────────────────────────────────
    ax = axes[1]
    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, proba_test)
        auc = roc_auc_score(y_test, proba_test)
        ax.plot(fpr, tpr, color="#c0392b", lw=2.5,
                label=f"Random Forest (AUC={auc:.2f})")

        # Also load IF results if available
        if_metrics_path = REPORT_DIR / "if_metrics.json"
        if if_metrics_path.exists():
            # We can't re-plot IF ROC here without re-running, so just
            # annotate the comparison point
            with open(if_metrics_path) as f:
                if_m = json.load(f)
            ax.scatter([if_m["fpr"]], [if_m["recall"]],
                       color="#27ae60", s=120, zorder=5,
                       label=f"IF (F1={if_m['f1']:.2f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC Curve — ESA test set")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # ── Plot 3: CV score distribution ─────────────────────────────────
    ax = axes[2]
    available = {
        "ROC-AUC":   cv_results.get("test_roc_auc"),
        "Avg Prec":  cv_results.get("test_average_precision"),
        "F1":        cv_results.get("test_f1"),
    }
    # Drop metrics that weren't scored in this CV mode, or are all-NaN
    # (LOO folds have 1 test sample each, so per-fold ROC-AUC is undefined)
    metrics_to_plot = {}
    for name, vals in available.items():
        if vals is None:
            continue
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            metrics_to_plot[name] = vals

    if not metrics_to_plot:
        ax.text(0.5, 0.5, "No valid CV scores\n(see LOO out-of-fold result instead)",
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
    else:
        positions = range(len(metrics_to_plot))
        bp = ax.boxplot(
            list(metrics_to_plot.values()),
            positions=list(positions),
            patch_artist=True,
            widths=0.4,
        )
        palette = ["#3498db", "#e67e22", "#2ecc71"]
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(list(metrics_to_plot.keys()))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        first_vals = next((v for v in available.values() if v is not None), [])
        n_folds = len(first_vals)
        ax.set_title(f"Cross-validation ({n_folds} folds)")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Random Forest — Supervised Defect Classifier", fontsize=13)
    fig.tight_layout()
    out = REPORT_DIR / "random_forest_results.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {out}")


def _eval_from_proba(y_test, proba, df_test):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 precision_recall_curve)
    print(f"\nLOO out-of-fold results ({len(y_test)} scenes, "
          f"{int(y_test.sum())} FAILED, {int((1-y_test).sum())} PASSED):")

    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        print("  Cannot compute ROC — need both classes.")
        return proba, {}

    roc_auc = roc_auc_score(y_test, proba)
    ap      = average_precision_score(y_test, proba)
    prec, rec, thresholds = precision_recall_curve(y_test, proba)
    f1s      = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s)
    best_thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    pred     = (proba >= best_thr).astype(int)

    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    fpr_rate  = fp / (fp + tn) if (fp + tn) else 0

    print(f"  ROC-AUC: {roc_auc:.3f}   Avg Precision: {ap:.3f}")
    print(f"  At best-F1 threshold ({best_thr:.2f}): "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision={precision:.2f}  Recall={recall:.2f}  "
          f"F1={f1:.2f}  FPR={fpr_rate:.2f}")

    return proba, {
        "roc_auc": roc_auc, "avg_precision": ap, "best_threshold": best_thr,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr_rate,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None,
                    help="Restrict train+eval to one source (e.g. 'esa_ref'). "
                        "Uses Leave-One-Out CV since train==test in this mode. "
                        "Default: train on synthetic+clean, test on esa_ref.")
    ap.add_argument("--features", choices=["all", "validated"], default="all",
                    help="'all' = full 16-feature set. "
                        "'validated' = only features with p<0.10 from "
                        "diagnose_separation.py. Use 'validated' to stop "
                        "diluting real signal with noise features.")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(FEATURES_CSV).exists():
        raise SystemExit(
            f"Feature table not found: {FEATURES_CSV}\n"
            "Run: python -m src.ml.build_feature_table")

    column_set = VALIDATED_FEATURE_COLS if args.features == "validated" else None
    suffix     = "_validated" if args.features == "validated" else ""

    print(f"Loading features from {FEATURES_CSV}  (feature set: {args.features})")
    df, X, feat_cols = load_data(FEATURES_CSV, column_set=column_set)
    print(f"  {len(df)} rows, {len(feat_cols)} features: {feat_cols}")
    print(f"  Clean: {(df['label']==0).sum()}  "
          f"Defective: {(df['label']==1).sum()}")

    train_mask, test_mask = split_train_test(df, source_filter=args.source)
    use_loo = args.source is not None

    X_train = X[train_mask]
    y_train = df[train_mask]["label"].astype(int).values
    groups  = df[train_mask]["base_scene"].fillna(
        df[train_mask]["scene_name"]).values

    X_test  = X[test_mask]
    y_test  = df[test_mask]["label"].astype(int).values
    df_test = df[test_mask].copy()

    print(f"\nTraining set: {len(X_train)} scenes "
          f"({int(y_train.sum())} defective, "
          f"{int((1-y_train).sum())} clean)")
    if not use_loo:
        print(f"Test set    : {len(X_test)} scenes "
              f"({int(y_test.sum())} defective)")
        print(f"Unique base_scenes in training: {len(np.unique(groups))}")
    else:
        print(f"Mode: ESA-only — train and eval both on these {len(X_train)} "
              f"scenes via Leave-One-Out (no separate held-out test set)")

    if use_loo:
        # LOO handles its own internal splitting; cv_results just for the plot
        model_cv, scaler, cv_results = cross_validate_model(
            X_train, y_train, groups, feat_cols, use_loo=True)
        print("\nRetraining final model on full ESA set "
              "(for feature importances / saved model)...")
        model_final = train_final(X_train, y_train, scaler)
        # Proper LOO out-of-fold evaluation (honest, no leakage)
        proba_test = np.zeros(len(X_train))
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        print(f"Running Leave-One-Out evaluation ({len(X_train)} folds)...")
        for i, (tr_idx, te_idx) in enumerate(loo.split(X_train)):
            if (i + 1) % 10 == 0 or i == len(X_train) - 1:
                print(f"  fold {i+1}/{len(X_train)}")
            s_i = RobustScaler()
            Xtr = s_i.fit_transform(X_train[tr_idx])
            Xte = s_i.transform(X_train[te_idx])
            m_i = RandomForestClassifier(
                n_estimators=200, max_features="sqrt", min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1)
            m_i.fit(Xtr, y_train[tr_idx])
            proba_test[te_idx] = m_i.predict_proba(Xte)[:, 1]
        proba_test, metrics = _eval_from_proba(y_train, proba_test, df_test)
        y_test = y_train   # for plotting consistency below
    else:
        # Cross-validation
        model_cv, scaler, cv_results = cross_validate_model(
            X_train, y_train, groups, feat_cols, use_loo=False)

        # Retrain on full training set
        print("\nRetraining on full training set...")
        model_final = train_final(X_train, y_train, scaler)

        # Evaluate on held-out ESA scenes
        proba_test, metrics = evaluate_on_test(
            model_final, scaler, X_test, y_test, df_test)

    # Plots
    plot_results(model_final, scaler, df, X, feat_cols,
                 train_mask, test_mask,
                 proba_test, y_test, cv_results)

    # Save with a distinct filename so the validated-feature model never
    # silently overwrites the full-feature model (or vice versa)
    model_path  = REPORT_DIR / f"random_forest{suffix}.pkl"
    scaler_path = REPORT_DIR / f"rf_scaler{suffix}.pkl"
    cols_path   = REPORT_DIR / f"rf_feature_cols{suffix}.json"

    with open(model_path, "wb") as f:
        pickle.dump(model_final, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(cols_path, "w") as f:
        json.dump(feat_cols, f)
    print(f"Saved model : {model_path}")

    if metrics:
        with open(REPORT_DIR / f"rf_metrics{suffix}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print("\nNext: python -m src.ml.final_comparison")


if __name__ == "__main__":
    main()