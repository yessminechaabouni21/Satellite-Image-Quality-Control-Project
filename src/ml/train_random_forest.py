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
    "NoDataFilter__unexpected_nodata_ratio",
    "BlurFilter__laplacian_variance",
    "StripeFilter__periodic_power_ratio",
    "NoiseFilter__noise_std_ratio",
    "dn_p05", "dn_p25", "dn_p75", "dn_p95",
    "dn_range", "dn_iqr", "dn_skew",
    "inter_band_ratio_nir_red",
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
}


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(feat_cols)
    if missing:
        print(f"  Note: {len(missing)} feature columns missing, skipping")

    X = df[feat_cols].copy()
    for col in feat_cols:
        X[col] = X[col].fillna(X[col].median())

    return df, X.values, feat_cols


def split_train_test(df):
    """
    Training set: all sources (synthetic + ESA + clean)
    Test set    : ESA scenes only (held-out real-world data)

    CRITICAL: group by base_scene to prevent leakage in CV.
    """
    esa_mask   = df["source"] == "esa_ref"
    train_mask = ~esa_mask
    test_mask  = esa_mask & df["label"].notna()

    return train_mask, test_mask


def cross_validate_model(X_train, y_train, groups_train, feat_cols):
    """GroupKFold CV on training set — splits by base_scene."""
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

    n_splits = min(5, len(np.unique(groups_train)))
    cv = GroupKFold(n_splits=n_splits)

    cv_results = cross_validate(
        model, X_scaled, y_train,
        groups=groups_train,
        cv=cv,
        scoring=["roc_auc", "average_precision", "f1"],
        return_train_score=False,
    )

    print(f"\nGroupKFold CV ({n_splits} folds, split by base_scene):")
    for metric in ["test_roc_auc", "test_average_precision", "test_f1"]:
        vals = cv_results[metric]
        name = metric.replace("test_", "")
        print(f"  {name:20s}: {vals.mean():.3f} ± {vals.std():.3f}")

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
    metrics_to_plot = {
        "ROC-AUC":   cv_results["test_roc_auc"],
        "Avg Prec":  cv_results["test_average_precision"],
        "F1":        cv_results["test_f1"],
    }
    positions = range(len(metrics_to_plot))
    bp = ax.boxplot(
        list(metrics_to_plot.values()),
        positions=list(positions),
        patch_artist=True,
        widths=0.4,
    )
    colors = ["#3498db", "#e67e22", "#2ecc71"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(list(metrics_to_plot.keys()))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"GroupKFold CV ({len(cv_results['test_roc_auc'])} folds)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Random Forest — Supervised Defect Classifier", fontsize=13)
    fig.tight_layout()
    out = REPORT_DIR / "random_forest_results.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {out}")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(FEATURES_CSV).exists():
        raise SystemExit(
            f"Feature table not found: {FEATURES_CSV}\n"
            "Run: python -m src.ml.build_feature_table")

    print(f"Loading features from {FEATURES_CSV}")
    df, X, feat_cols = load_data(FEATURES_CSV)
    print(f"  {len(df)} rows, {len(feat_cols)} features")
    print(f"  Clean: {(df['label']==0).sum()}  "
          f"Defective: {(df['label']==1).sum()}")

    train_mask, test_mask = split_train_test(df)
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
    print(f"Test set    : {len(X_test)} scenes "
          f"({int(y_test.sum())} defective)")
    print(f"Unique base_scenes in training: {len(np.unique(groups))}")

    # Cross-validation
    model_cv, scaler, cv_results = cross_validate_model(
        X_train, y_train, groups, feat_cols)

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

    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_final, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(REPORT_DIR / "rf_feature_cols.json", "w") as f:
        json.dump(feat_cols, f)
    print(f"Saved model : {MODEL_PATH}")
    print(f"Saved scaler: {SCALER_PATH}")
    print(f"Saved feature columns: {REPORT_DIR / 'rf_feature_cols.json'}")

    if metrics:
        with open(REPORT_DIR / "rf_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print("\nNext: python -m src.ml.compare_methods")


if __name__ == "__main__":
    main()