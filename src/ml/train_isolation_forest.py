# src/ml/train_isolation_forest.py
#
# One-class anomaly detector using Isolation Forest.
# Trained ONLY on clean scenes — no defective examples needed.
#
# Run from repo root:
#   python -m src.ml.train_isolation_forest
#
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)

FEATURES_CSV = "reports/ml_features.csv"
REPORT_DIR   = Path("reports/ml")
MODEL_PATH   = REPORT_DIR / "isolation_forest.pkl"
SCALER_PATH  = REPORT_DIR / "if_scaler.pkl"

# Use only ESA-validated scenes for evaluation (most reliable labels)
ESA_SOURCES  = {"esa_ref"}

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


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Keep only columns that actually exist
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(feat_cols)
    if missing:
        print(f"  Note: {len(missing)} feature columns not found, skipping them")
        print(f"  Missing: {missing}")

    # Median-impute NaNs per column
    X = df[feat_cols].copy()
    for col in feat_cols:
        X[col] = X[col].fillna(X[col].median())

    return df, X, feat_cols


def train(df, X, feat_cols):
    # Train ONLY on clean scenes
    clean_mask = df["label"] == 0
    X_clean    = X[clean_mask].values
    print(f"Training on {X_clean.shape[0]} clean scenes, "
          f"{X_clean.shape[1]} features")

    scaler  = RobustScaler()           # robust to outliers (better than StandardScaler)
    X_clean_scaled = scaler.fit_transform(X_clean)

    # contamination=0.05 means we expect ~5% of training data might be mislabeled
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_features=0.8,
        random_state=42,
    )
    model.fit(X_clean_scaled)
    return model, scaler


def evaluate(model, scaler, df, X, feat_cols):
    """
    Evaluate on ESA-validated scenes only (most reliable ground truth).
    decision_function: higher = more normal, lower = more anomalous.
    We negate it so higher score = more defective (standard convention).
    """
    esa_mask = df["source"].isin(ESA_SOURCES) & df["label"].notna()
    df_esa   = df[esa_mask].copy()
    X_esa    = X[esa_mask].values

    if len(df_esa) == 0:
        print("No ESA-reference scenes found for evaluation.")
        return None

    X_esa_scaled  = scaler.transform(X_esa)
    anomaly_score = -model.decision_function(X_esa_scaled)   # higher = more anomalous
    pred_label    = (model.predict(X_esa_scaled) == -1).astype(int)

    df_esa = df_esa.copy()
    df_esa["anomaly_score"] = anomaly_score
    df_esa["if_predicted"]  = pred_label

    y_true = df_esa["label"].astype(int).values

    print(f"\nEvaluation on {len(df_esa)} ESA scenes "
          f"({int(y_true.sum())} FAILED, {int((1-y_true).sum())} PASSED)")

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print("  Cannot compute ROC — need both classes in ESA set.")
        return df_esa

    roc_auc = roc_auc_score(y_true, anomaly_score)
    ap      = average_precision_score(y_true, anomaly_score)
    print(f"  ROC-AUC              : {roc_auc:.3f}")
    print(f"  Average Precision    : {ap:.3f}")

    # Best threshold by F1
    prec, rec, thresholds = precision_recall_curve(y_true, anomaly_score)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s)
    best_thr = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    pred_best = (anomaly_score >= best_thr).astype(int)

    tp = int(((pred_best == 1) & (y_true == 1)).sum())
    fp = int(((pred_best == 1) & (y_true == 0)).sum())
    fn = int(((pred_best == 0) & (y_true == 1)).sum())
    tn = int(((pred_best == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    fpr_rate  = fp / (fp + tn) if (fp + tn) else 0

    print(f"\n  At best-F1 threshold ({best_thr:.3f}):")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"    Precision={precision:.2f}  Recall={recall:.2f}  "
          f"F1={f1:.2f}  FPR={fpr_rate:.2f}")

    return df_esa, {
        "roc_auc": roc_auc, "avg_precision": ap,
        "best_threshold": float(best_thr),
        "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr_rate,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def plot_results(model, scaler, df, X, feat_cols, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Plot 1: anomaly score distribution ───────────────────────────
    ax = axes[0]
    esa_mask = df["source"].isin(ESA_SOURCES) & df["label"].notna()
    df_esa   = df[esa_mask].copy()
    X_esa_scaled = scaler.transform(X[esa_mask].values)
    scores = -model.decision_function(X_esa_scaled)
    df_esa = df_esa.copy()
    df_esa["score"] = scores

    clean    = df_esa[df_esa["label"] == 0]["score"]
    defect   = df_esa[df_esa["label"] == 1]["score"]

    bins = np.linspace(scores.min(), scores.max(), 30)
    ax.hist(clean,  bins=bins, alpha=0.6, color="#2ecc71", label=f"Clean (n={len(clean)})")
    ax.hist(defect, bins=bins, alpha=0.6, color="#e74c3c", label=f"Defective (n={len(defect)})")
    if metrics and "best_threshold" in metrics:
        ax.axvline(metrics["best_threshold"], color="navy", ls="--",
                   lw=2, label=f"Threshold={metrics['best_threshold']:.2f}")
    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    ax.set_ylabel("Count")
    ax.set_title("Isolation Forest: score distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Plot 2: ROC curve ─────────────────────────────────────────────
    ax = axes[1]
    y_true = df_esa["label"].astype(int).values
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, color="#c0392b", lw=2,
                label=f"Isolation Forest (AUC={auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(alpha=0.3)

    # ── Plot 3: feature contributions (mean anomaly score per feature) ─
    ax = axes[2]
    # Show how anomalous the clean vs defective scenes are per feature
    feat_importance = {}
    for col in feat_cols[:10]:   # top 10 for readability
        col_idx = feat_cols.index(col)
        clean_vals   = X[esa_mask & (df["label"] == 0)].iloc[:, col_idx]
        defect_vals  = X[esa_mask & (df["label"] == 1)].iloc[:, col_idx]
        if clean_vals.std() > 0:
            separation = abs(defect_vals.mean() - clean_vals.mean()) / clean_vals.std()
            feat_importance[col.split("__")[-1]] = float(separation)

    if feat_importance:
        fi = pd.Series(feat_importance).sort_values(ascending=True)
        fi.plot(kind="barh", ax=ax, color="#2980b9")
        ax.set_xlabel("Mean separation (std units)")
        ax.set_title("Feature separation\n(clean vs defective)")
        ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Isolation Forest — One-Class Anomaly Detection", fontsize=13)
    fig.tight_layout()
    out = REPORT_DIR / "isolation_forest_results.png"
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
          f"Defective: {(df['label']==1).sum()}\n")

    model, scaler = train(df, X, feat_cols)

    result = evaluate(model, scaler, df, X, feat_cols)
    metrics = result[1] if isinstance(result, tuple) else None

    plot_results(model, scaler, df, X, feat_cols, metrics)

    # Save model + scaler + feature columns
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(REPORT_DIR / "if_feature_cols.json", "w") as f:
        json.dump(feat_cols, f)
    print(f"Saved model : {MODEL_PATH}")
    print(f"Saved scaler: {SCALER_PATH}")
    print(f"Saved feature columns: {REPORT_DIR / 'if_feature_cols.json'}")

    if metrics:
        with open(REPORT_DIR / "if_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print("\nNext: python -m src.ml.train_random_forest")


if __name__ == "__main__":
    main()