# src/ml/calibrate_thresholds.py
#
# Reads reports/ml_features.csv (built by build_feature_table.py) and
# shows the actual metric distributions across ESA-PASSED scenes so you
# can set thresholds from real data instead of guessing.
#
# Also suggests new thresholds at mean + 3*std (or percentile-based)
# and shows how many PASSED scenes each candidate threshold would wrongly reject.
#
# Run from repo root:
#   python -m src.ml.calibrate_thresholds
#
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES_CSV = "reports/ml_features.csv"
REPORT_DIR   = Path("reports/ml")

# Filter metric columns and their current thresholds + direction
# direction: "above" means reject if metric > threshold
#            "below" means reject if metric < threshold
FILTERS = {
    "TOAScalingFilter__max_dn": {
        "label":     "Max DN (TOAScalingFilter)",
        "current":   11200,
        "direction": "above",
        "unit":      "DN",
    },
    "NoiseFilter__noise_std_ratio": {
        "label":     "Noise std ratio (NoiseFilter)",
        "current":   0.03,
        "direction": "above",
        "unit":      "ratio",
    },
    "BlurFilter__laplacian_variance": {
        "label":     "Laplacian variance (BlurFilter)",
        "current":   15.0,
        "direction": "below",
        "unit":      "variance",
    },
    "StripeFilter__periodic_power_ratio": {
        "label":     "Periodic power ratio (StripeFilter)",
        "current":   0.30,
        "direction": "above",
        "unit":      "ratio",
    },
    "MetadataFilter__cloud_cover": {
        "label":     "Cloud cover (MetadataFilter)",
        "current":   60.0,
        "direction": "above",
        "unit":      "%",
    },
    "NoDataFilter__unexpected_nodata_ratio": {
        "label":     "Unexpected no-data ratio (NoDataFilter)",
        "current":   0.05,
        "direction": "above",
        "unit":      "ratio",
    },
}


def load(csv_path):
    df = pd.read_csv(csv_path)
    passed = df[(df["source"] == "esa_ref") & (df["label"] == 0)].copy()
    failed = df[(df["source"] == "esa_ref") & (df["label"] == 1)].copy()
    return df, passed, failed


def suggest_threshold(values, direction, percentile=99):
    """
    For 'above' filters: suggest threshold at the Nth percentile of PASSED values
    (so only the top (100-N)% of clean scenes would be wrongly rejected).
    For 'below' filters: use the (100-N)th percentile.
    """
    clean = values.dropna()
    if len(clean) == 0:
        return None
    if direction == "above":
        return float(np.percentile(clean, percentile))
    else:
        return float(np.percentile(clean, 100 - percentile))


def false_positive_rate(values, threshold, direction):
    """How many PASSED scenes would be rejected at this threshold?"""
    clean = values.dropna()
    if len(clean) == 0:
        return 0, 0
    if direction == "above":
        rejected = (clean > threshold).sum()
    else:
        rejected = (clean < threshold).sum()
    return int(rejected), len(clean)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(FEATURES_CSV).exists():
        raise SystemExit(
            f"{FEATURES_CSV} not found.\n"
            "Run: python -m src.ml.build_feature_table")

    df, passed, failed = load(FEATURES_CSV)
    print(f"ESA-PASSED scenes : {len(passed)}")
    print(f"ESA-FAILED scenes : {len(failed)}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    recommendations = {}

    print(f"\n{'='*78}")
    print(f"{'Metric':<35} {'Min':>8} {'P50':>8} {'P95':>8} {'P99':>8} "
          f"{'Max':>8} {'Current thr':>12} {'FP@current':>10}")
    print(f"{'='*78}")

    for ax, (col, cfg) in zip(axes, FILTERS.items()):
        if col not in passed.columns:
            ax.set_visible(False)
            continue

        vals_pass  = passed[col].dropna()
        vals_fail  = failed[col].dropna() if col in failed.columns else pd.Series()

        if len(vals_pass) == 0:
            ax.set_visible(False)
            continue

        p50 = np.percentile(vals_pass, 50)
        p95 = np.percentile(vals_pass, 95)
        p99 = np.percentile(vals_pass, 99)
        mn  = vals_pass.min()
        mx  = vals_pass.max()

        current_thr = cfg["current"]
        direction   = cfg["direction"]

        fp_n, total = false_positive_rate(vals_pass, current_thr, direction)
        fp_pct = fp_n / total * 100 if total else 0

        print(f"{cfg['label']:<35} {mn:>8.2f} {p50:>8.2f} {p95:>8.2f} "
              f"{p99:>8.2f} {mx:>8.2f} {current_thr:>12.4f} "
              f"{fp_n}/{total} ({fp_pct:.0f}%)")

        # Suggest threshold at P99 of PASSED (1% FPR from threshold alone)
        suggested = suggest_threshold(vals_pass, direction, percentile=99)

        fp_sug_n, _ = false_positive_rate(vals_pass, suggested, direction)
        recommendations[col] = {
            "current":   current_thr,
            "suggested": suggested,
            "direction": direction,
            "fp_current": fp_n,
            "fp_suggested": fp_sug_n,
            "n_passed": total,
            "p50": p50, "p95": p95, "p99": p99,
        }

        # ── histogram ──────────────────────────────────────────────
        all_vals = pd.concat([vals_pass, vals_fail]).dropna()
        bins = np.linspace(all_vals.quantile(0.01), all_vals.quantile(0.99), 40)

        ax.hist(vals_pass, bins=bins, alpha=0.65, color="#2ecc71",
                label=f"PASSED (n={len(vals_pass)})")
        if len(vals_fail):
            ax.hist(vals_fail, bins=bins, alpha=0.65, color="#e74c3c",
                    label=f"FAILED (n={len(vals_fail)})")

        # Current threshold
        ax.axvline(current_thr, color="red", ls="--", lw=2,
                   label=f"Current: {current_thr:.4g}")
        # Suggested threshold
        if suggested is not None:
            ax.axvline(suggested, color="navy", ls=":", lw=2,
                       label=f"Suggested (P99): {suggested:.4g}")

        ax.set_title(cfg["label"], fontsize=10)
        ax.set_xlabel(cfg["unit"])
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Threshold calibration — metric distributions on ESA-PASSED scenes\n"
        "Red dashed = current threshold    Blue dotted = suggested (P99 of PASSED)",
        fontsize=12,
    )
    fig.tight_layout()
    out = REPORT_DIR / "threshold_calibration.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {out}")

    # ── Recommendation table ───────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("RECOMMENDED THRESHOLD CHANGES")
    print(f"{'='*78}")
    print(f"{'Filter col':<42} {'Current':>10} {'Suggested':>10} "
          f"{'FP before':>10} {'FP after':>10}")
    print("-" * 78)

    for col, rec in recommendations.items():
        n = rec["n_passed"]
        print(f"{col:<42} "
              f"{rec['current']:>10.4g} "
              f"{rec['suggested']:>10.4g} "
              f"{rec['fp_current']:>4}/{n} ({rec['fp_current']/n*100:.0f}%)"
              f"  →  "
              f"{rec['fp_suggested']:>4}/{n} ({rec['fp_suggested']/n*100:.0f}%)")

    print(f"\n{'='*78}")
    print("COPY THESE INTO YOUR FILTER INSTANTIATION:")
    print(f"{'='*78}")
    print("\npipeline = Pipeline([")
    print(f"    MetadataFilter(max_cloud=60.0),          # unchanged")
    print(f"    MissingBandsFilter(),                    # unchanged")

    toa = recommendations.get("TOAScalingFilter__max_dn", {})
    if toa:
        sug = toa["suggested"]
        # TOAScalingFilter uses tolerance = ceiling - (quant + shift)
        # quant=10000, shift=1000, so tolerance = suggested - 11000
        tolerance = max(200, int(sug - 11000) + 200)
        print(f"    TOAScalingFilter(tolerance={tolerance}),      "
              f"# was 200, max_dn P99={sug:.0f}")

    print(f"    NoDataFilter(max_unexpected_nodata_ratio=0.05, min_unique_values=100),")

    blur = recommendations.get("BlurFilter__laplacian_variance", {})
    if blur:
        sug = blur["suggested"]
        print(f"    BlurFilter(min_variance={sug:.1f}),          "
              f"# was 15.0, P1 of PASSED={sug:.1f}")

    stripe = recommendations.get("StripeFilter__periodic_power_ratio", {})
    if stripe:
        sug = stripe["suggested"]
        print(f"    StripeFilter(max_periodic_power_ratio={sug:.3f}),  "
              f"# was 0.300, P99={sug:.3f}")

    noise = recommendations.get("NoiseFilter__noise_std_ratio", {})
    if noise:
        sug = noise["suggested"]
        print(f"    NoiseFilter(max_noise_std_ratio={sug:.4f}),  "
              f"# was 0.0300, P99={sug:.4f}")

    print(f"])")

    # Save recommendations as CSV
    rec_df = pd.DataFrame(recommendations).T.reset_index()
    rec_df.columns = ["metric"] + list(rec_df.columns[1:])
    rec_df.to_csv(REPORT_DIR / "threshold_recommendations.csv", index=False)
    print(f"\nSaved recommendations: {REPORT_DIR}/threshold_recommendations.csv")


if __name__ == "__main__":
    main()