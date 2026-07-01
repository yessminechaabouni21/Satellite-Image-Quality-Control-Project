# src/ml/diagnose_separation.py
#
# For each feature, measures how well it separates ESA-FAILED from
# ESA-PASSED scenes using Mann-Whitney U test + effect size.
# This tells you EXACTLY which signals are worth keeping/adding,
# instead of guessing.
#
# Run from repo root:
#   python -m src.ml.diagnose_separation
#
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

FEATURES_CSV = "reports/ml_features.csv"

ALL_NUMERIC_FEATURE_PREFIXES = ["__", "dn_"]  # any column matching these

# Exclude columns that contain "__" or "dn_" but are categorical/metadata,
# not continuous measurements — including them gives misleading p-values
# (e.g. processing_baseline "05.10"/"05.11" parses as float but isn't a
# real continuous signal; a fake low p-value with Cohen's d≈0 is the tell).
EXCLUDE_COLUMNS = {
    "TOAScalingFilter__processing_baseline",
    "TOAScalingFilter__metadata_source",
    "TOAScalingFilter__radio_add_offset_xml",   # categorical per-baseline constant
    "TOAScalingFilter__quantification_value",   # categorical per-baseline constant
    "TOAScalingFilter__dn_ceiling",             # derived constant, not a measurement
    "NoDataFilter__threshold",                  # fixed filter parameter, not data
    "BlurFilter__threshold",                    # fixed filter parameter, not data
    "StripeFilter__threshold",                  # fixed filter parameter, not data
    "NoiseFilter__threshold",                   # fixed filter parameter, not data
    "MissingBandsFilter__required_count",       # always the same constant
}


def main():
    if not Path(FEATURES_CSV).exists():
        raise SystemExit(f"{FEATURES_CSV} not found. Run build_feature_table.py first.")

    df = pd.read_csv(FEATURES_CSV)
    esa = df[df["source"] == "esa_ref"].dropna(subset=["label"])
    esa["label"] = esa["label"].astype(int)

    passed = esa[esa["label"] == 0]
    failed = esa[esa["label"] == 1]

    print(f"PASSED: {len(passed)}   FAILED: {len(failed)}\n")

    feat_cols = [c for c in esa.columns
                 if any(p in c for p in ALL_NUMERIC_FEATURE_PREFIXES)
                 and c not in EXCLUDE_COLUMNS]

    # Coerce each candidate column to numeric; drop any that aren't truly numeric
    # (dtype alone is unreliable here — object columns can still be numeric-looking
    # after a CSV round-trip, and some "__" columns like processing_baseline or
    # metadata_source are genuinely text)
    numeric_feat_cols = []
    for col in feat_cols:
        coerced = pd.to_numeric(esa[col], errors="coerce")
        # Keep only if at least half the non-null values survived coercion
        non_null = esa[col].notna().sum()
        if non_null == 0:
            continue
        survived = coerced.notna().sum()
        if survived >= 0.5 * non_null:
            numeric_feat_cols.append(col)
    feat_cols = numeric_feat_cols
    print(f"Using {len(feat_cols)} numeric feature columns "
          f"(dropped non-numeric ones like processing_baseline, metadata_source)\n")

    results = []
    for col in feat_cols:
        p_vals = pd.to_numeric(passed[col], errors="coerce").dropna()
        f_vals = pd.to_numeric(failed[col], errors="coerce").dropna()
        if len(p_vals) < 3 or len(f_vals) < 3:
            continue

        # Mann-Whitney U test (non-parametric, robust to small samples)
        try:
            u_stat, p_value = stats.mannwhitneyu(
                p_vals.to_numpy(dtype=float),
                f_vals.to_numpy(dtype=float),
                alternative="two-sided")
        except ValueError:
            continue

        # Effect size: rank-biserial correlation (-1 to 1)
        n1, n2 = len(p_vals), len(f_vals)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)

        # Separation in std units (Cohen's d style, robust version)
        pooled_std = np.sqrt((p_vals.std()**2 + f_vals.std()**2) / 2)
        cohens_d = (f_vals.mean() - p_vals.mean()) / pooled_std if pooled_std > 0 else 0

        results.append({
            "feature": col,
            "passed_mean": p_vals.mean(),
            "failed_mean": f_vals.mean(),
            "passed_median": p_vals.median(),
            "failed_median": f_vals.median(),
            "p_value": p_value,
            "effect_size": abs(effect_size),
            "cohens_d": cohens_d,
        })

    res_df = pd.DataFrame(results).sort_values("effect_size", ascending=False)

    print("="*100)
    print("FEATURE SEPARATION — ranked by effect size (higher = better discriminator)")
    print("="*100)
    print(f"{'Feature':<42} {'PASSED mean':>13} {'FAILED mean':>13} "
          f"{'p-value':>9} {'effect':>7} {'Cohen d':>8}")
    print("-"*100)
    for _, r in res_df.iterrows():
        sig = "**" if r["p_value"] < 0.05 else "  "
        suspicious = " <-- check: p sig but |Cohen d|~0, may be categorical" \
            if r["p_value"] < 0.05 and abs(r["cohens_d"]) < 0.1 else ""
        print(f"{r['feature']:<42} {r['passed_mean']:>13.3f} {r['failed_mean']:>13.3f} "
              f"{r['p_value']:>9.3f} {r['effect_size']:>7.3f} {r['cohens_d']:>8.2f} "
              f"{sig}{suspicious}")

    print(f"\n** = statistically significant at p<0.05 (n is small, interpret cautiously)")

    n_sig = (res_df["p_value"] < 0.05).sum()
    print(f"\n{n_sig}/{len(res_df)} features show significant separation (p<0.05)")

    if n_sig == 0:
        print("\nNo feature individually separates FAILED from PASSED at p<0.05.")
        print("This confirms: general_quality/geometric_quality defects are NOT")
        print("visible in any single pixel-statistic feature you currently compute.")
        print("To improve detection you need NEW feature types, not better models.")
    else:
        print(f"\nTop discriminating features (worth emphasizing in your model):")
        for _, r in res_df.head(min(5, n_sig)).iterrows():
            direction = "higher" if r["cohens_d"] > 0 else "lower"
            print(f"  {r['feature']}: FAILED scenes have {direction} values "
                  f"(p={r['p_value']:.3f})")

    out_csv = "reports/ml/feature_separation.csv"
    Path("reports/ml").mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()