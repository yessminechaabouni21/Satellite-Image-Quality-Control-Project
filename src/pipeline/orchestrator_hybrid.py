# src/pipeline/orchestrator_hybrid.py
"""
Hybrid Orchestrator - Two-stage QC pipeline (Rule-based + ML).

This is a NEW file. The original orchestrator.py is kept untouched.

Usage:
    from src.pipeline.orchestrator_hybrid import HybridPipeline

    # Rule-only mode (backward compatible API)
    pipeline = HybridPipeline(filters, mode="rule_only")
    result = pipeline.run("scene.SAFE")
    # result: {accepted, failed_filter, failure_reason, results}

    # Hybrid mode (Rule Stage 1 + ML Stage 2)
    pipeline = HybridPipeline(filters, mode="hybrid", if_model_path="models/isolation_forest.pkl")
    result = pipeline.run("scene.SAFE")
    # result adds: {decision, stage1_passed, stage1_failed_filter, stage2_score, reason}
"""

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


class HybridPipeline:
    """
    Two-stage quality control pipeline.

    Stage 1: Rule-based filters (fast, deterministic, explainable)
    Stage 2: Isolation Forest anomaly scoring (catches subtle defects)

    Parameters
    ----------
    filters : list
        List of filter instances for Stage 1 (same as original Pipeline)
    mode : str
        "rule_only" (original behavior) or "hybrid" (Rule + ML)
    if_model_path : str, optional
        Path to saved Isolation Forest model (required for hybrid mode)
    review_threshold_low : float
        Lower bound for REVIEW zone (default 0.3)
    review_threshold_high : float
        Upper bound for REVIEW zone (default 0.7)
    """

    def __init__(
        self,
        filters: List,
        mode: str = "rule_only",
        if_model_path: Optional[str] = None,
        review_threshold_low: float = 0.3,
        review_threshold_high: float = 0.7,
    ):
        self.filters = filters
        self.mode = mode
        self.if_model_path = if_model_path
        self.review_low = review_threshold_low
        self.review_high = review_threshold_high

        # Lazy-load Stage 2 only when needed
        self._stage2 = None
        self._scaler = None

    @property
    def stage2(self):
        """Lazy initialization of Stage 2 ML model."""
        if self._stage2 is None and self.mode == "hybrid":
            import pickle

            if self.if_model_path is None:
                raise ValueError("if_model_path required for hybrid mode")

            p = Path(self.if_model_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"Isolation Forest model not found at {p}. "
                    "Train first: python -m src.models.train_models"
                )

            with open(p, "rb") as f:
                self._stage2 = pickle.load(f)

            # Load scaler if exists
            scaler_path = p.with_suffix(".scaler.pkl")
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self._scaler = pickle.load(f)

        return self._stage2

    def run(self, scene_path: str) -> Dict:
        """
        Run hybrid pipeline on a single scene.

        Returns (backward compatible + new fields):
            - scene: str
            - accepted: bool (True=ACCEPT, False=REJECT/REVIEW)
            - failed_filter: str or None (Stage 1 filter that failed)
            - failure_reason: str or None (reason for Stage 1 failure)
            - results: dict (per-filter results from Stage 1)

            # NEW fields (hybrid mode only):
            - decision: "ACCEPT", "REVIEW", or "REJECT"
            - stage1_passed: bool
            - stage1_failed_filter: str or None
            - stage2_score: float or None (normalized anomaly score 0-1)
            - stage2_raw_score: float or None (IF decision_function output)
            - reason: str (human-readable explanation)
        """
        scene_path = Path(scene_path)

        # =====================================================================
        # STAGE 1: Rule-based filters (EXACT same logic as original)
        # =====================================================================
        results = {}
        failed_filter = None
        failure_reason = None

        for f in self.filters:
            res = f.run(scene_path)

            # CLEAN NUMPY TYPES (same as original)
            results[f.name] = {
                "passed": bool(res.passed),
                "reason": res.reason,
                "metrics": {
                    k: (float(v) if hasattr(v, "item") else v)
                    for k, v in res.metrics.items()
                }
            }

            if not res.passed:
                failed_filter = f.name
                failure_reason = res.reason

                # Build result dict (backward compatible)
                base_result = {
                    "scene": str(scene_path),
                    "accepted": False,
                    "failed_filter": failed_filter,
                    "failure_reason": failure_reason,
                    "results": results,
                }

                # Add hybrid fields
                if self.mode == "hybrid":
                    base_result.update({
                        "decision": "REJECT",
                        "stage1_passed": False,
                        "stage1_failed_filter": failed_filter,
                        "stage2_score": None,
                        "stage2_raw_score": None,
                        "reason": f"Stage 1 (Rule-based): Failed {failed_filter} — {failure_reason}",
                    })

                return base_result

        # =====================================================================
        # All Stage 1 filters passed
        # =====================================================================
        base_result = {
            "scene": str(scene_path),
            "accepted": True,
            "failed_filter": None,
            "failure_reason": None,
            "results": results,
        }

        # =====================================================================
        # MODE: RULE_ONLY (original behavior — return now)
        # =====================================================================
        if self.mode == "rule_only":
            return base_result

        # =====================================================================
        # MODE: HYBRID (Rule passed — run Stage 2 ML)
        # =====================================================================
        if self.mode == "hybrid":
            try:
                from src.features.extractor import extract_scene_features

                features = extract_scene_features(scene_path)
                feature_vector = self._prepare_features(features)

                # Scale features
                if self._scaler is not None:
                    feature_vector = self._scaler.transform(feature_vector.reshape(1, -1))
                else:
                    feature_vector = feature_vector.reshape(1, -1)

                # Get anomaly score from Isolation Forest
                raw_score = self.stage2.decision_function(feature_vector)[0]

                # Normalize to [0, 1] for interpretability
                # decision_function: positive = normal, negative = anomaly
                # We want higher score = more anomalous
                normalized_score = 1.0 / (1.0 + np.exp(5 * raw_score))

                # Apply thresholds
                if normalized_score > self.review_high:
                    decision = "REJECT"
                    accepted = False
                    reason = (
                        f"Stage 2 (ML): Anomaly score {normalized_score:.3f} > "
                        f"{self.review_high} (high confidence defect)"
                    )
                elif normalized_score > self.review_low:
                    decision = "REVIEW"
                    accepted = False  # REVIEW = not accepted (needs human)
                    reason = (
                        f"Stage 2 (ML): Anomaly score {normalized_score:.3f} in "
                        f"[{self.review_low}, {self.review_high}] (borderline, human review)"
                    )
                else:
                    decision = "ACCEPT"
                    accepted = True
                    reason = (
                        f"Stage 2 (ML): Anomaly score {normalized_score:.3f} < "
                        f"{self.review_low} (clean)"
                    )

                base_result.update({
                    "accepted": accepted,
                    "decision": decision,
                    "stage1_passed": True,
                    "stage1_failed_filter": None,
                    "stage2_score": float(normalized_score),
                    "stage2_raw_score": float(raw_score),
                    "reason": reason,
                })

            except Exception as e:
                # Stage 2 failed — fall back to Stage 1 result
                base_result.update({
                    "decision": "ACCEPT",
                    "stage1_passed": True,
                    "stage1_failed_filter": None,
                    "stage2_score": None,
                    "stage2_raw_score": None,
                    "reason": f"Stage 1 passed, Stage 2 error: {e}",
                })

            return base_result

        raise ValueError(f"Unknown mode: {self.mode}. Use 'rule_only' or 'hybrid'.")

    def _prepare_features(self, features: Dict) -> np.ndarray:
        """Convert feature dict to numpy array in training order."""
        # This must match the feature order used during training
        feature_cols = [
            'severity',
            'MetadataFilter__cloud_cover',
            'MissingBandsFilter__found_count',
            'MissingBandsFilter__required_count',
            'TOAScalingFilter__min_dn',
            'TOAScalingFilter__max_dn',
            'TOAScalingFilter__dn_p9999',
            'TOAScalingFilter__pct_above_ceiling',
            'TOAScalingFilter__unique_values',
            'TOAScalingFilter__quantification_value',
            'TOAScalingFilter__radio_add_offset_xml',
            'TOAScalingFilter__dn_ceiling',
            'TOAScalingFilter__processing_baseline',
            'NoDataFilter__unexpected_nodata_ratio',
            'NoDataFilter__threshold',
            'NoDataFilter__total_nodata_ratio',
            'NoDataFilter__unique_values',
            'NoDataFilter__saturated_ratio',
            'BlurFilter__laplacian_variance',
            'BlurFilter__threshold',
            'BlurFilter__dn_range',
            'StripeFilter__periodic_power_ratio',
            'StripeFilter__peak_period_pixels',
            'StripeFilter__threshold',
            'StripeFilter__detrended_std',
            'NoiseFilter__noise_std_ratio',
            'NoiseFilter__noise_std',
            'NoiseFilter__signal_median',
            'NoiseFilter__threshold',
            'dn_p05',
            'dn_p25',
            'dn_p75',
            'dn_p95',
            'dn_range',
            'dn_iqr',
            'dn_skew',
            'inter_band_ratio_nir_red',
            'StripeFilter__periodic_ratio',
        ]

        vec = []
        for col in feature_cols:
            val = features.get(col, 0.0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0.0
            vec.append(float(val))

        return np.array(vec)


# =====================================================================
# Factory function (optional convenience)
# =====================================================================

def build_pipeline(
    mode: str = "rule_only",
    if_model_path: Optional[str] = None,
    review_threshold_low: float = 0.3,
    review_threshold_high: float = 0.7,
):
    """
    Build hybrid pipeline with standard filter stack.

    Parameters
    ----------
    mode : str
        "rule_only" or "hybrid"
    if_model_path : str, optional
        Path to saved Isolation Forest model
    """
    from src.filters.metadata_filter import MetadataFilter
    from src.filters.noData_filter import NoDataFilter
    from src.filters.toascaling_filter import TOAScalingFilter
    from src.filters.blur_filter import BlurFilter
    from src.filters.noise_filter import NoiseFilter
    from src.filters.missing_bands_filter import MissingBandsFilter
    from src.filters.stripe_filter import StripeFilter

    filters = [
        MetadataFilter(max_cloud=60.0),
        MissingBandsFilter(),
        TOAScalingFilter(),
        NoDataFilter(max_unexpected_nodata_ratio=0.05, min_unique_values=100),
        BlurFilter(min_variance=15.0),
        StripeFilter(max_periodic_power_ratio=0.3),
        NoiseFilter(max_noise_std_ratio=0.15),
    ]

    return HybridPipeline(
        filters,
        mode=mode,
        if_model_path=if_model_path,
        review_threshold_low=review_threshold_low,
        review_threshold_high=review_threshold_high,
    )


# =====================================================================
# CLI entry point for testing
# =====================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.orchestrator_hybrid <scene.SAFE> [scene2.SAFE ...]")
        print("       python -m src.pipeline.orchestrator_hybrid --hybrid <scene.SAFE> [...]")
        sys.exit(1)

    # Check for --hybrid flag
    if sys.argv[1] == "--hybrid":
        mode = "hybrid"
        scene_paths = sys.argv[2:]
        pipeline = build_pipeline(mode="hybrid", if_model_path="models/isolation_forest.pkl")
    else:
        mode = "rule_only"
        scene_paths = sys.argv[1:]
        pipeline = build_pipeline(mode="rule_only")

    print(f"Mode: {mode}")
    print(f"Scenes: {len(scene_paths)}")
    print("=" * 60)

    for path in scene_paths:
        result = pipeline.run(path)
        print(f"\nScene: {path}")
        print(f"  accepted: {result['accepted']}")
        print(f"  failed_filter: {result.get('failed_filter')}")
        if mode == "hybrid":
            print(f"  decision: {result.get('decision')}")
            print(f"  stage2_score: {result.get('stage2_score')}")
            print(f"  reason: {result.get('reason')}")