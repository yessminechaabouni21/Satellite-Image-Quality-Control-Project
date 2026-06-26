# src/pipeline/hybrid_pipeline.py
"""
Hybrid QC Pipeline: Rule-based Stage 1 + ML Stage 2

Stage 1: Fast rule-based filters catch obvious defects (low FPR)
Stage 2: Isolation Forest scores borderline cases (high recall on subtle defects)

Output classes:
    REJECT  - Defective (either stage caught it)
    REVIEW  - Borderline (Stage 2 score 0.3-0.7, needs human inspection)
    ACCEPT  - Clean (passed both stages with low anomaly score)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle

from src.pipeline.orchestrator import Pipeline
from src.filters.metadata_filter import MetadataFilter
from src.filters.noData_filter import NoDataFilter
from src.filters.toascaling_filter import TOAScalingFilter
from src.filters.blur_filter import BlurFilter
from src.filters.noise_filter import NoiseFilter
from src.filters.missing_bands_filter import MissingBandsFilter
from src.filters.stripe_filter import StripeFilter

# Feature extraction for Stage 2
from src.features.extract_features import extract_scene_features


class HybridPipeline:
    """
    Two-stage quality control pipeline.

    Stage 1: Rule-based filters (fast, deterministic, explainable)
    Stage 2: Isolation Forest anomaly scoring (catches subtle defects)

    Parameters
    ----------
    if_model_path : str
        Path to saved Isolation Forest model (.pkl)
    review_threshold_low : float
        Lower bound for REVIEW zone (default 0.3)
    review_threshold_high : float
        Upper bound for REVIEW zone (default 0.7)
    """

    def __init__(
        self,
        if_model_path: str = "models/isolation_forest.pkl",
        review_threshold_low: float = 0.3,
        review_threshold_high: float = 0.7,
    ):
        self.review_low = review_threshold_low
        self.review_high = review_threshold_high

        # Stage 1: Rule-based pipeline
        self.rule_pipeline = Pipeline([
            MetadataFilter(max_cloud=60.0),
            MissingBandsFilter(),
            TOAScalingFilter(),
            NoDataFilter(max_unexpected_nodata_ratio=0.05, min_unique_values=100),
            BlurFilter(min_variance=15.0),
            StripeFilter(max_periodic_power_ratio=0.3),
            NoiseFilter(max_noise_std_ratio=0.15),
        ])

        # Stage 2: Load trained Isolation Forest
        self.if_model = self._load_model(if_model_path)

        # Feature scaler (fit on training data, saved with model)
        self.scaler = self._load_scaler(if_model_path)

    def _load_model(self, path: str):
        """Load trained Isolation Forest."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Isolation Forest model not found at {path}. "
                "Train first: python -m src.models.train_models"
            )
        with open(p, "rb") as f:
            return pickle.load(f)

    def _load_scaler(self, path: str):
        """Load feature scaler (saved alongside model)."""
        scaler_path = Path(path).with_suffix(".scaler.pkl")
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                return pickle.load(f)
        return None

    def run(self, scene_path: str) -> Dict:
        """
        Run hybrid pipeline on a single scene.

        Returns
        -------
        dict with keys:
            - decision: "ACCEPT", "REVIEW", or "REJECT"
            - stage1_passed: bool
            - stage1_failed_filter: str or None
            - stage2_score: float (anomaly score, -1 to +1)
            - stage2_decision: str or None
            - reason: str (human-readable explanation)
            - features: dict (extracted features for logging)
        """
        result = {
            "scene": scene_path,
            "decision": None,
            "stage1_passed": False,
            "stage1_failed_filter": None,
            "stage2_score": None,
            "stage2_decision": None,
            "reason": "",
            "features": {},
        }

        # =====================================================================
        # STAGE 1: Rule-based filters
        # =====================================================================
        stage1_result = self.rule_pipeline.run(scene_path)
        result["stage1_passed"] = stage1_result["accepted"]

        if not stage1_result["accepted"]:
            # Stage 1 caught an obvious defect
            failed = next(
                (k for k, v in stage1_result["results"].items() if not v["passed"]),
                None
            )
            result["stage1_failed_filter"] = failed
            result["decision"] = "REJECT"
            result["reason"] = f"Stage 1 (Rule-based): Failed {failed}"
            return result

        # =====================================================================
        # STAGE 2: ML anomaly scoring
        # =====================================================================
        # Extract features for ML scoring
        features = extract_scene_features(scene_path)
        result["features"] = features

        # Prepare feature vector (same order as training)
        feature_vector = self._prepare_features(features)

        # Scale features
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
        else:
            feature_vector = feature_vector.reshape(1, -1)

        # Get anomaly score
        # decision_function: negative = anomaly, positive = normal
        # We negate so higher = more anomalous
        raw_score = self.if_model.decision_function(feature_vector)[0]
        anomaly_score = -raw_score  # Normalize to 0-1 range

        # Normalize to [0, 1] using sigmoid-like transform
        # decision_function outputs roughly [-0.5, +0.5]
        normalized_score = 1 / (1 + np.exp(-5 * anomaly_score))
        result["stage2_score"] = float(normalized_score)

        # Apply thresholds
        if normalized_score > self.review_high:
            result["stage2_decision"] = "REJECT"
            result["decision"] = "REJECT"
            result["reason"] = (
                f"Stage 2 (ML): Anomaly score {normalized_score:.3f} > "
                f"{self.review_high} (high confidence defect)"
            )
        elif normalized_score > self.review_low:
            result["stage2_decision"] = "REVIEW"
            result["decision"] = "REVIEW"
            result["reason"] = (
                f"Stage 2 (ML): Anomaly score {normalized_score:.3f} in "
                f"[{self.review_low}, {self.review_high}] (borderline, human review)"
            )
        else:
            result["stage2_decision"] = "ACCEPT"
            result["decision"] = "ACCEPT"
            result["reason"] = (
                f"Stage 2 (ML): Anomaly score {normalized_score:.3f} < "
                f"{self.review_low} (clean)"
            )

        return result

    def _prepare_features(self, features: Dict) -> np.ndarray:
        """Convert feature dict to numpy array in training order."""
        # This must match the feature order used during training
        # Extract from your feature_table.csv column order
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
            if val is None or np.isnan(val):
                val = 0.0
            vec.append(float(val))

        return np.array(vec)

    def batch_process(self, scene_paths: list) -> pd.DataFrame:
        """Process multiple scenes and return results dataframe."""
        results = []
        for path in scene_paths:
            r = self.run(path)
            results.append({
                "scene": r["scene"],
                "decision": r["decision"],
                "stage1_passed": r["stage1_passed"],
                "stage1_failed_filter": r["stage1_failed_filter"],
                "stage2_score": r["stage2_score"],
                "reason": r["reason"],
            })
        return pd.DataFrame(results)


# =====================================================================
# Integration with existing orchestrator
# =====================================================================

class HybridOrchestrator(Pipeline):
    """
    Drop-in replacement for Pipeline that adds Stage 2 ML scoring.

    Usage:
        from src.pipeline.hybrid_pipeline import HybridOrchestrator

        pipeline = HybridOrchestrator()
        result = pipeline.run("path/to/scene.SAFE")
        # result now includes ML scoring for scenes that pass rules
    """

    def __init__(self, if_model_path: str = "models/isolation_forest.pkl"):
        # Initialize base rule-based pipeline
        super().__init__([
            MetadataFilter(max_cloud=60.0),
            MissingBandsFilter(),
            TOAScalingFilter(),
            NoDataFilter(max_unexpected_nodata_ratio=0.05, min_unique_values=100),
            BlurFilter(min_variance=15.0),
            StripeFilter(max_periodic_power_ratio=0.3),
            NoiseFilter(max_noise_std_ratio=0.15),
        ])

        # Initialize Stage 2
        self.hybrid = HybridPipeline(if_model_path=if_model_path)

    def run(self, scene_path: str) -> Dict:
        """Run hybrid pipeline and return enriched result."""
        return self.hybrid.run(scene_path)


# =====================================================================
# CLI entry point
# =====================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.hybrid_pipeline <scene.SAFE> [scene2.SAFE ...]")
        sys.exit(1)

    pipeline = HybridPipeline()

    for scene_path in sys.argv[1:]:
        print(f"\n{'='*60}")
        print(f"Processing: {scene_path}")
        print(f"{'='*60}")

        result = pipeline.run(scene_path)

        print(f"Decision: {result['decision']}")
        print(f"Stage 1 passed: {result['stage1_passed']}")
        if result['stage1_failed_filter']:
            print(f"Stage 1 failed filter: {result['stage1_failed_filter']}")
        if result['stage2_score'] is not None:
            print(f"Stage 2 anomaly score: {result['stage2_score']:.3f}")
        print(f"Reason: {result['reason']}")