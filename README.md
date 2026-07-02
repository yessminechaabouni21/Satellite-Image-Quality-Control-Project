# Sentinel-2 EO Quality Control Pipeline

Automated quality control system for Sentinel-2 L1C satellite imagery over tile **T32SPF** (Tunisia/Mediterranean). Combines rule-based filtering, statistical anomaly detection, and machine learning to detect defective scenes and classify them as accepted or rejected.

---

## Project Overview

Earth Observation (EO) satellites produce thousands of scenes per day. Not all are usable — some are affected by cloud cover, sensor noise, geometric mis-registration, or radiometric processing failures. This project builds a complete QC pipeline that:

1. **Detects pixel-level defects** (noise, blur, stripes, missing bands, no-data) using 7 rule-based filters
2. **Validates against real ESA ground truth** using official QC quality flags from 200 real Sentinel-2 acquisitions
3. **Extends detection** using statistical feature analysis and a radiometric anomaly composite score
4. **Delivers a single callable function** (`predict()`) that accepts/rejects any new scene with a structured explanation

---

## Results Summary

| Method | Recall | FPR | F1 | AUC |
|---|---|---|---|---|
| Rule-based (7 filters) | 25% | 7% | 0.31 | N/A |
| Isolation Forest | 12% | 5% | 0.18 | 0.57 |
| Random Forest (LOO) | 12% | 10% | 0.15 | 0.38 |
| **Radiometric composite** | **50%** | **3%** | **0.60** | **0.81** |

**Key finding:** Generic ML underperforms with only 8 real positive examples. Statistical feature selection (Mann-Whitney U, p=0.001) identified DN-distribution width as the only significant discriminator of real ESA `general_quality` failures, enabling a transparent 2-feature composite that outperforms all other methods.

**Coverage gap:** `geometric_quality` failures and 3/6 `general_quality` failures produce no detectable pixel-level or metadata signal in the distributed L1C product — detection would require ESA internal processing logs.

---

## Project Structure

```
src/
├── predict.py                    ← FINAL DELIVERABLE: single predict() function
├── validate_predict.py           ← validates deliverable against ESA ground truth
├── run_esa_scenes.py             ← runs pipeline on all ESA reference scenes
│
├── filters/
│   ├── metadata_filter.py        ← cloud cover check
│   ├── missing_bands_filter.py   ← required bands present
│   ├── toascaling_filter.py      ← DN range / radiometric validity
│   ├── noData_filter.py          ← no-data / zero-pixel ratio
│   ├── blur_filter.py            ← Laplacian variance sharpness check
│   ├── stripe_filter.py          ← FFT periodic artifact detection
│   ├── noise_filter.py           ← noise standard deviation ratio
│   ├── geometric_filter.py       ← phase cross-correlation (not in pipeline — see limitations)
│   └── metadata_quality_filter.py ← DEGRADED_ANC/MSI from XML (not in pipeline — see limitations)
│
├── pipeline/
│   └── orchestrator.py           ← sequential filter runner with short-circuit
│
├── defects/
│   └── test_defect.py            ← synthetic defect injection + severity sweep
│
├── real_world/
│   ├── esa_reference_T32SPF.py   ← queries CDSE for real ESA quality flags
│   ├── download_failed_scenes.py ← downloads ESA-FAILED scenes only
│   ├── download_passed_sample.py ← stratified sample of ESA-PASSED scenes
│   └── confusion_matrix.py       ← pipeline vs ESA ground truth evaluation
│
└── ml/
    ├── build_feature_table.py    ← extracts per-scene metrics into ml_features.csv
    ├── diagnose_separation.py    ← Mann-Whitney U feature significance test
    ├── calibrate_thresholds.py   ← data-driven threshold recommendations
    ├── radiometric_composite.py  ← 2-feature z-score anomaly detector (best result)
    ├── train_isol_for.py         ← Isolation Forest (one-class)
    ├── train_random_for.py       ← Random Forest (supervised, LOO CV)
    └── final_comparison.py       ← honest 4-method comparison on same test set

reports/
├── eo_qc.db                      ← SQLite: all pipeline decisions + ESA labels
├── esa_reference_T32SPF.csv      ← 200 ESA scenes with PASSED/FAILED flags
├── ml_features.csv               ← per-scene feature table (145 scenes, 16 features)
├── predict_validation.csv        ← deliverable validation results
├── ml/
│   ├── final_comparison.png      ← main result: 4-method comparison
│   ├── radiometric_composite.png ← best detector: scatter + score distribution + ROC
│   ├── threshold_calibration.png ← evidence for threshold choices
│   ├── isolation_forest_results.png
│   └── random_forest_results.png
└── visuals/
    ├── defect_visualization*.png ← visual examples of injected defects
    ├── all_scenes_qc_comparison.png
    └── qc_2x2_comparison.png
```

---

## Setup

```bash
# Clone and install dependencies
git clone https://github.com/yessminechaabouni21/Satellite-Image-Quality-Control-Project.git
cd Satellite-Image-Quality-Control-Project
pip install -r requirements.txt
```

**Required packages:**
```
rasterio
numpy
pandas
scipy
scikit-learn
scikit-image
matplotlib
requests
```

**CDSE credentials** (for downloading real scenes):
```bash
export CDSE_USERNAME="your@email.com"
export CDSE_PASSWORD="yourpassword"
```

---

## Usage

### Run the final deliverable on a single scene

```python
from src.predict import predict

result = predict("data/esa_reference/S2B_MSIL1C_....SAFE")
print(result["accepted"])   # True / False
print(result["reason"])     # human-readable explanation
```

Or from the command line:

```bash
python -m src.predict path/to/scene.SAFE
```

**Output structure:**
```python
{
    "scene": "S2B_MSIL1C_....SAFE",
    "accepted": False,
    "reason": "Rejected by rule-based pipeline: TOAScalingFilter — ...",
    "rule_based": {
        "accepted": False,
        "failed_filter": "TOAScalingFilter",
        "failure_reason": "..."
    },
    "radiometric_composite": {
        "z_score": 1.93,
        "flagged": True,
        "dn_range": 24150.0,
        "max_dn": 27800
    },
    "filter_details": {...}    # full per-filter metrics
}
```

### Validate the deliverable against ESA ground truth

```bash
python -m src.validate_predict
```

Runs `predict()` on every downloaded ESA scene with known labels and prints a confusion matrix.

---

## Reproducing the Results

Run these scripts in order to reproduce everything from scratch:

```bash
# 1. Query ESA quality flags (requires CDSE credentials)
python -m src.real_world.esa_reference_T32SPF \
    --start 2024-01-01 --max-products 200 --write-db \
    --username $CDSE_USERNAME --password $CDSE_PASSWORD

# 2. Download the 8 ESA-FAILED scenes
python -m src.real_world.download_failed_scenes \
    --username $CDSE_USERNAME --password $CDSE_PASSWORD

# 3. Download a stratified sample of ESA-PASSED scenes
python -m src.real_world.download_passed_sample \
    --username $CDSE_USERNAME --password $CDSE_PASSWORD --n 40

# 4. Run synthetic defect injection (takes ~2 hours, needs disk space)
python -m src.defects.test_defect

# 5. Build the ML feature table
python -m src.ml.build_feature_table

# 6. Diagnose which features carry real signal
python -m src.ml.diagnose_separation

# 7. Train models
python -m src.ml.train_isol_for --source esa_ref --features validated
python -m src.ml.train_random_for --source esa_ref --features validated

# 8. Generate the radiometric composite (best detector)
python -m src.ml.radiometric_composite

# 9. Final comparison
python -m src.ml.final_comparison --features validated

# 10. Validate the deliverable
python -m src.validate_predict
```

---

## Pipeline Configuration

Filter thresholds are calibrated from real ESA-PASSED T32SPF scenes (see `reports/ml/threshold_calibration.png`):

| Filter | Threshold | Basis |
|---|---|---|
| MetadataFilter | cloud ≤ 60% | domain knowledge |
| TOAScalingFilter | pct_above_ceiling ≤ 5% | valid DN range [1000, 11200] for baseline ≥ 04.00 |
| NoDataFilter | unexpected_nodata ≤ 5% | P99 of PASSED scenes |
| BlurFilter | variance ≥ 6.827 | P1 of PASSED scenes |
| StripeFilter | periodic_power_ratio ≤ 0.30 | P99 of PASSED scenes |
| NoiseFilter | noise_std_ratio ≤ 0.15 | P99 of PASSED scenes |

---

## Limitations

- **8 real positive examples** — the ESA FAILED test set is too small for reliable supervised learning; LOO cross-validation was used to prevent leakage
- **Tile-specific** — thresholds calibrated on T32SPF; other tiles need recalibration
- **Single band** — all filters operate on B04; inter-band defects partially captured only by NIR/Red ratio
- **Coverage gap** — 3/8 `general_quality` failures and 2/2 `geometric_quality` failures produce no detectable pixel-level or metadata signal in the L1C product
- **Geometric filter not deployed** — phase cross-correlation implemented but not included in the production pipeline; geometric failures in this dataset coincide with high cloud cover (caught by MetadataFilter), making the filter redundant on T32SPF
- **MetadataQualityFilter not deployed** — `DEGRADED_ANC_DATA_PERCENTAGE = 0.0` for all tested scenes despite ESA flagging them as `general_quality=FAILED`, suggesting ESA's internal threshold differs from the publicly exposed value

---

## Database Schema

`reports/eo_qc.db` contains three tables:

**`scenes`** — clean scene pipeline decisions
**`defects`** — synthetic injection results (defect type, severity, caught/missed)
**`esa_reference`** — 200 real ESA scenes with official quality flags

---

## Tile Information

**Tile:** T32SPF (UTM Zone 32N)
**Location:** Tunisia / Mediterranean coast
**Sensor:** Sentinel-2 L1C (MSI instrument, 13 bands, 10m–60m GSD)
**Period:** January 2024 – June 2026
**Processing baseline:** 05.10 – 05.12 (RADIO_ADD_OFFSET = −1000 applied)