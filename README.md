# EO QC Pipeline

Earth observation quality control pipeline for Sentinel-2 `.SAFE` scenes. Applies sequential filters to detect defects including missing bands, noise, blur, stripes, haze, and radiometric anomalies.

## What is included

- `src/pipeline/orchestrator.py` — pipeline runner that applies filters sequentially and stops at the first failure
- `src/filters/` — filter implementations for scene quality checks
- `src/run_all_scenes.py` — process all clean scenes in `data/extracted` and generate a summary report
- `src/defects/test_defect.py` — inject synthetic defects and verify pipeline detection
- `src/visualization/visualize_defects.py` — visualize defective scenes at full resolution (100×100 pixel crops)
- `data/` — source data and generated outputs
- `reports/` — generated CSV reports and visualizations

## Pipeline filters

| Filter | Purpose | Catches |
|--------|---------|---------|
| `DuplicateFilter` | Detect repeated scenes or duplicate ingestion | Duplicate files |
| `MetadataFilter` | Check scene metadata | Cloud cover > threshold |
| `MissingBandsFilter` | Verify required Sentinel-2 bands | Missing B02, B03, B04, B08, B11 |
| `RadiometricResolutionFilter` | Check bit depth and value distribution | 8-bit instead of 12-bit |
| `TOAScalingFilter` | Validate TOA reflectance scaling | Wrong scaling, flat bands |
| `NoDataFilter` | Check for unexpected no-data | >5% no-data, <100 unique values |
| `BlurFilter` | Measure scene sharpness via Laplacian variance | Gaussian blur (k≥15) |
| `StripeFilter` | Detect periodic row/column stripes via FFT | Readout interference |
| `NoiseFilter` | Detect excessive sensor noise via SNR | σ≥500 Gaussian noise |
| `HazeFilter` | Estimate haze via dark object subtraction | Atmospheric haze |

**Note:** `HazeFilter` is currently disabled in the defect test pipeline due to false positives on vegetated scenes.

## Usage

### Process clean scenes

```powershell
& .\.venv\Scripts\Activate.ps1
python -m src.run_all_scenes
```

Output: `reports/pipeline_report.csv` with accept/reject status per scene.

### Test defect detection

```powershell
python -m src.defects.test_defect
```

Injects 11 synthetic defect types and verifies which are caught. Output: `reports/defect_injection_results.csv`.

### Visualize defects

```powershell
python -m src.visualization.visualize_defects
```

Shows 100×100 full-resolution crops of each defective scene. Automatically splits into multiple figures if >9 scenes.

## Defect injection catalog

| Defect | Injection | Detected by |
|--------|-----------|-------------|
| `CORRUPTION_zero_50` | 50% of B04 set to zero | `NoDataFilter` |
| `CORRUPTION_flat` | All B04 set to 5000 DN | `TOAScalingFilter` |
| `CORRUPTION_missing_B11` | B11 file deleted | `MissingBandsFilter` |
| `NOISE_moderate` | σ=500 Gaussian noise | `NoiseFilter` |
| `NOISE_severe` | σ=2000 Gaussian noise | `NoiseFilter` |
| `BLUR_moderate` | k=15 Gaussian blur | `BlurFilter` |
| `BLUR_severe` | k=51 Gaussian blur | `BlurFilter` |
| `STRIPE_light` | +500 DN every 10 rows | `StripeFilter` |
| `STRIPE_heavy` | +2000 DN every 10 rows | `StripeFilter` |

**Design note:** Moderate defects (σ=100, k=7) are intentionally below rejection thresholds — they represent degraded but usable data. The pipeline targets **severe defects** that invalidate scientific use.

## Project structure

```
eo_project/
├── data/
│   ├── extracted/          # Clean Sentinel-2 .SAFE scenes
│   └── defective/          # Generated defect scenes (gitignored)
├── reports/
│   ├── pipeline_report.csv
│   ├── defect_injection_results.csv
│   └── visuals/            # Generated PNG figures (gitignored)
├── src/
│   ├── filters/            # All filter classes
│   ├── pipeline/           # Orchestrator
│   ├── defects/            # Injection utilities + test runner
│   ├── visualization/      # Plotting utilities
│   ├── run_all_scenes.py   # Clean scene pipeline
│   └── run_scenes_test.py  # Generic test runner
└── .gitignore              # Excludes outputs/ data/defective/ reports/visuals/
```

## Important implementation notes

- **JPEG2000 write limitation:** Defect injection writes modified bands back to `.jp2` files. Some GDAL builds do not support JP2 write — the code falls back to GeoTIFF conversion internally.
- **Full-resolution visualization:** The visualization script reads 100×100 pixel crops at **native 10m resolution** using `rasterio.windows.Window`. Do not use `out_shape` downsampling — it destroys defect visibility.
- **Pipeline order matters:** Filters are applied sequentially. Place structural checks (`Metadata`, `MissingBands`) first and sensitive detectors (`Noise`, `Stripe`) last to avoid false positives blocking later filters.
- **Module imports:** The project uses absolute imports (`src.filters.xxx`). Always run as `python -m src.module_name`, not `python src/module.py`.

## Next steps

- [ ] Add temporal consistency filter (compare against previous acquisition)
- [ ] Implement band-to-band registration check
- [ ] Add compression artifact detection
- [ ] Build Streamlit dashboard for interactive results review
- [ ] Add unit tests with pytest for each filter
- [ ] Optimize filter performance with parallel scene processing
