# EO QC Pipeline

This repository contains an Earth observation quality control pipeline for Sentinel-2 `.SAFE` scenes. The pipeline extracts scene metadata and applies a series of filters to detect issues such as missing bands, radiometric problems, haze, blur, noise, and duplicate scenes.

## What is included

- `src/pipeline/orchestrator.py` - simple pipeline runner that applies filters sequentially and stops at the first failed filter.
- `src/filters/` - filter implementations for scene quality checks and preprocessing.
- `src/run_all_scenes.py` - module entrypoint for processing all scenes in `data/extracted` and generating a results summary.
- `src/run_scenes_test.py` - test runner for processing a specified directory and saving a CSV report.
- `src/defects/` - utilities for generating defective scene variants for pipeline testing.
- `data/` - source data directories and generated outputs.
- `reports/` - generated report CSV files.

## Pipeline filters

The current pipeline includes the following filters:

- `DuplicateFilter` - detects repeated scenes or duplicate ingestion.
- `MetadataFilter` - checks scene metadata such as cloud cover.
- `MissingBandsFilter` - verifies that required Sentinel-2 bands are present.
- `RadiometricResolutionFilter` - checks radiometric bit depth and value distribution.
- `TOAScalingFilter` - validates top-of-atmosphere scaling and output dtype.
- `NoDataFilter` - checks for unexpected no-data pixels and low value diversity.
- `HazeFilter` - estimates haze contamination using a haze score.
- `BlurFilter` - measures scene sharpness.
- `NoiseFilter` - detects sensor noise and dead pixel patterns.

## Usage

Activate your Python virtual environment and run the main pipeline:

```powershell
& .\.venv\Scripts\Activate.ps1
python -m src.run_all_scenes
```

Generate a report for a specific directory of defective scenes:

```powershell
python -m src.run_scenes_test --dir data/defective --output reports/defect_report.csv
```

## Project structure

- `data/extracted/` - extracted Sentinel-2 `.SAFE` scene directories.
- `data/defective/` - generated defect scenes for QA testing.
- `src/filters/` - all scene quality filter classes.
- `src/pipeline/` - pipeline orchestration code.
- `src/defects/` - defect injection utilities.
- `src/visualization/` - visualization helpers for scene inspection.

## Notes

- The repository is structured to run as a package entrypoint (for example `python -m src.run_all_scenes`).
- Import paths use `src.filters.<module_name>` and filter filenames must match these module names.
- The pipeline currently returns a pandas DataFrame with scene-level metrics and accepts/rejects status.

## Next steps

Possible future improvements:

- add end-to-end dataset report generation
- implement calibration checks for more sensor bands
- add visualization scripts for accepted vs rejected scenes
- add structured unit tests for each filter
