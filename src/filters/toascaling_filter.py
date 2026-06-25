# src/filters/toa_scaling_filter.py
import numpy as np
import rasterio
from pathlib import Path
from xml.etree import ElementTree as ET
from .base import BaseFilter, FilterResult


class TOAScalingFilter(BaseFilter):
    """
    Validates Sentinel-2 L1C TOA reflectance scaling.

    Baseline < 04.00  (pre Jan 2022):
        stored_DN = reflectance * QUANTIFICATION_VALUE
        valid range: [0, QUANTIFICATION_VALUE]   e.g. [0, 10000]

    Baseline >= 04.00 (post Jan 2022):
        stored_DN = reflectance * QUANTIFICATION_VALUE + abs(RADIO_ADD_OFFSET)
        RADIO_ADD_OFFSET is stored as a NEGATIVE number (e.g. -1000) in the XML,
        but is ADDED as a positive shift to the DN before storage.
        valid range: [abs(offset), QUANTIFICATION_VALUE + abs(offset)]
                     e.g. [1000, 11000]

    DN = 28000 on a baseline 05.x scene means reflectance > 1.7 — physically
    impossible and a real defect (saturated detector, corrupted data, or
    wrong quantification applied during processing).
    """

    def __init__(self, expected_quantification: int = 10000,
                 tolerance: int = 200):
        super().__init__()
        self.expected_quantification = expected_quantification
        self.tolerance = tolerance          # allow small overrun for sensor noise

    # ------------------------------------------------------------------
    def _parse_metadata(self, scene_path: Path) -> dict:
        mtd_files = list(scene_path.glob("MTD_MSIL1C.xml"))
        if not mtd_files:
            mtd_files = list(scene_path.rglob("MTD_MSIL1C.xml"))

        if not mtd_files:
            return {
                "quantification_value": self.expected_quantification,
                "radio_add_offset": 0,
                "baseline": None,
                "source": "fallback_no_mtd",
            }

        mtd_path = mtd_files[0]
        root = ET.parse(mtd_path).getroot()

        def first_int(tag, default):
            for el in root.iter(tag):          # bare iter ignores namespace
                try:
                    return int(float(el.text.strip()))
                except (ValueError, AttributeError):
                    pass
            return default

        def first_str(tag):
            for el in root.iter(tag):
                if el.text:
                    return el.text.strip()
            return None

        quant  = first_int("QUANTIFICATION_VALUE", self.expected_quantification)
        # RADIO_ADD_OFFSET is per-band (all identical for L1C); take the first one
        offset = first_int("RADIO_ADD_OFFSET", 0)   # e.g. -1000
        baseline_str = first_str("PROCESSING_BASELINE")
        try:
            baseline = float(baseline_str) if baseline_str else None
        except ValueError:
            baseline = None

        return {
            "quantification_value": quant,
            "radio_add_offset": offset,        # negative in XML, e.g. -1000
            "baseline": baseline,
            "source": mtd_path.name,
        }

    # ------------------------------------------------------------------
    def apply(self, scene_path: str) -> FilterResult:
        scene_path = Path(scene_path)

        band_files = list(scene_path.rglob("*_B04.jp2"))
        if not band_files:
            band_files = list(scene_path.rglob("*B04*.jp2"))
        if not band_files:
            return FilterResult(
                passed=False,
                reason="B04 band not found — check .SAFE structure",
                metrics={},
            )

        meta       = self._parse_metadata(scene_path)
        quant      = meta["quantification_value"]
        offset_xml = meta["radio_add_offset"]
        dn_shift   = abs(offset_xml)
        dn_ceiling = quant + dn_shift + self.tolerance   # e.g. 11200

        with rasterio.open(band_files[0]) as src:
            data    = src.read(1)
            profile = src.profile

        # All variables defined here, before any conditional return
        valid        = data[data > 0]
        max_val      = int(data.max())
        unique_count = int(np.unique(data).size)

        # Safe defaults in case valid is empty
        p9999     = float(np.percentile(valid, 99.99)) if valid.size >= 1000 else 0.0
        pct_above = float((valid > dn_ceiling).mean()) if valid.size > 0 else 0.0

        issues = []

        if valid.size < 1000:
            issues.append("Too few valid pixels — likely empty or edge tile")

        elif pct_above > 0.05:
            issues.append(
                f"{pct_above:.1%} of valid pixels exceed DN ceiling {dn_ceiling} "
                f"(quant={quant}, offset={offset_xml}, "
                f"baseline={meta['baseline']})"
            )

        if max_val == 0:
            issues.append("Band is entirely zero — corrupted or missing data")

        if unique_count < 10 and max_val > 0:
            issues.append(
                f"Only {unique_count} unique values — uniform or corrupted"
            )

        return FilterResult(
            passed=len(issues) == 0,
            reason="; ".join(issues) if issues else None,
            metrics={
                "band_file":            band_files[0].name,
                "dtype":                str(profile["dtype"]),
                "min_dn":               int(data.min()),
                "max_dn":               max_val,
                "dn_p9999":             p9999,
                "pct_above_ceiling":    pct_above,
                "unique_values":        unique_count,
                "quantification_value": quant,
                "radio_add_offset_xml": offset_xml,
                "dn_ceiling":           dn_ceiling,
                "processing_baseline":  meta["baseline"],
                "metadata_source":      meta["source"],
            },
        )