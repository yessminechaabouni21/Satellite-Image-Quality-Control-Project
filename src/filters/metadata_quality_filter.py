# src/filters/metadata_quality_filter.py
#
# Reads DEGRADED_ANC_DATA_PERCENTAGE and DEGRADED_MSI_DATA_PERCENTAGE from
# the scene's MTD_MSIL1C.xml — these are ESA's own internal quality metrics,
# written into the product at processing time.
#
# Scientific rationale:
#   DEGRADED_ANC_DATA_PERCENTAGE  — fraction of ancillary data (orbit,
#       attitude, GPS, GNSS) that was degraded during the L1C processing.
#       Non-zero values indicate the geometric and radiometric corrections
#       were applied with imperfect inputs. ESA uses this internally to
#       determine whether to set general_quality=FAILED.
#
#   DEGRADED_MSI_DATA_PERCENTAGE  — fraction of raw MSI data packets that
#       arrived at ground with errors. Non-zero values indicate actual
#       data-loss during downlink, which can manifest as striping, missing
#       scan lines, or corrupted DN values.
#
# These are the most direct, most reliable signals available in the L1C
# product for general_quality failures — they are the upstream cause,
# while DN-range anomalies are a downstream symptom.
#
# Thresholds: ANY non-zero value for ANC is suspicious (typical good scenes
# are exactly 0.0). For MSI, even small values (>0.01%) indicate real packet
# loss that your pipeline's stripe/noise filters may or may not catch
# depending on severity.
#
import xml.etree.ElementTree as ET
from pathlib import Path

from .base import BaseFilter, FilterResult


class MetadataQualityFilter(BaseFilter):
    """
    Checks ESA's internal degradation percentages from MTD_MSIL1C.xml.
    Rejects scenes where ancillary or MSI data quality was degraded during
    L1C processing — a direct, upstream indicator of general_quality failures.
    """

    def __init__(self,
                 max_degraded_anc_pct: float = 0.0,
                 max_degraded_msi_pct: float = 0.01):
        """
        Args:
            max_degraded_anc_pct: Maximum allowed DEGRADED_ANC_DATA_PERCENTAGE.
                Default 0.0 — any ancillary degradation is flagged.
            max_degraded_msi_pct: Maximum allowed DEGRADED_MSI_DATA_PERCENTAGE.
                Default 0.01 — flags meaningful MSI packet loss while tolerating
                rounding noise (some scenes report 1e-5 even when visually clean).
        """
        super().__init__()
        self.max_degraded_anc_pct = max_degraded_anc_pct
        self.max_degraded_msi_pct = max_degraded_msi_pct

    def _parse_degradation(self, scene_path: Path):
        """Parse DEGRADED_* fields from MTD_MSIL1C.xml. Returns (anc_pct, msi_pct)."""
        mtd = list(scene_path.glob("MTD_MSIL1C.xml"))
        if not mtd:
            mtd = list(scene_path.rglob("MTD_MSIL1C.xml"))
        if not mtd:
            return None, None

        root = ET.parse(mtd[0]).getroot()

        anc_pct = None
        msi_pct = None

        for el in root.iter():
            tag = el.tag.split("}")[-1].upper()
            text = (el.text or "").strip()
            if not text:
                continue
            try:
                val = float(text)
            except ValueError:
                continue
            if tag == "DEGRADED_ANC_DATA_PERCENTAGE":
                anc_pct = val
            elif tag == "DEGRADED_MSI_DATA_PERCENTAGE":
                msi_pct = val

        return anc_pct, msi_pct

    def apply(self, scene_path: str) -> FilterResult:
        scene_path = Path(scene_path)
        anc_pct, msi_pct = self._parse_degradation(scene_path)

        if anc_pct is None and msi_pct is None:
            # MTD not found — pass through (other filters will catch structural issues)
            return FilterResult(
                passed=True,
                reason=None,
                metrics={
                    "degraded_anc_pct": None,
                    "degraded_msi_pct": None,
                    "mtd_found": False,
                },
            )

        issues = []

        if anc_pct is not None and anc_pct > self.max_degraded_anc_pct:
            issues.append(
                f"Ancillary data degraded: {anc_pct:.4f}% > {self.max_degraded_anc_pct}% "
                f"(orbit/attitude/GPS inputs to L1C processing were imperfect)"
            )

        if msi_pct is not None and msi_pct > self.max_degraded_msi_pct:
            issues.append(
                f"MSI data degraded: {msi_pct:.4f}% > {self.max_degraded_msi_pct}% "
                f"(raw sensor data lost or corrupted during downlink)"
            )

        return FilterResult(
            passed=len(issues) == 0,
            reason="; ".join(issues) if issues else None,
            metrics={
                "degraded_anc_pct": anc_pct,
                "degraded_msi_pct": msi_pct,
                "mtd_found": True,
            },
        )