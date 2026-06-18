from shapely.geometry import box
import json
from .base import BaseFilter, FilterResult

class AOIFilter(BaseFilter):

    def __init__(self, aoi_geojson, min_overlap=0.5):
        super().__init__("AOIFilter")
        self.aoi = box(*aoi_geojson)  # simple bbox
        self.min_overlap = min_overlap

    def _apply(self, scene_path, context):

        footprint = context.get("footprint")

        if footprint is None:
            return FilterResult(False, "No footprint provided")

        intersection = self.aoi.intersection(footprint).area
        union = self.aoi.union(footprint).area

        iou = intersection / union if union > 0 else 0

        passed = iou >= self.min_overlap

        return FilterResult(
            passed=passed,
            reason=None if passed else f"Low AOI overlap: {iou:.2f}",
            metrics={"iou": iou}
        )