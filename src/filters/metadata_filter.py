import xml.etree.ElementTree as ET
from pathlib import Path
from .base import BaseFilter, FilterResult


class MetadataFilter(BaseFilter):
    def __init__(self, max_cloud=40):
        super().__init__("MetadataFilter")
        self.max_cloud = max_cloud

    def _apply(self, scene_path, context):
        scene_path = Path(scene_path)

        xml_files = list(scene_path.rglob("MTD_MSIL1C.xml"))

        if not xml_files:
            xml_files = list(scene_path.rglob("*MTD*.xml"))

        if not xml_files:
            return FilterResult(False, "Missing Sentinel metadata file")

        xml_path = xml_files[0]

        tree = ET.parse(xml_path)
        root = tree.getroot()

        cloud = root.find(".//Cloud_Coverage_Assessment")

        if cloud is None:
            return FilterResult(False, "No cloud metadata")

        cloud = float(cloud.text)

        passed = cloud <= self.max_cloud

        return FilterResult(
            passed=passed,
            reason=None if passed else f"Cloud too high: {cloud}",
            metrics={"cloud_cover": cloud}
        )