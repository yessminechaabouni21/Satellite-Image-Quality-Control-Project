# src/filters/duplicate_filter.py
import hashlib
from pathlib import Path
from .base import BaseFilter, FilterResult

class DuplicateFilter(BaseFilter):
    """
    Detects duplicate scenes by comparing metadata hash.
    """
    
    def __init__(self, processed_scenes: set = None):
        super().__init__()
        self.processed = processed_scenes or set()
    
    def apply(self, scene_path: str) -> FilterResult:
        # Hash of scene name + acquisition date
        scene_name = Path(scene_path).stem
        date_part = scene_name.split("_")[2]  # YYYYMMDDTHHMMSS
        
        # Extract tile + date (same area, same day = duplicate)
        tile = scene_name.split("_")[5]  # T32SPF
        key = f"{tile}_{date_part[:8]}"  # T32SPF_20260406
        
        passed = key not in self.processed
        
        if passed:
            self.processed.add(key)
        
        return FilterResult(
            passed=passed,
            reason=None if passed else f"Duplicate: {key} already processed",
            metrics={
                "scene_key": key,
                "is_duplicate": not passed
            }
        )