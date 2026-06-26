#!/usr/bin/env python3
"""
standalone_audit.py — Quick audit of defective scenes without importing project modules.

Usage:
    python standalone_audit.py [data/defective] [data/extracted]

This script scans your defective output directory and tells you exactly:
- How many base scenes you have
- How many defect types exist per base scene  
- Which defect types are missing
- Completion percentage
"""

import sys
from pathlib import Path
from collections import defaultdict

# These must match the severities in test_defect.py
NOISE_SIGMAS   = [10, 50, 100, 250, 500, 1000]
BLUR_KERNELS   = [3, 7, 15, 21, 51]
STRIPE_INTENS  = [100, 500, 1000, 2000]


def get_all_expected_defect_types():
    """Return the complete list of expected defect type names."""
    types = []
    types.extend(["CORRUPTION_zero_50", "CORRUPTION_flat", "CORRUPTION_missing_B11"])
    types.extend([f"NOISE_{s}" for s in NOISE_SIGMAS])
    types.extend([f"BLUR_{k}" for k in BLUR_KERNELS])
    types.extend([f"STRIPE_{i}" for i in STRIPE_INTENS])
    return types


def find_base_scenes(extracted_dir):
    """Find all .SAFE folders in the extracted directory."""
    p = Path(extracted_dir)
    if not p.exists():
        print(f"❌ Extracted directory not found: {extracted_dir}")
        return []
    return sorted(p.glob("*.SAFE"))


def audit_defects(defective_dir, base_scenes):
    """
    Scan defective_dir for existing defective scenes.
    Returns: dict {base_scene_name: set(existing_defect_types)}
    """
    existing = {bs.name: set() for bs in base_scenes}
    out_path = Path(defective_dir)

    if not out_path.exists():
        print(f"❌ Defective directory not found: {defective_dir}")
        return existing

    expected_types = set(get_all_expected_defect_types())

    print(f"Scanning {out_path} for defective scenes...")
    found_files = 0

    for item in out_path.iterdir():
        if not item.is_dir() or not item.suffix == ".SAFE":
            continue
        found_files += 1
        name = item.name  # e.g. "BLUR_15_S2A_... .SAFE"

        # Locate which base scene this belongs to
        matched_bs = None
        for bs in base_scenes:
            bs_stem = bs.name.replace(".SAFE", "")
            if bs_stem in name:
                matched_bs = bs
                break
        if not matched_bs:
            continue

        # Extract the defect token by removing the base scene substring
        bs_stem = matched_bs.name.replace(".SAFE", "")
        defect_part = name.replace(bs_stem, "").replace('.SAFE', '').strip('_')

        # Normalize a few alternate naming schemes used by the generators
        # e.g. FLAT_5000 -> CORRUPTION_flat, ZERO_0.5 -> CORRUPTION_zero_50,
        # HAZE_0.2 -> HAZE_light, HAZE_0.5 -> HAZE_heavy
        canonical = None
        if defect_part.startswith("FLAT"):
            canonical = "CORRUPTION_flat"
        elif defect_part.startswith("ZERO"):
            canonical = "CORRUPTION_zero_50"
        elif defect_part.startswith("MISSING_B11") or defect_part.startswith("MISSING"):
            canonical = "CORRUPTION_missing_B11"
        elif defect_part.startswith("HAZE"):
            if "0.2" in defect_part:
                canonical = "HAZE_light"
            else:
                canonical = "HAZE_heavy"
        else:
            canonical = defect_part

        # Record only canonical defect types that we expect
        if (canonical in expected_types or canonical.startswith("HAZE")
            or canonical.startswith("CORRUPTION") or canonical.startswith("NOISE_")):
            existing[matched_bs.name].add(canonical)

    print(f"  Found {found_files} .SAFE folders")
    return existing


def print_audit_report(existing, base_scenes):
    """Print a detailed audit report."""
    expected_types = get_all_expected_defect_types()
    total_expected = len(base_scenes) * len(expected_types)
    total_found = sum(len(v) for v in existing.values())

    print("\n" + "=" * 80)
    print("📊 DEFECT AUDIT REPORT")
    print("=" * 80)
    print(f"Base scenes:        {len(base_scenes)}")
    print(f"Defect types/scene: {len(expected_types)}")
    print(f"Total expected:     {total_expected}")
    print(f"Total found:        {total_found}")
    print(f"Missing:            {total_expected - total_found}")
    print(f"Completion:         {total_found/total_expected*100:.1f}%")
    print("=" * 80)

    # Per-base-scene breakdown
    print("\n📁 PER BASE SCENE:")
    print("-" * 80)
    complete_scenes = 0
    for bs in base_scenes:
        bs_name = bs.name
        found = existing.get(bs_name, set())
        missing = set(expected_types) - found
        if not missing:
            complete_scenes += 1
            status = "✅ COMPLETE"
        else:
            status = f"❌ Missing {len(missing)}"
        print(f"  {bs_name[:50]:50s} | {len(found):2d}/{len(expected_types)} | {status}")
        if missing:
            for m in sorted(missing):
                print(f"      → {m}")

    print(f"\n  Complete scenes: {complete_scenes}/{len(base_scenes)}")

    # Per-defect-type breakdown
    print("\n🔬 PER DEFECT TYPE (across all base scenes):")
    print("-" * 80)
    incomplete_types = []
    for dtype in expected_types:
        count = sum(1 for bs in base_scenes if dtype in existing.get(bs.name, set()))
        status = "✅" if count == len(base_scenes) else "❌"
        print(f"  {status} {dtype:25s}: {count:2d}/{len(base_scenes)} scenes")
        if count < len(base_scenes):
            incomplete_types.append((dtype, len(base_scenes) - count))

    # Summary of missing types
    print("\n⚠️  MISSING DEFECT TYPES SUMMARY:")
    print("-" * 80)
    if incomplete_types:
        for dtype, missing_count in sorted(incomplete_types, key=lambda x: -x[1]):
            print(f"  {dtype:25s}: missing from {missing_count} base scene(s)")
            # Show which scenes are missing it
            missing_scenes = [bs.name for bs in base_scenes 
                             if dtype not in existing.get(bs.name, set())]
            for ms in missing_scenes[:3]:
                print(f"      → {ms[:70]}")
            if len(missing_scenes) > 3:
                print(f"      ... and {len(missing_scenes) - 3} more")
    else:
        print("  🎉 ALL DEFECT TYPES COMPLETE!")

    print("=" * 80)
    return total_found, total_expected, incomplete_types


def main():
    # Default paths
    defective_dir = sys.argv[1] if len(sys.argv) > 1 else "data/defective"
    extracted_dir = sys.argv[2] if len(sys.argv) > 2 else "data/extracted"

    print("=" * 80)
    print("STANDALONE DEFECT AUDIT TOOL")
    print("=" * 80)
    print(f"Defective dir: {defective_dir}")
    print(f"Extracted dir: {extracted_dir}")

    base_scenes = find_base_scenes(extracted_dir)
    if not base_scenes:
        print("\nNo base scenes found. Cannot audit.")
        return 1

    existing = audit_defects(defective_dir, base_scenes)
    total_found, total_expected, incomplete = print_audit_report(existing, base_scenes)

    # Exit code: 0 if complete, 1 if missing
    if total_found == total_expected:
        print("\n✅ AUDIT PASSED: All defects present!")
        return 0
    else:
        print(f"\n❌ AUDIT FAILED: {total_expected - total_found} defects missing")
        print("\nTo generate missing defects, run:")
        print("  python -m src.defects.test_defect")
        print("(with RESUME_MODE = True in the script)")
        return 1


if __name__ == "__main__":
    sys.exit(main())