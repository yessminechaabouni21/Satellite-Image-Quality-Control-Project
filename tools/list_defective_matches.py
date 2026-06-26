from pathlib import Path

NOISE_SIGMAS   = [10, 50, 100, 250, 500, 1000]
BLUR_KERNELS   = [3, 7, 15, 21, 51]
STRIPE_INTENS  = [100, 500, 1000, 2000]

def get_all_expected_defect_types():
    types = []
    types.extend(["CORRUPTION_zero_50", "CORRUPTION_flat", "CORRUPTION_missing_B11"])
    types.extend([f"NOISE_{s}" for s in NOISE_SIGMAS])
    types.extend([f"BLUR_{k}" for k in BLUR_KERNELS])
    types.extend([f"STRIPE_{i}" for i in STRIPE_INTENS])
    return types

def find_base_scenes(extracted_dir):
    p = Path(extracted_dir)
    return sorted(p.glob('*.SAFE'))

defective_dir = Path('data/defective')
base_scenes = find_base_scenes('data/extracted')
expected = set(get_all_expected_defect_types())

matches = []
for item in sorted(defective_dir.iterdir()):
    if not item.is_dir():
        continue
    name = item.name
    matched = None
    for et in expected:
        if et in name:
            matched = et
            break
    bs_matched = None
    for bs in base_scenes:
        if bs.name.replace('.SAFE','') in name:
            bs_matched = bs.name
            break
    matches.append((name, matched, bs_matched))

print(f"Total folders: {len(matches)}\n")
for n,m,b in matches:
    print(f"{n} | defect={m} | base={b}")

matched_count = sum(1 for _,m,_ in matches if m)
print(f"\nMatched defect types: {matched_count}/{len(matches)}")
matched_base = sum(1 for _,m,b in matches if m and b)
print(f"Matched both defect+base: {matched_base}/{len(matches)}")
