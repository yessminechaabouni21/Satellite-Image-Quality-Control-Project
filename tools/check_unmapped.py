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
    types.extend(["HAZE_light","HAZE_heavy"]) 
    return types

expected = set(get_all_expected_defect_types())

base_scenes = sorted(Path('data/extracted').glob('*.SAFE'))

unmapped = []
for item in sorted(Path('data/defective').iterdir()):
    if not item.is_dir() or item.suffix != '.SAFE':
        continue
    name = item.name
    matched_bs = None
    for bs in base_scenes:
        bs_stem = bs.name.replace('.SAFE','')
        if bs_stem in name:
            matched_bs = bs
            break
    if not matched_bs:
        unmapped.append((name,'no_base'))
        continue
    bs_stem = matched_bs.name.replace('.SAFE','')
    defect_part = name.replace(bs_stem, '').replace('.SAFE','').strip('_')
    canonical = None
    if defect_part.startswith('FLAT'):
        canonical = 'CORRUPTION_flat'
    elif defect_part.startswith('ZERO'):
        canonical = 'CORRUPTION_zero_50'
    elif defect_part.startswith('MISSING_B11') or defect_part.startswith('MISSING'):
        canonical = 'CORRUPTION_missing_B11'
    elif defect_part.startswith('HAZE'):
        if '0.2' in defect_part:
            canonical = 'HAZE_light'
        else:
            canonical = 'HAZE_heavy'
    else:
        canonical = defect_part
    if not (canonical in expected or canonical.startswith('HAZE') or canonical.startswith('CORRUPTION')):
        unmapped.append((name, canonical))

print('Unmapped or excluded folders:')
for u in unmapped:
    print(u)
print('Count:', len(unmapped))
