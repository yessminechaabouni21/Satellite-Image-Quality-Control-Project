import zipfile
from pathlib import Path

zip_dir = Path("data/raw")
out_dir = Path("data/extracted")
out_dir.mkdir(parents=True, exist_ok=True)

for zip_path in zip_dir.glob("*.zip"):
    print(f"Extracting {zip_path.name}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        extract_path = out_dir / zip_path.stem
        extract_path.mkdir(exist_ok=True)
        zip_ref.extractall(extract_path)

print("Done extracting all scenes.")