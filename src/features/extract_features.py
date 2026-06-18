# src/features/extract_features.py
import numpy as np
import rasterio
from pathlib import Path
import pandas as pd


def scale_to_reflectance(dn_array: np.ndarray) -> np.ndarray:
    return dn_array.astype(np.float32) / 10000.0


def read_band_at_10m(band_path: str, target_shape: tuple) -> np.ndarray:
    """Read band and resample to target shape (10m resolution)."""
    with rasterio.open(band_path) as src:
        return scale_to_reflectance(src.read(1, out_shape=target_shape))


def compute_ndvi(b04_path: str, b08_path: str, target_shape: tuple) -> dict:
    """Compute NDVI — both bands at 10m."""
    red = read_band_at_10m(b04_path, target_shape)
    nir = read_band_at_10m(b08_path, target_shape)
    
    valid = (red > 0) & (nir > 0)
    ndvi = np.full_like(red, np.nan)
    ndvi[valid] = (nir[valid] - red[valid]) / (nir[valid] + red[valid] + 1e-8)
    
    return {
        "ndvi_mean": float(np.nanmean(ndvi)),
        "ndvi_std": float(np.nanstd(ndvi)),
        "ndvi_min": float(np.nanmin(ndvi)),
        "ndvi_max": float(np.nanmax(ndvi))
    }


def compute_ndwi(b03_path: str, b08_path: str, target_shape: tuple) -> dict:
    """Compute NDWI — both bands at 10m."""
    green = read_band_at_10m(b03_path, target_shape)
    nir = read_band_at_10m(b08_path, target_shape)
    
    valid = (green > 0) & (nir > 0)
    ndwi = np.full_like(green, np.nan)
    ndwi[valid] = (green[valid] - nir[valid]) / (green[valid] + nir[valid] + 1e-8)
    
    return {
        "ndwi_mean": float(np.nanmean(ndwi)),
        "ndwi_std": float(np.nanstd(ndwi))
    }


def compute_ndbi(b11_path: str, b08_path: str, target_shape: tuple) -> dict:
    """Compute NDBI — B11 resampled to 10m."""
    swir = read_band_at_10m(b11_path, target_shape)
    nir = read_band_at_10m(b08_path, target_shape)
    
    valid = (swir > 0) & (nir > 0)
    ndbi = np.full_like(swir, np.nan)
    ndbi[valid] = (swir[valid] - nir[valid]) / (swir[valid] + nir[valid] + 1e-8)
    
    return {
        "ndbi_mean": float(np.nanmean(ndbi)),
        "ndbi_std": float(np.nanstd(ndbi))
    }


def compute_evi(b02_path: str, b04_path: str, b08_path: str, target_shape: tuple) -> dict:
    """Compute EVI — all bands at 10m."""
    blue = read_band_at_10m(b02_path, target_shape)
    red = read_band_at_10m(b04_path, target_shape)
    nir = read_band_at_10m(b08_path, target_shape)
    
    valid = (blue > 0) & (red > 0) & (nir > 0)
    evi = np.full_like(red, np.nan)
    evi[valid] = 2.5 * (nir[valid] - red[valid]) / \
                 (nir[valid] + 6 * red[valid] - 7.5 * blue[valid] + 1 + 1e-8)
    
    return {
        "evi_mean": float(np.nanmean(evi)),
        "evi_std": float(np.nanstd(evi))
    }


def compute_texture(b04_path: str, target_shape: tuple) -> dict:
    """Compute GLCM texture from B04 at 10m."""
    from skimage.feature import graycomatrix, graycoprops
    
    red = read_band_at_10m(b04_path, target_shape)
    
    # Scale to 8-bit
    scaled = ((red - red.min()) / (red.max() - red.min() + 1e-8) * 255).astype(np.uint8)
    
    # GLCM
    glcm = graycomatrix(scaled, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    return {
        "texture_contrast": float(contrast),
        "texture_homogeneity": float(homogeneity)
    }


def extract_all_features(scene_path: str) -> dict:
    """Extract all features from a scene."""
    granule = list(Path(scene_path).rglob("GRANULE"))[0]
    
    def find_band(band_name):
        files = list(granule.rglob(f"*{band_name}.jp2"))
        return str(files[0]) if files else None
    
    b02 = find_band("B02")
    b03 = find_band("B03")
    b04 = find_band("B04")
    b08 = find_band("B08")
    b11 = find_band("B11")
    
    if not b04 or not b08:
        raise ValueError("Missing required B04 or B08")
    
    # Get target shape from B04 (10m)
    with rasterio.open(b04) as src:
        target_shape = (src.height, src.width)
    
    features = {"scene": Path(scene_path).name}
    
    # NDVI
    features.update(compute_ndvi(b04, b08, target_shape))
    
    # NDWI
    if b03 and b08:
        features.update(compute_ndwi(b03, b08, target_shape))
    
    # NDBI
    if b11 and b08:
        features.update(compute_ndbi(b11, b08, target_shape))
    
    # EVI
    if b02 and b04 and b08:
        features.update(compute_evi(b02, b04, b08, target_shape))
    
    # Texture
    if b04:
        features.update(compute_texture(b04, target_shape))
    
    return features


def extract_features_batch(scene_paths: list, output_csv: str = "dataset/features.csv"):
    all_features = []
    
    for scene_path in scene_paths:
        print(f"Extracting: {Path(scene_path).name}")
        try:
            features = extract_all_features(scene_path)
            all_features.append(features)
        except Exception as e:
            print(f"  Failed: {e}")
    
    df = pd.DataFrame(all_features)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    return df


if __name__ == "__main__":
    scenes = sorted(Path("data/extracted").glob("*.SAFE"))
    extract_features_batch([str(s) for s in scenes])