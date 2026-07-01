# src/filters/geometric_filter.py
#
# Detects geometric mis-registration by comparing this scene to a known-clean
# reference scene of the same tile using phase cross-correlation.
#
# Scientific rationale:
#   geometric_quality=FAILED in ESA OLQC means the GCP-based co-registration
#   between this scene and the reference grid failed — the scene's pixels are
#   spatially offset compared to where they should be. A simple pixel-level
#   filter (noise, blur, stripe) cannot detect this because the image values
#   are radiometrically normal; only the *positions* are wrong.
#
#   Phase cross-correlation (Fourier-shift theorem) estimates the sub-pixel
#   translation between two images of the same scene:
#     - If registration is good: shift ≈ 0 px
#     - If registration has failed: shift can be several pixels or more
#   The error (shift magnitude in pixels) is the detection signal.
#
# Reference scene:
#   Must be a known-PASSED (clean) scene of the same tile. The reference
#   is stored as a downsampled numpy array to avoid repeated full-resolution
#   reads. You set the reference once at pipeline construction:
#       GeometricFilter(reference_scene_path="data/esa_reference/S2A_....SAFE")
#
# Limitations (state these in your report):
#   1. Needs a pre-specified reference scene of the SAME tile — not portable
#      to other tiles without changing the reference.
#   2. Detects only translational mis-registration (x/y shift). Rotational
#      or scale errors are not captured by this method.
#   3. Cloud cover or very different seasonal appearance (snow vs bare ground)
#      can reduce cross-correlation confidence. Use the `error` field (output
#      uncertainty) to detect low-confidence results.
#   4. Threshold (max_shift_pixels) is approximate — set empirically from
#      your known-good scenes (see PASSED-scene shift distribution below).
#
import numpy as np
import rasterio
from pathlib import Path

from .base import BaseFilter, FilterResult


def _load_band_downsampled(scene_path: Path, scale: int = 8):
    """
    Load B04 at 1/scale resolution.
    Full L1C is 10980×10980 px at 10m; at scale=8 → 1372×1372 px.
    This is fast enough for cross-correlation while preserving structure.
    Returns float32 array or None on failure.
    """
    band_files = list(scene_path.rglob("*_B04.jp2"))
    if not band_files:
        band_files = list(scene_path.rglob("*B04*.jp2"))
    if not band_files:
        return None

    with rasterio.open(band_files[0]) as src:
        h = max(1, src.height // scale)
        w = max(1, src.width  // scale)
        arr = src.read(1, out_shape=(h, w)).astype(np.float32)

    return arr


def _cross_corr_shift(ref: np.ndarray, scene: np.ndarray):
    """
    Estimate pixel shift between ref and scene using phase cross-correlation.
    Returns (shift_row, shift_col, error, phase_diff) via skimage.
    Falls back to scipy FFT-based method if skimage is unavailable.
    """
    try:
        from skimage.registration import phase_cross_correlation
        shift, error, phasediff = phase_cross_correlation(
            ref, scene,
            upsample_factor=4,      # sub-pixel accuracy to 0.25 px
            normalization="phase",
        )
        return float(shift[0]), float(shift[1]), float(error), float(phasediff)

    except ImportError:
        # Fallback: scipy FFT-based normalized cross-correlation
        # Less accurate (integer-pixel only) but no skimage dependency
        from scipy.signal import fftconvolve
        corr = fftconvolve(ref, scene[::-1, ::-1], mode="full")
        idx  = np.unravel_index(np.argmax(corr), corr.shape)
        shift_row = idx[0] - (ref.shape[0] - 1)
        shift_col = idx[1] - (ref.shape[1] - 1)
        return float(shift_row), float(shift_col), 0.0, 0.0


class GeometricFilter(BaseFilter):
    """
    Detects geometric mis-registration vs a known-clean reference scene.
    A large estimated shift (in pixels at the downsampled resolution) indicates
    that this scene's geographic registration has failed — consistent with
    ESA's geometric_quality=FAILED flag.
    """

    def __init__(self,
                 reference_scene_path: str,
                 max_shift_pixels: float = 2.0,
                 downsample_scale: int = 8):
        """
        Args:
            reference_scene_path: Path to a known-clean .SAFE scene of the
                SAME tile. Must have B04. This scene's downsampled B04 image
                is used as the spatial reference.
            max_shift_pixels: Maximum allowed shift magnitude (Euclidean,
                in pixels at the downsampled resolution). Each downsampled
                pixel represents `downsample_scale × 10m` on the ground —
                at scale=8 that's 80m per px, so max_shift_pixels=2.0 means
                a 160m maximum allowed mis-registration.
                Calibrate from your PASSED-scene shift distribution.
            downsample_scale: Factor by which to downsample before correlation.
                Higher = faster but less sensitive. Default 8 is a good balance.
        """
        super().__init__()
        self.max_shift_pixels = max_shift_pixels
        self.downsample_scale = downsample_scale

        # Load the reference once at construction time (not per-scene)
        self._ref_path = str(reference_scene_path)
        self._ref_band = None
        self._ref_error = None
        self._load_reference()

    def _load_reference(self):
        ref_path = Path(self._ref_path)
        if not ref_path.exists():
            self._ref_error = f"Reference scene not found: {self._ref_path}"
            return
        arr = _load_band_downsampled(ref_path, self.downsample_scale)
        if arr is None:
            self._ref_error = f"Could not find B04 in reference scene: {self._ref_path}"
            return
        # Normalize to zero-mean (improves cross-correlation robustness)
        arr = arr - arr.mean()
        self._ref_band = arr

    def apply(self, scene_path: str) -> FilterResult:
        scene_path = Path(scene_path)

        # If reference failed to load, pass through (don't reject everything)
        if self._ref_band is None:
            return FilterResult(
                passed=True,
                reason=None,
                metrics={
                    "shift_row_px": None,
                    "shift_col_px": None,
                    "shift_magnitude_px": None,
                    "shift_error": self._ref_error,
                    "reference": self._ref_path,
                },
            )

        # Load and normalize the scene band
        scene_arr = _load_band_downsampled(scene_path, self.downsample_scale)
        if scene_arr is None:
            return FilterResult(
                passed=False,
                reason="B04 band not found — cannot compute geometric registration",
                metrics={"shift_magnitude_px": None},
            )
        scene_arr = scene_arr - scene_arr.mean()

        # Match sizes (scenes should be identical resolution, but guard anyway)
        h = min(self._ref_band.shape[0], scene_arr.shape[0])
        w = min(self._ref_band.shape[1], scene_arr.shape[1])
        ref   = self._ref_band[:h, :w]
        scene = scene_arr[:h, :w]

        # Compute shift
        try:
            shift_row, shift_col, corr_error, _ = _cross_corr_shift(ref, scene)
        except Exception as e:
            return FilterResult(
                passed=True,    # if computation fails, pass through
                reason=None,
                metrics={"shift_magnitude_px": None, "shift_error": str(e)},
            )

        magnitude = float(np.sqrt(shift_row**2 + shift_col**2))
        ground_dist_m = magnitude * self.downsample_scale * 10  # 10m GSD

        passed = magnitude <= self.max_shift_pixels
        reason = None
        if not passed:
            reason = (
                f"Geometric mis-registration detected: shift={magnitude:.2f} px "
                f"(={ground_dist_m:.0f}m at ground) exceeds threshold "
                f"{self.max_shift_pixels} px. "
                f"This is consistent with ESA geometric_quality=FAILED."
            )

        return FilterResult(
            passed=passed,
            reason=reason,
            metrics={
                "shift_row_px":       round(shift_row, 3),
                "shift_col_px":       round(shift_col, 3),
                "shift_magnitude_px": round(magnitude, 3),
                "shift_ground_m":     round(ground_dist_m, 1),
                "corr_error":         round(corr_error, 4),
                "max_shift_px":       self.max_shift_pixels,
                "downsample_scale":   self.downsample_scale,
                "reference":          Path(self._ref_path).name,
            },
        )