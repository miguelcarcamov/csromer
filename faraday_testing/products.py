"""
Zarr-backed product cache: save/load simulation + reconstruction outputs
so plots can be regenerated without re-running simulations or reconstructions.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import zarr
except ImportError:
    zarr = None

from faraday_testing.config import SKA_BANDS


def _cache_dir_band_short(cache_dir: Path, band_name: str, reconstructor: str) -> Path:
    """Directory for this band and reconstructor: cache_dir / SHORT / reconstructor."""
    short = SKA_BANDS[band_name]["short"]
    return cache_dir / short / reconstructor


def _product_path(cache_dir: Path, band_name: str, reconstructor: str, source_key: str) -> Path:
    """Path to the zarr group for one product (e.g. thin_clean.zarr)."""
    base = _cache_dir_band_short(cache_dir, band_name, reconstructor)
    return base / f"{source_key}.zarr"


def save_product(
    cache_dir: Path,
    band_name: str,
    reconstructor: str,
    source_key: str,
    source,
    recon,
) -> bool:
    """
    Save source (Dataset-like) and recon (CSROMERReconstructorWrapper-like) to a zarr group.
    Returns True if saved, False if zarr not available or write failed.
    """
    if zarr is None:
        return False
    path = _product_path(cache_dir, band_name, reconstructor, source_key)
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    try:
        root = zarr.open(str(path), mode="w")
        # Source: lambda2, data (complex)
        l2 = np.asarray(source.lambda2)
        data = np.asarray(source.data)
        root["source/lambda2"] = l2
        root["source/data_real"] = data.real
        root["source/data_imag"] = data.imag
        # Recon: phi, fd_dirty, fd_restored, fd_residual (complex)
        phi = np.asarray(recon.parameter.phi)
        root["recon/phi"] = phi
        for name, arr in (
            ("fd_dirty", recon.fd_dirty),
            ("fd_restored", recon.fd_restored),
            ("fd_residual", recon.fd_residual),
        ):
            a = np.asarray(arr)
            root[f"recon/{name}_real"] = a.real
            root[f"recon/{name}_imag"] = a.imag
        # Scalars in recon group attributes
        root["recon"].attrs["rm_restored"] = float(getattr(recon, "rm_restored", np.nan))
        root["recon"].attrs["rm_restored_error"] = float(
            getattr(recon, "rm_restored_error", np.nan)
        )
        root["recon"].attrs["rm_restored_quadratic_interpolation"] = float(
            getattr(recon, "rm_restored_quadratic_interpolation", np.nan)
        )
        root["recon"].attrs["rm_restored_quadratic_interpolation_error"] = float(
            getattr(recon, "rm_restored_quadratic_interpolation_error", np.nan)
        )
        root.attrs["band_name"] = band_name
        root.attrs["reconstructor"] = reconstructor
        root.attrs["source_key"] = source_key
        return True
    except Exception:
        return False


def load_product(
    cache_dir: Path,
    band_name: str,
    reconstructor: str,
    source_key: str,
) -> tuple[SimpleNamespace, SimpleNamespace] | None:
    """
    Load source-like and recon-like objects from zarr for plotting.
    Returns (source_like, recon_like) or None if not found / zarr not available.
    """
    if zarr is None:
        return None
    path = _product_path(cache_dir, band_name, reconstructor, source_key)
    if not path.exists():
        return None
    try:
        root = zarr.open(str(path), mode="r")
        # Source-like: .lambda2, .data
        l2 = np.asarray(root["source/lambda2"])
        dr = np.asarray(root["source/data_real"])
        di = np.asarray(root["source/data_imag"])
        data = dr + 1j * di
        source_like = SimpleNamespace(lambda2=l2, data=data)
        # Recon-like: .parameter.phi, .fd_dirty, .fd_restored, .fd_residual, .rm_*
        phi = np.asarray(root["recon/phi"])
        recon_phi = SimpleNamespace(phi=phi)
        fd_dirty = np.asarray(root["recon/fd_dirty_real"]) + 1j * np.asarray(
            root["recon/fd_dirty_imag"]
        )
        fd_restored = np.asarray(root["recon/fd_restored_real"]) + 1j * np.asarray(
            root["recon/fd_restored_imag"]
        )
        fd_residual = np.asarray(root["recon/fd_residual_real"]) + 1j * np.asarray(
            root["recon/fd_residual_imag"]
        )
        attrs = root["recon"].attrs
        recon_like = SimpleNamespace(
            parameter=recon_phi,
            fd_dirty=fd_dirty,
            fd_restored=fd_restored,
            fd_residual=fd_residual,
            rm_restored=float(attrs.get("rm_restored", np.nan)),
            rm_restored_error=float(attrs.get("rm_restored_error", np.nan)),
            rm_restored_quadratic_interpolation=float(
                attrs.get("rm_restored_quadratic_interpolation", np.nan)
            ),
            rm_restored_quadratic_interpolation_error=float(
                attrs.get("rm_restored_quadratic_interpolation_error", np.nan)
            ),
        )
        return (source_like, recon_like)
    except Exception:
        return None


def cache_available() -> bool:
    """True if zarr is installed and cache can be used."""
    return zarr is not None
