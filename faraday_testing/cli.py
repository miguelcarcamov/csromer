"""
Command-line entry point: run Faraday testing for selected SKA bands.

Usage (from repo root):
  python -m faraday_testing --band all
  python -m faraday_testing --band low
  python -m faraday_testing --band low --band mid-b2
  python -m faraday_testing -b mid-b5a -b mid-b5b --reconstructor csromer --outdir ./figs
  python -m faraday_testing -b mid-b5a --reconstructor clean
  python -m faraday_testing -b mid-b1 -e thick_clean -e thick_rfi   # only thick clean vs RFI for B1
  python -m faraday_testing -b mid-b2 -e thick_clean -e thick_rfi -e thick_depol  # thick experiments only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src (csromer) is on path when run as __main__
_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from faraday_testing import config
from faraday_testing.plotting import plot_2x2_clean_vs_depol, plot_2x2_clean_vs_rfi
from faraday_testing.products import cache_available, load_product, save_product
from faraday_testing.reconstruction import run_csromer_reconstruction
from faraday_testing.simulation import intrinsic_components_for_key, simulate_one_source


def _parse_args():
    p = argparse.ArgumentParser(
        description=
        "CS-ROMER Faraday testing: thin/thick/mixed sources, RFI and depolarization, SKA bands.",
    )
    p.add_argument(
        "--band",
        "-b",
        action="append",
        dest="bands",
        choices=config.BAND_CHOICES + ["all"],
        help="Band(s) to run: low, mid-b1, mid-b2, mid-b5a, mid-b5b, or all (default).",
    )
    p.add_argument(
        "--reconstructor",
        choices=config.RECONSTRUCTOR_CHOICES,
        default=config.RECONSTRUCTOR_DEFAULT,
        help="Reconstructor: csromer (FISTA+L1), cg, or clean.",
    )
    p.add_argument(
        "--outdir",
        "-o",
        type=Path,
        default=Path("."),
        help="Output directory for PNG figures (default: current directory).",
    )
    p.add_argument(
        "--cache",
        action="store_true",
        default=True,
        help=
        "Use zarr cache to skip re-running simulations/reconstructions when products exist (default).",
    )
    p.add_argument(
        "--no-cache",
        action="store_false",
        dest="cache",
        help="Disable cache: always run simulation and reconstruction.",
    )
    p.add_argument(
        "--cachedir",
        type=Path,
        default=None,
        help="Directory for zarr product cache (default: <outdir>/zarr).",
    )
    p.add_argument(
        "--experiment",
        "-e",
        action="append",
        dest="experiments",
        choices=[
            "thin_clean",
            "thin_rfi",
            "thin_depol",
            "thick_clean",
            "thick_rfi",
            "thick_depol",
            "mixed_clean",
            "mixed_rfi",
        ],
        metavar="KEY",
        help=
        "Run only these experiments (can repeat). If not set, run all for each band. For LOW only thin_* apply.",
    )
    p.add_argument(
        "--target-residual-sigma",
        type=float,
        default=None,
        metavar="S",
        help=
        "FISTA: target data residual level in sigma (1=noise, 5=~5σ). Default: 1.0; if only --fd-accept-n-sigma is set, that value is used here too.",
    )
    p.add_argument(
        "--no-compute-sigma-fd",
        action="store_true",
        dest="no_compute_sigma_fd",
        help=
        "Disable FD-space σ_fd computation (Hutchinson A^H Σ_d A). By default σ_fd is computed for CLEAN threshold and FD-panel lines.",
    )
    p.add_argument(
        "--fd-accept-n-sigma",
        type=float,
        default=None,
        metavar="N",
        help=
        "FISTA: also require max|fd_residual| <= N*σ_fd to accept λ. Default: not set (χ²-only acceptance).",
    )
    p.add_argument(
        "--show-intrinsic-model",
        action="store_true",
        help="Overlay intrinsic (ground-truth) |F(phi)| model on the upper-right FD panel.",
    )
    return p.parse_args()


def _keys_for_band(band_name: str, experiments: list[str] | None = None) -> list[str]:
    """Source keys needed for this band (thin always; thick/mixed only for non-LOW)."""
    keys = ["thin_clean", "thin_rfi", "thin_depol"]
    if band_name != "SKA-LOW":
        keys += ["thick_clean", "thick_rfi", "thick_depol", "mixed_clean", "mixed_rfi"]
    if experiments is not None:
        keys = [k for k in keys if k in experiments]
    return keys


def run_band(
    band_name: str,
    outdir: Path,
    reconstructor: str,
    use_cache: bool,
    cache_dir: Path,
    experiments: list[str] | None = None,
    target_residual_sigma: float | None = None,
    compute_sigma_fd: bool = True,
    fd_accept_n_sigma: float | None = None,
    show_intrinsic_model: bool = False,
) -> None:
    """Run simulation, reconstruction (or load from zarr cache), and plotting for one band."""
    cfg = config.SKA_BANDS[band_name]
    short = cfg["short"]
    phi_xlim = config.PHI_XLIM_BY_BAND.get(band_name, config.PHI_MAX)
    keys = _keys_for_band(band_name, experiments)
    if not keys:
        print(
            f"  No experiments to run for {band_name} (check --experiment choices for this band)."
        )
        return

    sims = {}
    recons = {}
    if use_cache and cache_available():
        for key in keys:
            loaded = load_product(cache_dir, band_name, reconstructor, key)
            if loaded is not None:
                sims[key], recons[key] = loaded
                if not hasattr(sims[key], "intrinsic_components"):
                    fallback_components = intrinsic_components_for_key(key, band_name)
                    if fallback_components is not None:
                        sims[key].intrinsic_components = fallback_components
    # Only run simulation + reconstruction for experiments missing from cache
    missing = [k for k in keys if k not in sims]
    if missing:
        nu = cfg["freq"]()
        for key in missing:
            print(f"  Simulate + reconstruct {key}...")
            source = simulate_one_source(key, nu, band_name)
            sims[key] = source
            if source is not None:
                recon = run_csromer_reconstruction(
                    source,
                    reconstructor=reconstructor,
                    target_residual_sigma=target_residual_sigma,
                    compute_sigma_fd=compute_sigma_fd,
                    fd_accept_n_sigma=fd_accept_n_sigma,
                )
                recons[key] = recon
                if use_cache and cache_available():
                    save_product(
                        cache_dir,
                        band_name,
                        reconstructor,
                        key,
                        source,
                        recon,
                    )
            else:
                recons[key] = None

    # Plot only when both sources for a comparison exist
    rec = reconstructor
    if sims.get("thin_clean") is not None and sims.get("thin_rfi") is not None:
        print("    Plot: Thin clean vs RFI...")
        plot_2x2_clean_vs_rfi(
            sims["thin_clean"],
            sims["thin_rfi"],
            recons["thin_clean"],
            recons["thin_rfi"],
            band_label=band_name,
            source_type="Thin",
            filename=str(outdir / f"thin_clean_vs_rfi_{short}_{rec}.png"),
            phi_xlim=phi_xlim,
            show_intrinsic_model=show_intrinsic_model,
        )
    if sims.get("thin_clean") is not None and sims.get("thin_depol") is not None:
        print("    Plot: Thin clean vs depolarized...")
        plot_2x2_clean_vs_depol(
            sims["thin_clean"],
            sims["thin_depol"],
            recons["thin_clean"],
            recons["thin_depol"],
            band_label=band_name,
            source_type="Thin",
            filename=str(outdir / f"thin_depolarization_{short}_{rec}.png"),
            phi_xlim=phi_xlim,
            show_intrinsic_model=show_intrinsic_model,
        )

    if band_name != "SKA-LOW":
        if sims.get("thick_clean") is not None and sims.get("thick_rfi") is not None:
            print("    Plot: Thick clean vs RFI...")
            plot_2x2_clean_vs_rfi(
                sims["thick_clean"],
                sims["thick_rfi"],
                recons["thick_clean"],
                recons["thick_rfi"],
                band_label=band_name,
                source_type="Thick",
                filename=str(outdir / f"thick_clean_vs_rfi_{short}_{rec}.png"),
                phi_xlim=phi_xlim,
                show_intrinsic_model=show_intrinsic_model,
            )
        if sims.get("thick_clean") is not None and sims.get("thick_depol") is not None:
            print("    Plot: Thick clean vs depolarized...")
            plot_2x2_clean_vs_depol(
                sims["thick_clean"],
                sims["thick_depol"],
                recons["thick_clean"],
                recons["thick_depol"],
                band_label=band_name,
                source_type="Thick",
                filename=str(outdir / f"thick_depolarization_{short}_{rec}.png"),
                phi_xlim=phi_xlim,
                show_intrinsic_model=show_intrinsic_model,
            )
        if sims.get("mixed_clean") is not None and sims.get("mixed_rfi") is not None:
            print("    Plot: Mixed clean vs RFI...")
            plot_2x2_clean_vs_rfi(
                sims["mixed_clean"],
                sims["mixed_rfi"],
                recons["mixed_clean"],
                recons["mixed_rfi"],
                band_label=band_name,
                source_type="Mixed",
                filename=str(outdir / f"mixed_clean_vs_rfi_{short}_{rec}.png"),
                phi_xlim=phi_xlim,
                show_intrinsic_model=show_intrinsic_model,
            )


def main() -> None:
    args = _parse_args()
    bands = config.get_bands_to_run(args.bands or ["all"])
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cachedir if args.cachedir is not None else outdir / "zarr"
    use_cache = args.cache and cache_available()
    if args.cache and not cache_available():
        print("Warning: zarr not installed; cache disabled. Install zarr for product caching.")

    print("=" * 80)
    print("CS-ROMER Faraday testing (thin/thick/mixed, SKA bands)")
    print("=" * 80)
    print(f"Reconstructor: {args.reconstructor}")
    print(f"Bands: {bands}")
    print(f"Output: {outdir.resolve()}")
    print(f"Cache: {'on' if use_cache else 'off'} ({cache_dir.resolve()})")
    print()

    # If only --fd-accept-n-sigma is set, use it for data-space target too (one number for both).
    target_residual_sigma = args.target_residual_sigma
    if target_residual_sigma is None and args.fd_accept_n_sigma is not None:
        target_residual_sigma = args.fd_accept_n_sigma

    experiments = args.experiments  # None = all; else list of keys
    for band_name in bands:
        print("=" * 80)
        print(f"Processing band: {band_name}")
        print("=" * 80)
        run_band(
            band_name,
            outdir,
            args.reconstructor,
            use_cache,
            cache_dir,
            experiments=experiments,
            target_residual_sigma=target_residual_sigma,
            compute_sigma_fd=not getattr(args, "no_compute_sigma_fd", False),
            fd_accept_n_sigma=args.fd_accept_n_sigma,
            show_intrinsic_model=args.show_intrinsic_model,
        )
        print()

    print("All 2×2 comparison figures generated (Clean vs RFI, Clean vs Depolarized).")


if __name__ == "__main__":
    main()
