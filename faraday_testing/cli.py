"""
Command-line entry point: run Faraday testing for selected SKA bands.

Usage (from repo root):
  python -m faraday_testing --band all
  python -m faraday_testing --band low
  python -m faraday_testing --band low --band mid-b2
  python -m faraday_testing -b mid-b5a -b mid-b5b --reconstructor csromer --outdir ./figs
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
from faraday_testing.reconstruction import run_csromer_reconstruction
from faraday_testing.simulation import simulate_sources_for_band
from faraday_testing.plotting import plot_2x2_clean_vs_rfi, plot_2x2_clean_vs_depol


def _parse_args():
    p = argparse.ArgumentParser(
        description="CS-ROMER Faraday testing: thin/thick/mixed sources, RFI and depolarization, SKA bands.",
    )
    p.add_argument(
        "--band", "-b",
        action="append",
        dest="bands",
        choices=config.BAND_CHOICES + ["all"],
        help="Band(s) to run: low, mid-b2, mid-b5a, mid-b5b, or all (default).",
    )
    p.add_argument(
        "--reconstructor",
        choices=("csromer", "cg"),
        default=config.RECONSTRUCTOR_DEFAULT,
        help="Reconstructor: csromer (FISTA+L1) or cg.",
    )
    p.add_argument(
        "--outdir", "-o",
        type=Path,
        default=Path("."),
        help="Output directory for PNG figures (default: current directory).",
    )
    return p.parse_args()


def run_band(band_name: str, outdir: Path, reconstructor: str) -> None:
    """Run simulation, reconstruction, and plotting for one band."""
    cfg = config.SKA_BANDS[band_name]
    nu = cfg["freq"]()
    short = cfg["short"]
    phi_xlim = 100.0 if band_name == "SKA-LOW" else None

    sims = simulate_sources_for_band(nu, band_name)

    print("  Reconstructing thin sources...")
    recon_thin_clean = run_csromer_reconstruction(sims["thin_clean"], reconstructor=reconstructor)
    recon_thin_rfi = run_csromer_reconstruction(sims["thin_rfi"], reconstructor=reconstructor)
    recon_thin_depol = run_csromer_reconstruction(sims["thin_depol"], reconstructor=reconstructor)

    print("    Plot: Thin clean vs RFI...")
    plot_2x2_clean_vs_rfi(
        sims["thin_clean"], sims["thin_rfi"],
        recon_thin_clean, recon_thin_rfi,
        band_label=band_name, source_type="Thin",
        filename=str(outdir / f"thin_clean_vs_rfi_{short}.png"),
        phi_xlim=phi_xlim,
    )
    print("    Plot: Thin clean vs depolarized...")
    plot_2x2_clean_vs_depol(
        sims["thin_clean"], sims["thin_depol"],
        recon_thin_clean, recon_thin_depol,
        band_label=band_name, source_type="Thin",
        filename=str(outdir / f"thin_depolarization_{short}.png"),
        phi_xlim=phi_xlim,
    )

    if band_name != "SKA-LOW":
        print("  Reconstructing thick sources...")
        recon_thick_clean = run_csromer_reconstruction(sims["thick_clean"], reconstructor=reconstructor)
        recon_thick_rfi = run_csromer_reconstruction(sims["thick_rfi"], reconstructor=reconstructor)
        recon_thick_depol = run_csromer_reconstruction(sims["thick_depol"], reconstructor=reconstructor)

        print("    Plot: Thick clean vs RFI...")
        plot_2x2_clean_vs_rfi(
            sims["thick_clean"], sims["thick_rfi"],
            recon_thick_clean, recon_thick_rfi,
            band_label=band_name, source_type="Thick",
            filename=str(outdir / f"thick_clean_vs_rfi_{short}.png"),
            phi_xlim=phi_xlim,
        )
        print("    Plot: Thick clean vs depolarized...")
        plot_2x2_clean_vs_depol(
            sims["thick_clean"], sims["thick_depol"],
            recon_thick_clean, recon_thick_depol,
            band_label=band_name, source_type="Thick",
            filename=str(outdir / f"thick_depolarization_{short}.png"),
            phi_xlim=phi_xlim,
        )

        print("  Reconstructing mixed sources...")
        recon_mixed_clean = run_csromer_reconstruction(sims["mixed_clean"], reconstructor=reconstructor)
        recon_mixed_rfi = run_csromer_reconstruction(sims["mixed_rfi"], reconstructor=reconstructor)
        print("    Plot: Mixed clean vs RFI...")
        plot_2x2_clean_vs_rfi(
            sims["mixed_clean"], sims["mixed_rfi"],
            recon_mixed_clean, recon_mixed_rfi,
            band_label=band_name, source_type="Mixed",
            filename=str(outdir / f"mixed_clean_vs_rfi_{short}.png"),
            phi_xlim=phi_xlim,
        )


def main() -> None:
    args = _parse_args()
    bands = config.get_bands_to_run(args.bands or ["all"])
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CS-ROMER Faraday testing (thin/thick/mixed, SKA bands)")
    print("=" * 80)
    print(f"Reconstructor: {args.reconstructor}")
    print(f"Bands: {bands}")
    print(f"Output: {outdir.resolve()}")
    print()

    for band_name in bands:
        print("=" * 80)
        print(f"Processing band: {band_name}")
        print("=" * 80)
        run_band(band_name, outdir, args.reconstructor)
        print()

    print("All 2×2 comparison figures generated (Clean vs RFI, Clean vs Depolarized).")


if __name__ == "__main__":
    main()
