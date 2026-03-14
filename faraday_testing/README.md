# Faraday testing (CS-ROMER)

SKA-band Faraday rotation testing: thin/thick/mixed sources, with RFI and depolarization. Produces 2×2 comparison figures (Clean vs RFI, Clean vs Depolarization) using the csromer pipeline.

## Layout

- **config.py** — SKA band definitions, source parameters, noise/RFI/depolarization constants, plotting defaults.
- **simulation.py** — Build thin/thick/mixed datasets (clean, RFI, depolarized) per band.
- **reconstruction.py** — Run reconstruction (FISTA, CG, or CLEAN) on a single dataset.
- **plotting.py** — 2×2 comparison plots; shared panel helpers to avoid duplication.
- **cli.py** — Command-line entry: band selection, output directory, reconstructor choice.
- **products.py** — Zarr product cache: save/load simulation and reconstruction outputs so plots can be regenerated without re-running.

## Usage (from repo root)

```bash
# All bands (default)
python -m faraday_testing

# Single band
python -m faraday_testing --band low
python -m faraday_testing -b mid-b2
python -m faraday_testing -b mid-b5a
python -m faraday_testing -b mid-b5b

# Multiple bands
python -m faraday_testing --band low --band mid-b2

# Output directory and reconstructor (csromer = FISTA+L1, cg, or clean)
python -m faraday_testing -b mid-b5a -o ./figs --reconstructor csromer
python -m faraday_testing -b mid-b5a --reconstructor clean

# Product cache (default: on). Saves products under <outdir>/zarr so re-runs skip simulation/reconstruction.
python -m faraday_testing -b mid-b2 -o ./figs
# Second run: loads from ./figs/zarr and only regenerates plots
python -m faraday_testing -b mid-b2 -o ./figs --no-cache   # always run sim + recon
python -m faraday_testing -o ./figs --cachedir /path/to/cache   # custom cache directory
```

## Bands

| Option   | Band       | Short label |
|----------|------------|-------------|
| `low`    | SKA-LOW    | LOW         |
| `mid-b1` | SKA-MID B1 | B1          |
| `mid-b2` | SKA-MID B2 | B2          |
| `mid-b5a`| SKA-MID B5a| B5a         |
| `mid-b5b`| SKA-MID B5b| B5b         |
| `all`    | (all five) | —           |

SKA-LOW runs only thin sources; other bands run thin, thick, and mixed.
