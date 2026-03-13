# Faraday testing (CS-ROMER)

SKA-band Faraday rotation testing: thin/thick/mixed sources, with RFI and depolarization. Produces 2×2 comparison figures (Clean vs RFI, Clean vs Depolarization) using the csromer pipeline.

## Layout

- **config.py** — SKA band definitions, source parameters, noise/RFI/depolarization constants, plotting defaults.
- **simulation.py** — Build thin/thick/mixed datasets (clean, RFI, depolarized) per band.
- **reconstruction.py** — Run CS-ROMER (FISTA or CG) on a single dataset.
- **plotting.py** — 2×2 comparison plots; shared panel helpers to avoid duplication.
- **cli.py** — Command-line entry: band selection, output directory, reconstructor choice.

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

# Output directory and reconstructor
python -m faraday_testing -b mid-b5a -o ./figs --reconstructor csromer
```

## Bands

| Option   | Band       | Short label |
|----------|------------|-------------|
| `low`    | SKA-LOW    | LOW         |
| `mid-b2` | SKA-MID B2 | B2          |
| `mid-b5a`| SKA-MID B5a| B5a         |
| `mid-b5b`| SKA-MID B5b| B5b         |
| `all`    | (all four) | —           |

SKA-LOW runs only thin sources; other bands run thin, thick, and mixed.
