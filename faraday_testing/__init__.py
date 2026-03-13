"""
CS-ROMER Faraday testing: SKA bands, thin/thick/mixed sources,
RFI and depolarization, with 2×2 comparison figures.

Run from repo root:
  python -m faraday_testing --band all
  python -m faraday_testing --band low
  python -m faraday_testing --band mid-b2 --band mid-b5a
"""

from faraday_testing.cli import main

__all__ = ["main"]
