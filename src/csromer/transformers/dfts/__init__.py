# Backward compatibility: re-export from measurement_operator
from ..measurement_operator import NUFFT1D, DirectFourier1D, GriddedFFT1D, MeasurementOperator

FT = MeasurementOperator
NDFT1D = DirectFourier1D

__all__ = ["FT", "NDFT1D", "NUFFT1D", "MeasurementOperator", "DirectFourier1D", "GriddedFFT1D"]
