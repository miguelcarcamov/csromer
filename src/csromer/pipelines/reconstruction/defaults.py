"""
Default builders for parameter, measurement operator, and objective.

Used when the user does not pass pre-built instances. Enables a simple
API (e.g. dataset + oversampling + kind) while supporting full injection.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from csromer.objectivefunction import L1, ChiSquared, OFunction
from csromer.reconstruction import Parameter
from csromer.transformers.dfts import GriddedFFT1D, NDFT1D, NUFFT1D
from csromer.transformers.gridding import Gridding


def build_parameter(dataset, oversampling: float = 7.0, cellsize: float = None) -> Parameter:
    """
    Build Faraday depth parameter from dataset.

    Args:
        dataset: Dataset with polarization data.
        oversampling: Oversampling factor (used if cellsize is None).
        cellsize: Optional grid spacing (rad/m²). If set, oversampling ignored.

    Returns:
        Parameter instance with phi grid and cellsize set.
    """
    param = Parameter()
    if cellsize is not None:
        param.calculate_cellsize(dataset=dataset, cellsize=cellsize)
    else:
        param.calculate_cellsize(dataset=dataset, oversampling=oversampling)
    return param


def build_measurement_operator(
    dataset,
    parameter: Parameter,
    kind: str = "direct",
    gridding_kernel: str = "kaiser",
    gridding_kernel_half_width: float = 4.0,
    gridding_kernel_beta: float = 2.5,
):
    """
    Build measurement operator (and optionally update dataset for gridding).

    Args:
        dataset: Dataset (may be replaced for "gridded").
        parameter: Faraday depth parameter.
        kind: "direct" (NDFT), "nufft", or "gridded" (grid then FFT).
        gridding_kernel: For kind "gridded", "kaiser" or "box".
        gridding_kernel_half_width: For Kaiser, half-width in grid units.
        gridding_kernel_beta: For Kaiser, shape parameter.

    Returns:
        (measurement_operator, dataset). For "gridded", dataset is the gridded one.
    """
    kind = (kind or "direct").strip().lower()
    if kind == "direct":
        op = NDFT1D(dataset=dataset, parameter=parameter)
        return op, dataset
    if kind == "nufft":
        op = NUFFT1D(dataset=dataset, parameter=parameter, solve=True)
        return op, dataset
    if kind == "gridded":
        d_lambda2 = np.pi / (parameter.n * parameter.cellsize)
        gridding = Gridding(
            dataset=dataset,
            d_lambda2=d_lambda2,
            n=parameter.n,
            kernel=gridding_kernel,
            gridding_kernel_half_width=gridding_kernel_half_width,
            gridding_kernel_beta=gridding_kernel_beta,
        )
        dataset = gridding.run()
        # After gridding, update RMTF FWHM from the gridded dataset so that
        # downstream steps (e.g. restoration kernel, RM error estimates) use
        # the RMTF associated with the actual operator in use.
        if parameter is not None and getattr(dataset, "delta_phi", None) is not None:
            delta_phi_fwhm = dataset.delta_phi
            if delta_phi_fwhm is not None:
                parameter.rmtf_fwhm = float(delta_phi_fwhm)
        op = GriddedFFT1D(dataset=dataset, parameter=parameter)
        return op, dataset
    raise ValueError(
        "measurement_operator kind must be 'direct', 'nufft', or 'gridded'; got %r" % kind
    )


def default_objective_factory(
    lambda_l_norm: float = 0.0, wavelet=None
) -> Callable:
    """
    Return an objective factory: (measurement_operator, parameter) -> OFunction.

    Default objective is ChiSquared + optional L1. Use when you do not pass
    your own objective_factory to the reconstructor.
    """

    def factory(measurement_operator, parameter):
        chi_squared = ChiSquared(
            measurement_operator=measurement_operator, wavelet=wavelet
        )
        terms = [chi_squared]
        if lambda_l_norm != 0:
            terms.append(L1(reg=lambda_l_norm))
        return OFunction(terms, persist_gradient=True)

    return factory
