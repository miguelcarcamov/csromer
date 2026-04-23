"""
Gridded FFT for uniformly spaced lambda². Uses da.fft.fft / da.fft.ifft
when inputs are dask arrays.

Implements the Faraday depth Fourier transform (Burn 1966):
P(lambda²) = ∫ F(phi) * exp(+2j * phi * lambda²) dphi

Discrete setup (after :class:`csromer.transformers.gridding.Gridding`):

- **Phi grid** matches :meth:`csromer.reconstruction.parameter.Parameter.calculate_cellsize`:
  ``phi[j] = cellsize * (j - n//2)`` for ``j = 0 … n-1``, so the most negative φ is at
  ``j=0``, **φ = 0 at ``j = n//2``**, and positive φ at large ``j``. This is *not*
  numpy's default FFT order (DC at index 0).

- **Lambda² grid** is uniform: ``lambda2[k] = l2_0 + k * d_l2``, with only ``l2_0 > 0``
  required (no need to sample λ² < 0). The pipeline chooses
  ``d_l2 = π / (n * cellsize)``, so ``2 * cellsize * d_l2 = 2π/n`` and the cross-term
  ``exp(2j * phi_j * k * d_l2)`` matches ``exp(2π i * (j - n//2) * k / n)`` after half-index
  reordering.

- **ifftshift / fftshift**: ``numpy.ifft`` pairs output bin ``k`` with input order where
  index 0 is the “zero-wavenumber” mode. ``ifftshift`` moves the sample at **φ = 0**
  (grid centre) to index 0 before ``ifft``; ``fft`` + ``fftshift`` inverts that on the
  adjoint. This is the standard centred-grid ↔ FFT indexing map; it does **not** assume
  infinite φ, only a periodic torus on the finite φ grid.

- **Phase ``exp(2j * phi * l2_0)``**: for ``lambda2[k] = l2_0 + k*d_l2``,
  ``exp(2j*phi*lambda2[k]) = exp(2j*phi*l2_0) * exp(2j*phi*k*d_l2)``. The first factor is
  applied to the φ-domain vector before the FFT; the second is what the ``ifft`` implements
  (``norm="forward"`` matches the rest of the stack). If ``l2_0 == 0``, that prefactor is 1.

- **``norm="forward"`` and shifts**: With the same ``norm``, ``numpy.fft.fft`` and
  ``numpy.fft.ifft`` are exact inverses (``fft(ifft(u)) = ifft(fft(u)) = u``). ``ifftshift``
  and ``fftshift`` are inverse permutations. Forward ``ifft(ifftshift(·))`` and adjoint
  ``fftshift(fft(·))`` therefore pair correctly with no extra scaling; unit-modulus diagonal
  phase commutes with the shift.

See ``tests/unit/transformers/test_measurement_operator.py::TestGriddedFFT1D`` (Parseval /
round-trip) and ``tests/integration/test_ft_conventions.py`` (dirty-map peak vs direct).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import numpy as np

from ...utils.array_utils import asnumpy, is_dask_array, math_module
from ...utils.fft_utils import fft1d_forward, fft1d_inverse
from .base import MeasurementOperator

try:
    import dask.array as da
except ImportError:
    da = None


@dataclass(init=True, repr=True)
class GriddedFFT1D(MeasurementOperator):
    """
    FFT-based measurement operator when lambda² is on a regular grid.

    Implements the Faraday depth Fourier transform (Burn 1966):
    P(lambda²) = ∫ F(phi) * exp(+2j * phi * lambda²) dphi

    Expects gridded lambda² = l2_0, l2_0+d_l2, ... (l2_0 = first channel, typically > 0).
    Applies phase exp(2j φ l2_0) so the kernel is correct for physical λ²; when l2_0=0
    (e.g. full resolution with grid at 0) the phase is 1.
    """

    def __post_init__(self):
        """Post-initialization: call configure if dataset and parameter are available."""
        super().__post_init__()
        if self.dataset is not None and self.parameter is not None:
            self.configure()

    def configure(self) -> None:
        """
        Set phase from first gridded λ² (l2_0). Kernel is exp(2j φ λ²_k) with
        λ²_k = l2_0 + k*d_l2, so we phase F by exp(2j φ l2_0) before the FFT.
        """
        if self.dataset is None or self.parameter is None:
            return
        phi = asnumpy(self.parameter.phi)
        l2_grid = asnumpy(self.dataset.lambda2)
        l2_0 = float(l2_grid[0])
        self._l2_ref_phase = np.exp(2j * phi * l2_0).astype(np.complex64)

    def _forward_impl(self, x: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """
        Forward operator implementation: phi -> P(lambda²) via FFT.

        Protected method. ``ifftshift`` then ``ifft(..., norm="forward")``; l2_0 phase from ``configure``.
        """
        if not hasattr(self, '_l2_ref_phase') or self._l2_ref_phase is None:
            self.configure()
        x_phased = x * self._l2_ref_phase
        # Centred phi grid -> DC-at-0 order; paired with inverse fft(..., centered=True).
        return fft1d_forward(x_phased, centered=True, norm="forward")

    def _adjoint_impl(self, b: Union[np.ndarray, Any], **kwargs) -> Union[np.ndarray, Any]:
        """
        Adjoint: P(lambda²) -> phi via ``fft(..., norm="forward")``, ``fftshift``, then conjugate
        l2_0 phase. ``fftshift`` undoes ``ifftshift``; same ``norm`` as forward ``ifft``.
        """
        x_fft = fft1d_inverse(b, centered=True, norm="forward")
        if not hasattr(self, '_l2_ref_phase') or self._l2_ref_phase is None:
            self.configure()
        phase_conj = np.conj(self._l2_ref_phase)
        if da is not None and is_dask_array(x_fft):
            return (x_fft * phase_conj).astype(np.complex64)
        return (x_fft * phase_conj).astype(np.complex64)

    def _dirty_spectrum_impl(self, data: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """
        Dirty spectrum: A^H(weighted data) with scaling so the result matches the
        continuous definition (sum over channels; no 1/N from FFT).
        The FFT adjoint with norm="forward" yields (1/N)*sum; multiply by N here.
        """
        raw = super()._dirty_spectrum_impl(data)
        n_chan = self.dataset.m
        return raw * n_chan

    def RMTF(self, phi_x: float = 0.0) -> Union[np.ndarray, Any]:
        """
        Rotation Measure Transfer Function (RMTF).

        Adjoint of (weights / sum(weights)), then multiplied by n_chan. Uses
        this class's adjoint (IFFT-based) so the gridded transform is used.

        Args:
            phi_x: Faraday depth of point source (rad/m², unused)

        Returns:
            RMTF array (n_phi,), same units as dirty map.
        """
        if self.dataset is None:
            raise RuntimeError("dataset is required for RMTF")
        w = self.dataset.w
        s = getattr(self.dataset, "s", None)
        xp = math_module(w)
        weights = (w / s) if s is not None else w
        sum_w = xp.sum(weights)
        sum_w = sum_w.compute() if hasattr(sum_w, "compute") else sum_w
        sum_w = float(sum_w) if sum_w is not None else 1.0
        if abs(sum_w) < 1e-10:
            sum_w = 1.0
        normalized = weights / sum_w
        # Use this class's adjoint (IFFT path), not the base helper
        raw = self.adjoint(normalized)
        l2_ref = getattr(self.dataset, "l2_ref", None)
        if (
            self.parameter is not None
            and l2_ref is not None
            and abs(float(l2_ref)) >= 1e-10
        ):
            xp = math_module(raw)
            phi = self.parameter.phi
            phi_same = xp.asarray(phi)
            phase_ramp = xp.exp(2.0j * phi_same * float(l2_ref)).astype(np.complex64)
            raw = raw * phase_ramp
        n_chan = self.dataset.m
        return (raw * n_chan).astype(np.complex64)
