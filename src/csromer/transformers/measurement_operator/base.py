"""
Base measurement operator: maps between Faraday depth (model) space and
lambda² (data) space. Subclasses implement _forward_impl and _adjoint_impl.

When wavelet_transform is set, forward expects PyWT coefficients: coefficients
→ (pywt reconstruct) → Faraday depth → (measurement) → P(lambda²). Adjoint then
returns coefficients (adjoint of measurement → decompose).
"""
from __future__ import annotations

import copy
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from ...utils.array_utils import math_module

if TYPE_CHECKING:
    from ...base import Dataset
    from ...dictionaries import Wavelet
    from ...reconstruction import Parameter


@dataclass(init=True, repr=True)
class MeasurementOperator(metaclass=ABCMeta):
    """
    Base class for 1D measurement operators (Faraday depth <-> lambda²).

    Implements the forward model: P(lambda²) = A @ phi, where A is the measurement
    operator mapping from Faraday depth space to lambda² space. Supports wavelet
    transforms for sparse representation.

    The Faraday depth Fourier transform convention is:
    P(lambda²) = ∫ F(phi) * exp(+2j * phi * lambda²) dphi

    This uses a "positive" sign convention (exp(+2j*phi*lambda²)) consistent with
    radio astronomy standards. The sign_convention attribute documents this.

    When wavelet_transform is set:
    - forward(x) expects wavelet coefficients, reconstructs to Faraday depth, then applies A
    - adjoint(b) applies A^H, then decomposes to coefficients

    Attributes:
        dataset: Dataset with lambda² coverage and weights
        parameter: Parameter with phi grid configuration
        wavelet_transform: Optional wavelet transform for sparse representation
        sign_convention: FFT sign convention ("positive" for exp(+2j*phi*lambda²), default "positive")
    """
    dataset: "Dataset" = None
    parameter: "Parameter" = None
    wavelet_transform: Optional["Wavelet"] = None
    sign_convention: str = "positive"

    def __post_init__(self):
        """
        Post-initialization: deep copy parameter to avoid shared state.
        """
        if self.parameter is not None:
            self.parameter = copy.deepcopy(self.parameter)

    @abstractmethod
    def _forward_impl(self, x: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """
        Forward operator implementation: phi -> P(lambda²).

        Protected method: subclasses must implement. Maps from complex Faraday depth
        (model) space to data (lambda²) space. Should handle both numpy and dask arrays.

        Args:
            x: Complex Faraday depth spectrum (n_phi,) or wavelet coefficients

        Returns:
            Complex polarization P(lambda²) (n_channels,)
        """
        pass

    @abstractmethod
    def _adjoint_impl(self, b: Union[np.ndarray, Any], **kwargs) -> Union[np.ndarray, Any]:
        """
        Adjoint operator implementation: P(lambda²) -> phi.

        Protected method: subclasses must implement. Maps from data (lambda²) space
        to complex Faraday depth space. Should handle both numpy and dask arrays.

        Args:
            b: Complex polarization P(lambda²) (n_channels,)
            **kwargs: Additional arguments (e.g. for iterative solvers)

        Returns:
            Complex Faraday depth spectrum (n_phi,) or wavelet coefficients
        """
        pass

    def _l2_ref_ramp(self, sign: float):
        """
        exp(sign * 2j * phi * l2_ref), or None if l2_ref is unset/zero.

        Private helper. Applied in forward()/adjoint() so that the Faraday-depth
        model CG/FISTA fit and dirty_spectrum() share one phase convention: with
        l2_ref != 0, dirty_spectrum estimates F(phi) rotated by exp(+2j*phi*l2_ref)
        (Brentjens & de Bruyn lambda0² referencing). Multiplying x by exp(-2j*phi*l2_ref)
        before the underlying (l2_ref-agnostic) forward, and the underlying adjoint's
        output by exp(+2j*phi*l2_ref), makes forward/adjoint estimate that same rotated
        F(phi)—so conv(model) and dirty/residual no longer mix conventions (which
        otherwise causes phase-cancellation when convolving with the restoring beam).

        Args:
            sign: +1.0 for adjoint (post-multiply), -1.0 for forward (pre-multiply)

        Returns:
            Ramp array (n_phi,) or None
        """
        if self.dataset is None or self.parameter is None:
            return None
        l2_ref = getattr(self.dataset, "l2_ref", None)
        if l2_ref is None or abs(float(l2_ref)) < 1e-10:
            return None
        xp = math_module(self.parameter.phi)
        phi = xp.asarray(self.parameter.phi)
        return xp.exp(sign * 2.0j * phi * float(l2_ref)).astype(np.complex64)

    def forward(self, x: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """
        Forward operator: phi -> P(lambda²).

        Public method. If wavelet_transform is set, reconstructs coefficients to
        Faraday depth first, then applies measurement operator. When dataset.l2_ref
        is set, rotates by exp(-2j*phi*l2_ref) first so the model this fits matches
        dirty_spectrum's rotated convention (see _l2_ref_ramp).

        Args:
            x: Complex Faraday depth spectrum (n_phi,) or wavelet coefficients

        Returns:
            Complex polarization P(lambda²) (n_channels,)
        """
        if self.wavelet_transform is not None:
            x = self.wavelet_transform.reconstruct_complex(np.array(x, copy=False))
        ramp = self._l2_ref_ramp(-1.0)
        if ramp is not None:
            x = x * ramp
        return self._forward_impl(x)

    def adjoint(self, b: Union[np.ndarray, Any], **kwargs) -> Union[np.ndarray, Any]:
        """
        Adjoint operator: P(lambda²) -> phi.

        Public method. Applies measurement operator adjoint. When dataset.l2_ref is
        set, rotates the result by exp(+2j*phi*l2_ref) (see _l2_ref_ramp) before any
        wavelet decomposition, so adjoint/dirty_spectrum/RMTF and forward all share
        one phase convention.

        Args:
            b: Complex polarization P(lambda²) (n_channels,)
            **kwargs: Additional arguments (passed to _adjoint_impl)

        Returns:
            Complex Faraday depth spectrum (n_phi,) or wavelet coefficients
        """
        res = self._adjoint_impl(b, **kwargs)
        ramp = self._l2_ref_ramp(1.0)
        if ramp is not None:
            res = res * ramp
        if self.wavelet_transform is not None:
            # Complex Faraday depth -> complex coefficients (pywt)
            res = self.wavelet_transform.decompose_complex(res)
        return res

    def backward(self, b: Union[np.ndarray, Any], **kwargs) -> Union[np.ndarray, Any]:
        """
        Backward operator: alias for adjoint (raw linear adjoint, no weighting).

        Public method. Used in chi-squared gradient: caller passes weighted
        residual (w * (data - model_data)); backward returns A^H(b). No
        weights or normalization are applied here—dirty_spectrum does that
        separately for the dirty map.

        Args:
            b: Complex polarization P(lambda²) (n_channels,)
            **kwargs: Additional arguments (passed to adjoint)

        Returns:
            Complex Faraday depth spectrum (n_phi,) or wavelet coefficients
        """
        return self.adjoint(b, **kwargs)

    def dirty_spectrum(self, data: Union[np.ndarray, Any] = None) -> Union[np.ndarray, Any]:
        """
        Compute dirty Faraday depth spectrum: A^H(weighted data) / K.

        Public method. Used for initial dirty map. Applies weights, spectral index
        correction, and normalization.

        Args:
            data: Input data (default: dataset.data)

        Returns:
            Dirty Faraday depth spectrum (n_phi,)
        """
        if data is None and self.dataset is not None:
            data = self.dataset.data
        return self._dirty_spectrum_impl(data)

    def _dirty_spectrum_impl(self, data: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """
        Dirty spectrum implementation: A^H(weighted data).

        Protected method. The adjoint operator expects weighted data: w*p/sum(w)
        (or (w/s)*p/sum(w/s) if spectral index correction is applied).
        The measurement operator adjoint only works as forward/adjoint for an x input,
        so we must apply weights and normalization before passing to adjoint.

        Since dividing by sum(w) or sum(w/s) is equivalent to dividing by k (where
        k = sum(w) or k = sum(w/s)), we normalize before the adjoint and do not
        need to divide by k after the transform.

        Args:
            data: Input data array (polarization p)

        Returns:
            Dirty Faraday depth spectrum (n_phi,)
        """
        if self.dataset is None:
            return self.adjoint(data)
        w = self.dataset.w
        s = getattr(self.dataset, "s", None)
        xp = math_module(w)

        # Compute weighted data: (w/s) * p (or w*p if no spectral index)
        # Then normalize by sum(w/s) or sum(w) to get w*p/sum(w) or (w/s)*p/sum(w/s)
        # This is what the adjoint operator expects
        if s is not None:
            weighted = (w / s) * data
            sum_w_over_s = xp.sum(w / s)
            sum_w_over_s = sum_w_over_s.compute(
            ) if hasattr(sum_w_over_s, "compute") else sum_w_over_s
            if sum_w_over_s is not None:
                sum_w_over_s = float(sum_w_over_s)
                if abs(sum_w_over_s) > 1e-10:  # Avoid division by very small numbers
                    weighted = weighted / sum_w_over_s
        else:
            weighted = w * data
            sum_w = xp.sum(w)
            sum_w = sum_w.compute() if hasattr(sum_w, "compute") else sum_w
            if sum_w is not None:
                sum_w = float(sum_w)
                if abs(sum_w) > 1e-10:  # Avoid division by very small numbers
                    weighted = weighted / sum_w

        # Adjoint operator receives properly weighted and normalized data
        # No need to divide by k after, since normalization by sum(w) or sum(w/s) is equivalent
        # l2_ref phase (if set) is applied inside adjoint() itself, so forward/adjoint/
        # dirty_spectrum/RMTF all share one convention (see _l2_ref_ramp).
        return self.adjoint(weighted)

    def _adjoint_normalized_weights(self) -> Union[np.ndarray, Any]:
        """
        Adjoint of (weights / sum(weights)).
        Used by subclasses to implement RMTF without going through forward.
        l2_ref phase (if set) is applied inside adjoint() itself (see _l2_ref_ramp).
        """
        if self.dataset is None:
            raise RuntimeError("dataset is required for RMTF")
        w = self.dataset.w
        s = getattr(self.dataset, "s", None)
        xp = math_module(w)
        if s is not None:
            weights = w / s
        else:
            weights = w
        sum_w = xp.sum(weights)
        sum_w = sum_w.compute() if hasattr(sum_w, "compute") else sum_w
        sum_w = float(sum_w) if sum_w is not None else 1.0
        if abs(sum_w) < 1e-10:
            sum_w = 1.0
        normalized = weights / sum_w
        return self.adjoint(normalized)

    def configure(self) -> None:
        """
        Configure operator (e.g. NUFFT plan, precompute matrices).

        Public method. Override in subclasses to perform one-time setup.
        Called automatically after initialization if parameter and dataset are set.
        """
        pass

    def RMTF(self, phi_x: float = 0.0) -> Union[np.ndarray, Any]:
        """
        Rotation Measure Transfer Function (RMTF).

        Public method. Override in subclasses if RMTF computation is supported.
        The RMTF describes the response to a point source at phi_x.

        Args:
            phi_x: Faraday depth of point source (rad/m², default: 0.0)

        Returns:
            RMTF array (n_phi,)

        Raises:
            NotImplementedError: If RMTF is not implemented for this operator
        """
        raise NotImplementedError("RMTF not implemented for this operator")
