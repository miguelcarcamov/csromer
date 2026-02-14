"""
Dataset: Base class for observational data (frequency/lambda², polarization, weights).

Supports both numpy and dask arrays. Handles Stokes Q/U weights and computes polarization
weight W_P as harmonic mean. Manages spectral index correction and residual analysis.
"""
from __future__ import annotations

import sys
from abc import ABCMeta
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Union

import astropy.units as u
import numpy as np
import scipy.signal as sci_signal
import scipy.stats
from scipy import special
from scipy.constants import speed_of_light as c

from ..utils.array_utils import asnumpy, is_dask_array, length_of, maybe_compute

if TYPE_CHECKING:
    from ..transformers.gridding import Gridding

try:
    import dask.array as da
except ImportError:
    da = None


def _calculate_sigma(
    image=None,
    x0=0,
    xn=0,
    y0=0,
    yn=0,
    sigma_error=None,
    residual_cal_error=None,
    nbeam=None,
):
    """
    Calculate noise sigma from image region or error parameters.
    
    Private helper function for noise estimation.
    
    Args:
        image: 2D image array
        x0, xn, y0, yn: Region bounds for variance calculation
        sigma_error: Per-pixel error
        residual_cal_error: Calibration error fraction
        nbeam: Number of beams
        
    Returns:
        Estimated sigma value
    """
    if sigma_error is None and residual_cal_error is None and nbeam is None:
        sigma = np.sqrt(np.mean(image[y0:yn, x0:xn]**2))
    else:
        flux = np.sum(image)
        sigma = np.sqrt((residual_cal_error * flux)**2 + (sigma_error * np.sqrt(nbeam))**2)

    return sigma


def _autocorr_gridded(x: np.ndarray) -> np.ndarray:
    """
    Compute autocorrelation of gridded 1D array.
    
    Private helper for residual analysis.
    
    Args:
        x: 1D numpy array
        
    Returns:
        Autocorrelation function (normalized, non-negative lags only)
    """
    variance = x.var()
    x_input = x - x.mean()
    result = sci_signal.correlate(x_input, x_input, mode="full", method="auto")
    result /= variance * len(x)
    return result[result.size // 2:]


def boxpierce(x: np.ndarray = None, k: Union[List, int] = None, conf_level: float = 0.95) -> Tuple[np.ndarray, float]:
    """
    Box-Pierce test statistic for residual autocorrelation.
    
    Public function for statistical analysis of residuals.
    
    Args:
        x: Autocorrelation function (1D array)
        k: Lag(s) to test (int or list of ints)
        conf_level: Confidence level (default 0.95)
        
    Returns:
        Tuple of (test statistic(s), chi2 critical value)
        
    Raises:
        ValueError: If k < 1
    """
    n = len(x)
    if type(k) == list:
        res = []
        for i in k:
            if i < 1:
                raise ValueError("Cannot calculate for lags lower than 1")
            idx = np.arange(1, i + 1, 1)
            x_sum = n * np.sum(x[idx]**2)
            res.append(x_sum)
        chi2 = scipy.stats.chi2.ppf(conf_level, df=k)
        return np.array(res), chi2
    else:
        if k < 1:
            raise ValueError("The lag cannot be less than 1")
        idx = np.arange(1, k + 1, 1)
        x_sum = n * np.sum(x[idx]**2)
        return np.array(x_sum), scipy.stats.chi2.ppf(conf_level, df=k)


def ljungbox(x: np.ndarray = None, k: Union[List, int] = None, conf_level: float = 0.95) -> Tuple[np.ndarray, float]:
    """
    Ljung-Box test statistic for residual autocorrelation.
    
    Public function for statistical analysis of residuals (modified Box-Pierce).
    
    Args:
        x: Autocorrelation function (1D array)
        k: Lag(s) to test (int or list of ints)
        conf_level: Confidence level (default 0.95)
        
    Returns:
        Tuple of (test statistic(s), chi2 critical value)
        
    Raises:
        ValueError: If k < 1
    """
    n = len(x)
    if isinstance(k, list):
        res = []
        for i in k:
            if i < 1:
                raise ValueError("Cannot calculate for lags lower than 1")
            idx = np.arange(1, i + 1, 1)
            x_sum = n * (n + 2) * np.sum(x[idx]**2 / (n - idx))
            res.append(x_sum)
        chi2 = scipy.stats.chi2.ppf(conf_level, df=k)
        return np.array(res), chi2
    else:
        if k < 1:
            raise ValueError("The lag cannot be less than 1")
        idx = np.arange(1, k + 1, 1)
        x_sum = n * (n + 2) * np.sum(x[idx]**2 / (n - idx))
        return np.array(x_sum), scipy.stats.chi2.ppf(conf_level, df=k)


def _harmonic_mean_w_p(w_q, w_u):
    """
    Compute polarization weight W_P = 2 / (1/W_Q + 1/W_U) element-wise.
    
    Private helper function. Handles dask and numpy arrays.
    
    Args:
        w_q: Stokes Q weights
        w_u: Stokes U weights
        
    Returns:
        Harmonic mean weights (same type as inputs)
    """
    if da is not None and (is_dask_array(w_q) or is_dask_array(w_u)):
        inv_sum = 1.0 / w_q + 1.0 / w_u
        return da.where(da.greater(inv_sum, 0), 2.0 / inv_sum, 0.0)
    wq = np.asarray(w_q)
    wu = np.asarray(w_u)
    inv_sum = 1.0 / np.where(wq > 0, wq, np.nan) + 1.0 / np.where(wu > 0, wu, np.nan)
    out = np.where(np.isfinite(inv_sum) & (inv_sum > 0), 2.0 / inv_sum, 0.0)
    return out.astype(np.float64)


@dataclass(init=False, repr=True)
class Dataset(metaclass=ABCMeta):
    """
    Base dataset class for polarization data in lambda² space.
    
    Manages frequency/lambda² coordinates, complex polarization data (P = Q + iU),
    weights (w, w_q, w_u, w_p), spectral index correction, and residuals.
    Supports both numpy and dask arrays for lazy computation.
    
    Attributes:
        nu: Frequency array (Hz)
        lambda2: Wavelength squared array (m²)
        data: Complex polarization P = Q + iU
        l2_ref: Reference lambda² for phase (default: weighted mean)
        w: Main weight (or w_p when w_q, w_u are set)
        w_q: Stokes Q weights (optional)
        w_u: Stokes U weights (optional)
        w_p: Polarization weight (harmonic mean of w_q, w_u, read-only)
        sigma: Noise per channel (derived from w)
        spectral_idx: Spectral index for correction
        gridded: Whether data is on regular grid
        s: Spectral index correction factor
        model_data: Model prediction
        m: Number of channels
        theo_noise: Theoretical noise level
        nu_0: Reference frequency
        k: Normalization factor (sum of weights)
        delta_phi_full: Full resolution (rad/m²): 2 / (lambda²_max + lambda²_min) (Rudnick & Cotton 2023, Eq. 9)
        delta_phi_nom: Nominal resolution (rad/m²): 2*sqrt(3) / (lambda²_max - lambda²_min)
        delta_phi: Resolution (rad/m²): full if lambda²_0=0, nominal if lambda²_0>0
    """
    nu: Union[np.ndarray, "da.Array"] = None
    lambda2: Union[np.ndarray, "da.Array"] = None
    data: Union[np.ndarray, "da.Array"] = None
    l2_ref: float = None
    w: Union[np.ndarray, "da.Array"] = None
    w_q: Union[np.ndarray, "da.Array"] = None
    w_u: Union[np.ndarray, "da.Array"] = None
    w_p: Union[np.ndarray, "da.Array"] = None
    sigma: Union[np.ndarray, "da.Array"] = None
    spectral_idx: float = None
    gridded: bool = None
    s: Union[np.ndarray, "da.Array"] = None
    model_data: Union[np.ndarray, "da.Array"] = None
    m: int = None
    theo_noise: float = None
    nu_0: float = None
    k: Union[float, "da.Array"] = None

    def __init__(
        self,
        nu=None,
        lambda2=None,
        data=None,
        l2_ref=None,
        w=None,
        w_q=None,
        w_u=None,
        sigma=None,
        spectral_idx=None,
        gridded=None,
    ):
        """
        Initialize Dataset.
        
        Args:
            nu: Frequency array (Hz). If None, lambda2 must be provided.
            lambda2: Wavelength squared array (m²). If None, computed from nu.
            data: Complex polarization P = Q + iU
            l2_ref: Reference lambda² (default: weighted mean)
            w: Main weight array (or set w_q, w_u for polarization weights)
            w_q: Stokes Q weights (optional, requires w_u)
            w_u: Stokes U weights (optional, requires w_q)
            sigma: Noise per channel (alternative to w)
            spectral_idx: Spectral index for correction (default: 0.0)
            gridded: Whether data is on regular grid (default: False)
        """
        self.k = None
        self.l2_ref = l2_ref

        if self.l2_ref is None:
            self.l2_ref = 0.0

        self.nu_0 = None
        self.delta_l2_min = 0.0
        self.delta_l2_max = 0.0
        self.delta_l2_mean = 0.0
        self.theo_noise = None
        self.__w = None
        self.__w_q = None
        self.__w_u = None
        self.__w_p = None
        self.lambda2 = lambda2
        self.nu = nu
        self.spectral_idx = spectral_idx
        self.gridded = gridded

        if self.gridded is None:
            self.gridded = False

        if self.nu is None and self.lambda2 is None:
            self.s = None

        if lambda2 is not None:
            self.m = length_of(self.lambda2)
        elif nu is not None:
            self.m = length_of(self.nu)
        else:
            self.m = None

        if sigma is None and w is None and w_q is None and w_u is None and self.m is not None:
            self.sigma = np.ones(self.m)
        elif w_q is not None and w_u is not None:
            self.w_q = w_q
            self.w_u = w_u
        elif w is not None:
            self.w = w
        else:
            self.sigma = sigma

        self.data = data
        self.residual = None
        if self.data is not None:
            if is_dask_array(self.data):
                self.model_data = da.zeros_like(self.data, dtype=self.data.dtype)
            else:
                self.model_data = np.zeros_like(self.data, dtype=self.data.dtype)
        else:
            self.model_data = None

    @property
    def spectral_idx(self) -> float:
        """Spectral index for flux correction (default: 0.0)."""
        return self.__spectral_idx

    @spectral_idx.setter
    def spectral_idx(self, val):
        """Set spectral index and update correction factor s."""
        if val is None:
            self.__spectral_idx = 0.0
        else:
            self.__spectral_idx = val

        if self.__lambda2 is not None and self.__nu_0 is not None:
            nu = c / np.sqrt(self.__lambda2)
            self.__s = (nu / self.__nu_0) ** self.__spectral_idx

    @property
    def s(self) -> Union[np.ndarray, "da.Array", None]:
        """Spectral index correction factor s = (nu/nu_0)^spectral_idx."""
        return self.__s

    @s.setter
    def s(self, val):
        """Set spectral correction factor and update normalization k."""
        self.__s = val
        if self.__s is not None:
            k_val = np.sum(self.w / self.__s)
            self.k = maybe_compute(k_val) if k_val is not None else k_val

    @property
    def nu_0(self) -> float:
        """Reference frequency (Hz), typically midpoint of band."""
        return self.__nu_0

    @nu_0.setter
    def nu_0(self, val):
        """Set reference frequency."""
        self.__nu_0 = val

    @property
    def nu(self) -> Union[np.ndarray, "da.Array", None]:
        """Frequency array (Hz)."""
        return self.__nu

    @nu.setter
    def nu(self, val):
        """Set frequency array and compute lambda² and reference frequency."""
        self.__nu = val
        if val is not None:
            mn, mx = maybe_compute(np.min(val)), maybe_compute(np.max(val))
            self.__nu_0 = 0.5 * (float(mn) + float(mx))
            self._nu_to_l2()

    @property
    def lambda2(self) -> Union[np.ndarray, "da.Array", None]:
        """Wavelength squared array (m²)."""
        return self.__lambda2

    @lambda2.setter
    def lambda2(self, val):
        """
        Set lambda² array, ensure ascending order, compute nu and reference frequency.
        
        Also initializes default weights if not set.
        """
        self.__lambda2 = val
        if val is not None:
            val_np = asnumpy(val) if is_dask_array(val) else np.asarray(val)
            if np.all(np.diff(val_np) < 0):
                val = val[::-1]
                self.__lambda2 = val
            self.__m = length_of(val)
            # Avoid divide-by-zero: nu = c/sqrt(lambda²) is invalid for lambda² <= 0
            val_np_safe = np.where(val_np > 0, val_np, np.nan)
            self.__nu = np.asarray(c / np.sqrt(val_np_safe), dtype=np.float64)
            if da is not None and is_dask_array(val):
                ch = getattr(val, "chunks", None)
                chunks = ch[0] if isinstance(ch, tuple) else "auto"
                self.__nu = da.from_array(self.__nu, chunks=chunks)
            nu_min = maybe_compute(np.nanmin(self.__nu))
            nu_max = maybe_compute(np.nanmax(self.__nu))
            if np.isfinite(nu_min) and np.isfinite(nu_max):
                self.__nu_0 = 0.5 * (float(nu_min) + float(nu_max))
            else:
                self.__nu_0 = np.nan
            if hasattr(self, "spectral_idx") and self.__spectral_idx is not None and np.isfinite(self.__nu_0):
                # Only compute s where nu is finite; otherwise use 1.0 (no spectral correction)
                with np.errstate(invalid="ignore"):
                    s_vals = (self.__nu / self.__nu_0) ** self.__spectral_idx
                self.__s = np.where(np.isfinite(self.__nu), s_vals, 1.0)
            elif hasattr(self, "spectral_idx") and self.__spectral_idx is not None:
                self.__s = np.ones_like(self.__nu)
            if da is not None and is_dask_array(val):
                ch = getattr(val, "chunks", None)
                chunks = ch[0] if isinstance(ch, tuple) else "auto"
                self.w = da.ones(self.__m, dtype=np.float64, chunks=chunks)
            else:
                self.w = np.ones(self.__m)
            self._calculate_l2_cellsize()

    @property
    def k(self) -> Union[float, "da.Array", None]:
        """Normalization factor: sum of weights (or sum of w/s if spectral correction)."""
        return self.__k

    @k.setter
    def k(self, val):
        """Set normalization factor."""
        self.__k = val

    @property
    def m(self) -> int:
        """Number of channels."""
        return self.__m

    @m.setter
    def m(self, val):
        """Set number of channels."""
        self.__m = val

    @property
    def theo_noise(self) -> float:
        """Theoretical noise level: 1/sqrt(sum(w))."""
        return self.__theo_noise

    @theo_noise.setter
    def theo_noise(self, val):
        """Set theoretical noise (usually computed automatically)."""
        self.__theo_noise = val

    @property
    def w(self) -> Union[np.ndarray, "da.Array", None]:
        """
        Main weight used in chi² and transforms.
        
        When w_q and w_u are set, this returns w_p (harmonic mean).
        """
        return self.__w

    @w.setter
    def w(self, val):
        """
        Set main weight array.
        
        Clears w_q, w_u, w_p. Updates sigma, k, l2_ref, theo_noise.
        """
        self.__w = val
        self.__w_q = None
        self.__w_u = None
        self.__w_p = None
        if val is not None:
            val_np = asnumpy(val)
            aux_copy = val_np.copy()
            aux_copy[aux_copy != 0] = 1.0 / np.sqrt(aux_copy[aux_copy != 0])
            self.__sigma = aux_copy
            if hasattr(self, "s") and self.__s is not None:
                k_val = np.sum(val / self.__s)
                self.k = maybe_compute(k_val) if hasattr(k_val, "compute") else k_val
            else:
                k_val = np.sum(val)
                self.k = maybe_compute(k_val) if hasattr(k_val, "compute") else k_val
            if self.__l2_ref is None:
                self.__l2_ref = self.calculate_l2ref()
        self.__theo_noise = self._calculate_theo_noise()

    @property
    def w_q(self) -> Union[np.ndarray, "da.Array", None]:
        """
        Weights for Stokes Q.
        
        When set with w_u, w_p is computed as harmonic mean and used as w.
        """
        return self.__w_q

    @w_q.setter
    def w_q(self, val):
        """
        Set Stokes Q weights.
        
        If w_u is also set, computes w_p and updates w, sigma, k, l2_ref, theo_noise.
        """
        self.__w_q = val
        if self.__w_u is not None and val is not None:
            self.__w_p = _harmonic_mean_w_p(val, self.__w_u)
            self.__w = self.__w_p
            val_np = asnumpy(self.__w)
            aux_copy = val_np.copy()
            aux_copy[aux_copy != 0] = 1.0 / np.sqrt(aux_copy[aux_copy != 0])
            self.__sigma = aux_copy
            if hasattr(self, "s") and self.__s is not None:
                k_val = np.sum(self.__w / self.__s)
                self.k = maybe_compute(k_val) if hasattr(k_val, "compute") else k_val
            else:
                k_val = np.sum(self.__w)
                self.k = maybe_compute(k_val) if hasattr(k_val, "compute") else k_val
            if self.__l2_ref is None:
                self.__l2_ref = self._calculate_l2ref()
            self.__theo_noise = self._calculate_theo_noise()

    @property
    def w_u(self) -> Union[np.ndarray, "da.Array", None]:
        """
        Weights for Stokes U.
        
        When set with w_q, w_p is computed as harmonic mean and used as w.
        """
        return self.__w_u

    @w_u.setter
    def w_u(self, val):
        """
        Set Stokes U weights.
        
        If w_q is also set, computes w_p and updates w, sigma, k, l2_ref, theo_noise.
        """
        self.__w_u = val
        if self.__w_q is not None and val is not None:
            self.__w_p = _harmonic_mean_w_p(self.__w_q, val)
            self.__w = self.__w_p
            val_np = asnumpy(self.__w)
            aux_copy = val_np.copy()
            aux_copy[aux_copy != 0] = 1.0 / np.sqrt(aux_copy[aux_copy != 0])
            self.__sigma = aux_copy
            if hasattr(self, "s") and self.__s is not None:
                k_val = np.sum(self.__w / self.__s)
                self.k = maybe_compute(k_val) if hasattr(k_val, "compute") else k_val
            else:
                k_val = np.sum(self.__w)
                self.k = maybe_compute(k_val) if hasattr(k_val, "compute") else k_val
            if self.__l2_ref is None:
                self.__l2_ref = self._calculate_l2ref()
            self.__theo_noise = self._calculate_theo_noise()

    @property
    def w_p(self) -> Union[np.ndarray, "da.Array", None]:
        """
        Polarization weight (harmonic mean of w_q and w_u when both are set).
        
        Read-only when derived from w_q, w_u.
        """
        return self.__w_p

    @property
    def l2_ref(self) -> float:
        """Reference lambda² for phase (default: weighted mean)."""
        return self.__l2_ref

    @l2_ref.setter
    def l2_ref(self, val):
        """Set reference lambda²."""
        self.__l2_ref = val

    @property
    def sigma(self) -> Union[np.ndarray, "da.Array", None]:
        """Noise per channel (derived from w: sigma = 1/sqrt(w))."""
        return self.__sigma

    @sigma.setter
    def sigma(self, val):
        """Set noise array (converts to weights: w = 1/sigma²)."""
        if val is not None:
            self.w = 1.0 / (val**2)
        self.__sigma = val

    @property
    def data(self) -> Union[np.ndarray, "da.Array", None]:
        """Complex polarization data P = Q + iU."""
        return self.__data

    @data.setter
    def data(self, val):
        """
        Set polarization data.
        
        Validates size matches m. Initializes model_data if needed.
        """
        if val is not None:
            n = length_of(val)
            if n == self.m:
                self.__data = val
            else:
                self.__m = n
                self.__data = val
            if hasattr(self, "model_data"):
                if self.__model_data is None:
                    dt = getattr(val, "dtype", np.complex64)
                    if is_dask_array(val):
                        self.__model_data = da.zeros_like(val, dtype=dt)
                    else:
                        self.__model_data = np.zeros_like(val, dtype=dt)
        else:
            self.__data = None

    @property
    def model_data(self) -> Union[np.ndarray, "da.Array", None]:
        """Model prediction (same shape as data)."""
        return self.__model_data

    @model_data.setter
    def model_data(self, val):
        """
        Set model prediction.
        
        Validates size matches m. Automatically computes residuals if data is set.
        
        Raises:
            ValueError: If size doesn't match m
        """
        if val is not None:
            if length_of(val) == self.m:
                self.__model_data = val
                if self.data is not None:
                    self._calculate_residuals()
            else:
                raise ValueError("Data must have same size as lambda2")
        else:
            self.__model_data = None

    def _nu_to_l2(self):
        """
        Convert frequency to lambda² and set lambda2 property.
        
        Private method: called automatically when nu is set.
        """
        lambda2 = (c / self.nu)**2
        self.lambda2 = lambda2[::-1]

    def calculate_amplitude(self, column: str = "data") -> np.ndarray:
        """
        Calculate amplitude |P| from complex polarization data.
        
        Args:
            column: Column name to use (default: "data")
            
        Returns:
            Amplitude array
            
        Raises:
            TypeError: If data is not complex
            ValueError: If column doesn't exist
        """
        if hasattr(self, column):
            data = getattr(self, column)
            if data.dtype == np.complex64 or data.dtype == np.complex128:
                amplitude = np.abs(data)
                return amplitude
            else:
                raise TypeError("Data is not complex")
        else:
            raise ValueError("Column does not exist")

    def calculate_polangle(self, column: str = "data") -> u.Quantity:
        """
        Calculate polarization angle from complex data.
        
        Args:
            column: Column name to use (default: "data")
            
        Returns:
            Polarization angle in radians (astropy Quantity)
            
        Raises:
            TypeError: If data is not complex
            ValueError: If column doesn't exist
        """
        if hasattr(self, column):
            data = getattr(self, column)
            if data.dtype == np.complex64 or data.dtype == np.complex128:
                pol_angle = 0.5 * np.arctan2(self.data.imag, self.data.real)
                return pol_angle * u.rad
            else:
                raise TypeError("Data is not complex")
        else:
            raise ValueError("Column does not exist")

    def calculate_l2ref(self) -> float:
        """
        Calculate reference lambda² as weighted mean.
        
        Public method: can be called to recompute l2_ref. Also called automatically
        when weights are set (via private _calculate_l2ref).
        
        Returns:
            Reference lambda² or None if lambda2 is not set
        """
        if self.lambda2 is not None:
            sum_weights = maybe_compute(np.sum(self.w))
            weighted_l2 = maybe_compute(np.sum(self.w * self.lambda2))
            return float(weighted_l2) / float(sum_weights) if sum_weights else None
        else:
            return None

    def _calculate_l2ref(self) -> float:
        """
        Internal wrapper for calculate_l2ref (for use in setters).
        
        Private method: called automatically when weights are set.
        """
        return self.calculate_l2ref()

    def _calculate_l2_cellsize(self):
        """
        Calculate lambda² cell size statistics (min, mean, max).
        
        Private method: called automatically when lambda2 is set.
        Updates delta_l2_min, delta_l2_max, delta_l2_mean.
        """
        if self.w is not None:
            w_np = asnumpy(self.w)
            l2_np = asnumpy(self.lambda2)
            lambda2_aux = l2_np[w_np > 0.0]
            if len(lambda2_aux) < 2:
                return
            diff_aux = np.abs(np.diff(lambda2_aux))
            delta_l2_min = float(np.min(diff_aux))
            delta_l2_mean = float(np.mean(diff_aux))
            delta_l2_max = float(np.max(diff_aux))

            self.delta_l2_min = delta_l2_min
            self.delta_l2_max = delta_l2_max
            self.delta_l2_mean = delta_l2_mean

    @property
    def delta_phi_full(self) -> float:
        """
        Full resolution (rad/m²): 2 / (lambda²_max + lambda²_min).
        
        Used when lambda²_0 = 0. This is the FWHM of the real beam peak
        for full resolution Faraday synthesis (Rudnick & Cotton 2023, Eq. 9).
        
        References:
            Rudnick & Cotton (2023), MNRAS, Eq. (9):
            Φ_full ≈ 2 / (λ²_max + λ²_min)
        
        Returns:
            Full resolution in rad/m², or None if lambda2 is not set
        """
        if self.lambda2 is None:
            return None
        l2_np = asnumpy(self.lambda2)
        if self.w is not None:
            w_np = asnumpy(self.w)
            l2_nonzero = l2_np[w_np > 0.0]
            if len(l2_nonzero) == 0:
                l2_nonzero = l2_np
        else:
            l2_nonzero = l2_np
        l2_min = float(np.min(l2_nonzero))
        l2_max = float(np.max(l2_nonzero))
        return 2.0 / (l2_max + l2_min)

    @property
    def delta_phi_nom(self) -> float:
        """
        Nominal resolution (rad/m²): 2 * sqrt(3) / (lambda²_max - lambda²_min).
        
        Used when lambda²_0 > 0. This is the FWHM of the RMTF (Rotation Measure
        Transfer Function), also known as the nominal resolution.
        
        Returns:
            Nominal resolution in rad/m², or None if lambda2 is not set
        """
        if self.lambda2 is None:
            return None
        l2_np = asnumpy(self.lambda2)
        if self.w is not None:
            w_np = asnumpy(self.w)
            l2_nonzero = l2_np[w_np > 0.0]
            if len(l2_nonzero) == 0:
                l2_nonzero = l2_np
        else:
            l2_nonzero = l2_np
        l2_min = float(np.min(l2_nonzero))
        l2_max = float(np.max(l2_nonzero))
        delta_l2 = l2_max - l2_min
        if delta_l2 <= 0:
            return None
        return 2.0 * np.sqrt(3.0) / delta_l2

    @property
    def delta_phi(self) -> float:
        """
        Resolution (rad/m²): full resolution if lambda²_0 = 0, nominal resolution if lambda²_0 > 0.
        
        Used by Parameter.calculate_cellsize for the phi grid. With l2_ref > 0 the grid is
        coarser (nominal), so the peak of the dirty/restored Faraday spectrum can be higher
        than with l2_ref = 0 (full resolution); integrated flux is consistent.
        
        Returns:
            Full resolution if l2_ref == 0, nominal resolution if l2_ref > 0,
            or None if lambda2 is not set
        """
        if self.lambda2 is None:
            return None
        l2_ref = self.l2_ref if self.l2_ref is not None else 0.0
        if abs(l2_ref) < 1e-10:  # Effectively zero
            return self.delta_phi_full
        else:
            return self.delta_phi_nom

    def _calculate_theo_noise(self) -> float:
        """
        Calculate theoretical noise: 1/sqrt(sum(w)).
        
        Private method: called automatically when weights are set.
        
        Returns:
            Theoretical noise or None if w is None or all weights are 1
        """
        if self.w is None:
            return None
        else:
            w_sum = np.sum(self.w)
            w_sum = maybe_compute(w_sum)
            if w_sum is None:
                return None
            if float(w_sum) == float(self.m):  # all ones
                return None
            return 1.0 / np.sqrt(float(w_sum))

    def _calculate_residuals(self):
        """
        Calculate residuals: data - model_data.
        
        Private method: called automatically when model_data is set.
        """
        self.residual = self.data - self.model_data

    def subtract_galacticrm(self, phi_gal: float):
        """
        Subtract Galactic rotation measure from data.
        
        Multiplies data by exp(-2j * phi_gal * lambda²) to remove Galactic RM.
        
        Args:
            phi_gal: Galactic rotation measure (rad/m²)
        """
        p = self.data
        galrm_shift = np.exp(-2j * phi_gal * self.lambda2)
        p_hat = p * galrm_shift
        self.data = p_hat

    def assess_residuals(self, gridding_object: "Gridding" = None, confidence_interval: float = 0.95) -> Tuple:
        """
        Assess residual autocorrelation for quality control.
        
        Computes autocorrelation of residuals (and squared residuals) and checks
        if values fall within confidence bounds (Ljung-Box style).
        
        Args:
            gridding_object: Gridding transformer (required if not gridded)
            confidence_interval: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (lags, autocorr_res, autocorr_res_sq, bound,
                     percentage_real_in, percentage_imag_in,
                     percentage_real_in_sq, percentage_imag_in_sq)
        """
        if self.gridded:
            autocorr_real = _autocorr_gridded(self.residual.real)
            autocorr_imag = _autocorr_gridded(self.residual.imag)
            autocorr_real_sq = _autocorr_gridded(self.residual.real**2)
            autocorr_imag_sq = _autocorr_gridded(self.residual.imag**2)
            lags = sci_signal.correlation_lags(self.m, self.m, mode="full")
        else:
            # Grid the irregular data
            gridding = gridding_object
            gridded_data = gridding.run()
            autocorr_real = _autocorr_gridded(gridded_data.residual.real)
            autocorr_imag = _autocorr_gridded(gridded_data.residual.imag)
            autocorr_real_sq = _autocorr_gridded(gridded_data.residual.real**2)
            autocorr_imag_sq = _autocorr_gridded(gridded_data.residual.imag**2)
            lags = sci_signal.correlation_lags(gridded_data.m, gridded_data.m, mode="full")

        autocorr_res = autocorr_real + 1j * autocorr_imag
        autocorr_res_sq = autocorr_real_sq + 1j * autocorr_imag_sq

        lags_pos = np.where(lags >= 0)
        lags = lags[lags_pos]

        vcrit = np.sqrt(2) * special.erfinv(confidence_interval)
        bound = vcrit / np.sqrt(self.m)

        elem_real = ((autocorr_res.real > -bound) & (autocorr_res.real < bound)).sum()
        percentage_real_in = 100.0 * elem_real / len(lags)
        elem_imag = ((autocorr_res.imag > -bound) & (autocorr_res.imag < bound)).sum()
        percentage_imag_in = 100.0 * elem_imag / len(lags)

        elem_real_sq = ((autocorr_res_sq.real > -bound) & (autocorr_res_sq.real < bound)).sum()
        percentage_real_in_sq = 100.0 * elem_real_sq / len(lags)
        elem_imag_sq = ((autocorr_res_sq.imag > -bound) & (autocorr_res.imag < bound)).sum()
        percentage_imag_in_sq = 100.0 * elem_imag_sq / len(lags)

        return (
            lags,
            autocorr_res,
            autocorr_res_sq,
            bound,
            percentage_real_in,
            percentage_imag_in,
            percentage_real_in_sq,
            percentage_imag_in_sq,
        )

    def histogram_residuals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute histograms of residual real and imaginary parts.
        
        Returns:
            Tuple of (hist_real, bins_real, hist_imag, bins_imag)
        """
        hist_real, bins_real = np.histogram(self.residual.real, bins="auto")
        hist_imag, bins_imag = np.histogram(self.residual.imag, bins="auto")

        # You can plot the histograms this way:
        # plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor="black", align="edge")

        return hist_real, bins_real, hist_imag, bins_imag
