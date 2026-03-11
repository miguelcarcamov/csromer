"""
CS-ROMER reconstructor: FISTA-based sparse reconstruction with L1 regularization.

Minimizes ChiSquared + L1 using FISTA optimizer. Supports wavelet transforms for
sparse representation. Computes dirty map, model, residual, and restored maps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from astropy.stats import sigma_clipped_stats

from ...dictionaries import Wavelet
from ...objectivefunction import L1, TSV, TV, ChiSquared, OFunction
from ...optimization import FISTA
from ...reconstruction import Parameter
from ...transformers.dfts import NDFT1D, NUFFT1D, GriddedFFT1D
from ...transformers.flaggers.flagger import Flagger
from .faraday_reconstructor import FaradayReconstructorWrapper

if TYPE_CHECKING:
    from ...transformers.measurement_operator import MeasurementOperator


@dataclass(init=True, repr=True)
class CSROMERReconstructorWrapper(FaradayReconstructorWrapper):
    """
    CS-ROMER reconstructor: FISTA-based sparse reconstruction.

    Minimizes ChiSquared + L1 using FISTA optimizer. Supports optional wavelet
    transforms for sparse representation. Computes dirty map, model, residual,
    and restored maps with error estimates.

    Attributes:
        parameter: Parameter object (Faraday depth space)
        flagger: Optional flagger for data quality control
        measurement_operator: Single operator for forward/adjoint/dirty/RMTF (built from fourier_mode if None)
        wavelet: Optional wavelet transform
        coefficients: Wavelet coefficients (if wavelet used)
        fd_restored: Restored Faraday depth spectrum
        rm_restored: Rotation measure at restored peak
        rm_restored_error: Error on rm_restored
        restored_peak_quadratic_interpolation: Peak value from quadratic interpolation
        rm_restored_quadratic_interpolation: RM from quadratic interpolation
        rm_restored_quadratic_interpolation_error: Error on quadratic interpolation RM
        fd_model: Model Faraday depth spectrum
        rm_model: Rotation measure at model peak
        fd_residual: Residual Faraday depth spectrum
        fd_dirty: Dirty Faraday depth spectrum
        rm_dirty: Rotation measure at dirty peak
        rm_dirty_error: Error on rm_dirty
        dirty_peak_quadratic_interpolation: Peak value from quadratic interpolation
        rm_dirty_quadratic_interpolation: RM from quadratic interpolation
        rm_dirty_quadratic_interpolation_error: Error on quadratic interpolation RM
        second_moment: Second moment of model
        cellsize: Grid spacing (rad/m², optional)
        oversampling: Oversampling factor (default: 7.0)
        lambda_l_norm: L1 regularization factor (required for L1; use 0.0 for Chi-squared only)
        calculate_l2_zero: Whether to compute l2_ref (default: False)
        fista_maxiter: Max FISTA iterations (optional; default 500)
        fista_tol: FISTA tolerance (optional)
        fista_verbose: FISTA verbose output (default: True)
        fista_step: FISTA gradient step size (optional; backtracking if None)
        fista_monotonic: If True, use MFISTA (monotone FISTA, reject non-decreasing steps)
    """
    parameter: Parameter = field(init=False)
    flagger: Flagger = None
    measurement_operator: Optional["MeasurementOperator"] = None
    wavelet: Wavelet = None
    coefficients: np.ndarray = field(init=False)
    fd_restored: np.ndarray = field(init=False)
    rm_restored: float = field(init=False)
    rm_restored_error: float = field(init=False)
    restored_peak_quadratic_interpolation: float = field(init=False)
    rm_restored_quadratic_interpolation: float = field(init=False)
    rm_restored_quadratic_interpolation_error: float = field(init=False)
    fd_model: np.ndarray = field(init=False)
    rm_model: float = field(init=False)
    fd_residual: np.ndarray = field(init=False)
    fd_dirty: np.ndarray = field(init=False)
    rm_dirty: float = field(init=False)
    rm_dirty_error: float = field(init=False)
    dirty_peak_quadratic_interpolation: float = field(init=False)
    rm_dirty_quadratic_interpolation: float = field(init=False)
    rm_dirty_quadratic_interpolation_error: float = field(init=False)
    second_moment: float = field(init=False)
    cellsize: float = None
    oversampling: float = None
    lambda_l_norm: float = None  # L1 reg; None => 0.0 (Chi-squared only, no L1)
    calculate_l2_zero: bool = None
    fista_maxiter: int = None
    fista_tol: float = None
    fista_verbose: bool = True
    fista_step: float = None  # Gradient step. None => backtracking.
    fista_monotonic: bool = False  # If True, use MFISTA.
    # How to build measurement_operator when measurement_operator is None: "direct" (NDFT1D),
    # "nufft" (NUFFT1D), or "gridded" (grid data then GriddedFFT1D). Pass measurement_operator
    # explicitly to use a custom operator (e.g. pre-built GriddedFFT1D).
    fourier_mode: str = "direct"

    def __post_init__(self):
        """
        Post-initialization: set defaults, configure parameter space and operators.
        """
        if self.oversampling is None:
            self.oversampling = 7.0

        if self.calculate_l2_zero is None:
            self.calculate_l2_zero = False

        if self.calculate_l2_zero:
            print("Calculating l2_0")
            self.dataset.l2_ref = self.dataset.calculate_l2ref()

        self.parameter = Parameter()
        self.config_fd_space(self.cellsize, self.oversampling)
        # #region agent log
        try:
            import json
            _m = getattr(self.dataset, "m", None)
            _mode = getattr(self, "fourier_mode", "direct")
            _log = {"sessionId": "95531f", "hypothesisId": "H4", "location": "csromer_reconstructor.py:__post_init__", "message": "Before config_fourier_transforms", "data": {"dataset_m": _m, "fourier_mode": _mode}, "timestamp": __import__("time").time() * 1000}
            open("/home/miguel/Documents/csromer/.cursor/debug-95531f.log", "a").write(json.dumps(_log) + "\n")
        except Exception:
            pass
        # #endregion
        self.config_fourier_transforms()

    @staticmethod
    def estimate_peak_quadratic_interpolation(fd_signal: np.ndarray, cellsize: float) -> tuple:
        """
        Estimate peak location and value using quadratic interpolation.

        Public static method. Fits quadratic to peak and neighbors to sub-pixel accuracy.

        Args:
            fd_signal: Faraday depth spectrum
            cellsize: Grid spacing (rad/m²)

        Returns:
            Tuple of (phi_peak, peak_value)
        """
        length_n = len(fd_signal)
        index_0 = int(np.argmax(np.abs(fd_signal)))

        # Need both neighbors for quadratic interpolation; at boundaries use raw peak
        if index_0 <= 0 or index_0 >= length_n - 1:
            location = float(index_0)
            estimated_peak = float(np.abs(fd_signal[index_0]))
            pos_phi_peak = (location - length_n / 2) * cellsize
            return pos_phi_peak, estimated_peak

        fd_signal_0 = np.abs(fd_signal[index_0])
        fd_signal_m1 = np.abs(fd_signal[index_0 - 1])
        fd_signal_p1 = np.abs(fd_signal[index_0 + 1])

        pos_estimated_peak = (fd_signal_p1 - fd_signal_m1
                              ) / (4 * fd_signal_0 - 2 * fd_signal_m1 - 2 * fd_signal_p1)

        estimated_peak = (fd_signal_0 - 0.25 * (fd_signal_m1 - fd_signal_p1) * pos_estimated_peak)

        location = index_0 + pos_estimated_peak

        pos_phi_peak = (location - length_n / 2) * cellsize

        return pos_phi_peak, estimated_peak

    @staticmethod
    def calculate_ricean_peak(peak: float, noise: float) -> float:
        """
        Calculate Ricean-corrected peak value.

        Public static method. Corrects for Ricean bias in peak estimation.

        Args:
            peak: Observed peak value
            noise: Noise level

        Returns:
            Ricean-corrected peak value
        """
        ricean_peak = np.sqrt(peak**2 - (2.3 * noise**2))
        return ricean_peak

    @staticmethod
    def calculate_fd_signal_noise(
        fd_signal: np.ndarray,
        phi: np.ndarray,
        max_fd_depth: float,
        threshold: float = 0.0,
        sigma: float = 0.3,
        cenfunc: str = 'mean',
        stdfunc: str = 'mad_std'
    ) -> float:
        """
        Calculate noise level in Faraday depth signal.

        Public static method. Uses sigma-clipped statistics on edge regions
        (where |phi| > max_fd_depth * threshold) to estimate background noise.

        Args:
            fd_signal: Faraday depth spectrum
            phi: Faraday depth grid
            max_fd_depth: Maximum Faraday depth
            threshold: Threshold for edge masking (default: 0.0)
            sigma: Sigma clipping threshold (default: 0.3)
            cenfunc: Center function for clipping (default: 'mean')
            stdfunc: Std function for clipping (default: 'mad_std')

        Returns:
            Estimated noise level (or small positive value if calculation fails)
        """
        # Select edge regions for noise estimation (where |phi| > threshold * max_fd_depth)
        # We want to mask the center (signal) and use edges (noise)
        # So we select points where |phi| > threshold * max_fd_depth
        edge_mask = np.abs(phi) > max_fd_depth * threshold
        n_edge_points = np.sum(edge_mask)

        # Need sufficient points for sigma clipping to work properly
        # If threshold=0.0 leaves too few edge points, use progressively larger thresholds
        # to get more edge points (further from center where signal is)
        if n_edge_points < 10:
            # Use outer 50% of phi range (|phi| > 0.5 * max_fd_depth)
            edge_mask = np.abs(phi) > max_fd_depth * 0.5
            n_edge_points = np.sum(edge_mask)
            if n_edge_points < 10:
                # Use outer 80% of phi range (|phi| > 0.2 * max_fd_depth)
                edge_mask = np.abs(phi) > max_fd_depth * 0.2
                n_edge_points = np.sum(edge_mask)
                if n_edge_points < 10:
                    # Last resort: use outer 90% (|phi| > 0.1 * max_fd_depth)
                    edge_mask = np.abs(phi) > max_fd_depth * 0.1
                    n_edge_points = np.sum(edge_mask)
                    if n_edge_points < 10:
                        # Very few points available, use all points
                        edge_mask = np.ones_like(phi, dtype=bool)
                        n_edge_points = len(phi)

        # Extract edge region data for noise analysis
        edge_real = fd_signal.real[edge_mask]
        edge_imag = fd_signal.imag[edge_mask]

        def robust_rms_estimate(data, sigma_val, cenfunc_val, stdfunc_val):
            """
            Robustly estimate RMS, avoiding warnings from sigma_clipped_stats.

            Uses sigma clipping only when data has sufficient points and variance.
            Otherwise falls back to simple std to avoid warnings.
            """
            n_points = len(data)

            # Need at least 10 points for sigma clipping to work reliably
            if n_points < 10:
                return np.std(data) if n_points > 1 else 0.0

            # Check initial statistics
            initial_std = np.std(data)
            initial_mean = np.mean(data)

            # If variance is too low, sigma clipping will remove all points
            # Use a threshold based on the data range
            data_range = np.max(data) - np.min(data)
            if initial_std < 1e-10 or data_range < 1e-10:
                return initial_std

            # Check if data is too uniform (all values very close)
            # If coefficient of variation is very small, skip sigma clipping
            if abs(initial_mean) > 1e-10:
                cv = initial_std / abs(initial_mean)
                if cv < 1e-6:
                    return initial_std

            # For sigma clipping to work without warnings, we need:
            # 1. Enough points (>= 10, already checked)
            # 2. Sufficient variance (checked above)
            # 3. Use conservative maxiters to avoid removing all points
            # Use maxiters=1 to be very conservative - only one iteration
            # This prevents the iterative removal that causes warnings
            try:
                _, _, rms = sigma_clipped_stats(
                    data, sigma=sigma_val, cenfunc=cenfunc_val, stdfunc=stdfunc_val, maxiters=1
                )
                # Verify result is valid and reasonable
                if np.isfinite(rms) and rms > 0 and rms <= initial_std * 10:
                    return rms
                else:
                    # Result is invalid or unreasonable, use std
                    return initial_std
            except (ValueError, RuntimeError):
                # Exception occurred, use std
                return initial_std

        # Use robust RMS estimation for both real and imaginary parts from edge regions
        background_real_rms = robust_rms_estimate(edge_real, sigma, cenfunc, stdfunc)
        background_imag_rms = robust_rms_estimate(edge_imag, sigma, cenfunc, stdfunc)

        # Final fallback if we somehow got invalid results
        if not np.isfinite(background_real_rms) or background_real_rms == 0:
            background_real_rms = np.std(edge_real) if len(edge_real) > 1 else np.std(fd_signal.real)
        if not np.isfinite(background_imag_rms) or background_imag_rms == 0:
            background_imag_rms = np.std(edge_imag) if len(edge_imag) > 1 else np.std(fd_signal.imag)

        fd_signal_noise = 0.5 * (background_real_rms + background_imag_rms)

        # Ensure non-zero noise to avoid NaN in error calculations
        # Use a small fraction of the peak as minimum noise estimate
        if fd_signal_noise == 0 or not np.isfinite(fd_signal_noise):
            fd_peak = np.max(np.abs(fd_signal))
            fd_signal_noise = fd_peak * 1e-6  # Use 1e-6 of peak as minimum noise

        return fd_signal_noise

    @staticmethod
    def calculate_sigma_phi_peak(rmtf_fwhm: float, fd_peak: float, fd_signal_noise: float) -> float:
        """
        Calculate error on rotation measure peak.

        Public static method. Uses RMTF FWHM and signal-to-noise ratio.
        Returns NaN if peak or noise is zero (avoids divide by zero).

        Args:
            rmtf_fwhm: RMTF FWHM (rad/m²)
            fd_peak: Peak Faraday depth value
            fd_signal_noise: Noise level

        Returns:
            Error on RM peak (rad/m²) or NaN if invalid
        """
        denom = 2.0 * fd_peak
        if denom == 0 or fd_signal_noise == 0:
            return np.nan
        sigma_phi_peak = rmtf_fwhm * fd_signal_noise / denom
        return sigma_phi_peak

    def flag_dataset(self, flagger: Flagger = None) -> tuple:
        """
        Flag dataset using flagger.

        Public method. Applies flagging to remove outliers.

        Args:
            flagger: Flagger instance (default: self.flagger)

        Returns:
            Tuple of (indexes, outliers_indexes)
        """
        if flagger is None:
            indexes, outliers_indexes = self.flagger.run()
        else:
            indexes, outliers_indexes = flagger.run()

        return indexes, outliers_indexes

    def config_fd_space(self, cellsize: float = None, oversampling: float = None):
        """
        Configure Faraday depth space (grid and cellsize).

        Public method (required by abstract base). Called during initialization.
        Computes optimal cellsize and phi grid from dataset.

        Args:
            cellsize: Grid spacing (rad/m², optional)
            oversampling: Oversampling factor (optional)

        Raises:
            ValueError: If both cellsize and oversampling are None
        """
        if cellsize is not None and oversampling is not None:
            self.parameter.calculate_cellsize(dataset=self.dataset, cellsize=cellsize)
        elif cellsize is None and oversampling is not None:
            self.parameter.calculate_cellsize(dataset=self.dataset, oversampling=oversampling)
        elif cellsize is not None and oversampling is None:
            self.parameter.calculate_cellsize(dataset=self.dataset, cellsize=cellsize)
        else:
            raise ValueError("Either cellsize or oversampling cannot be Nonetype values")

    def config_fourier_transforms(self):
        """
        Configure the measurement operator.

        Public method. Called during initialization. If measurement_operator is
        already set (e.g. passed by caller), it is left as is. Otherwise builds
        one from fourier_mode: "direct" (NDFT1D), "nufft" (NUFFT1D), or "gridded"
        (grid data onto regular λ² then GriddedFFT1D).
        """
        if self.measurement_operator is not None:
            return
        mode = (self.fourier_mode or "direct").lower()
        if mode == "direct":
            self.measurement_operator = NDFT1D(
                dataset=self.dataset, parameter=self.parameter
            )
        elif mode == "nufft":
            self.measurement_operator = NUFFT1D(
                dataset=self.dataset, parameter=self.parameter, solve=True
            )
        elif mode == "gridded":
            from ...transformers.gridding import Gridding

            # Nyquist d_phi * d_lambda2 = π/N: derive d_l2 from (full-resolution) cellsize
            d_l2 = np.pi / (self.parameter.n * self.parameter.cellsize)
            gridding = Gridding(
                dataset=self.dataset,
                d_lambda2=d_l2,
                n=self.parameter.n,
            )
            self.dataset = gridding.run()
            # Restore beam = Nyquist resolution (cellsize), so dirty and restored FWHM align
            self.parameter.rmtf_fwhm = np.pi / (self.parameter.n * d_l2)
            self.measurement_operator = GriddedFFT1D(
                dataset=self.dataset, parameter=self.parameter
            )
        else:
            raise ValueError(
                "fourier_mode must be 'direct', 'nufft', or 'gridded', got %r" % self.fourier_mode
            )

    def get_dirty_faraday_depth(self) -> np.ndarray:
        """
        Compute dirty Faraday depth spectrum.

        Public method. Returns A^H(weighted data) / K.

        Returns:
            Dirty Faraday depth spectrum (n_phi,)
        """
        return self.measurement_operator.dirty_spectrum(self.dataset.data)

    def get_rmtf(self) -> np.ndarray:
        """
        Get Rotation Measure Transfer Function.

        Public method.

        Returns:
            RMTF array (n_phi,)
        """
        return self.measurement_operator.RMTF()

    def get_rm(self, fd_data: np.ndarray) -> float:
        """
        Get rotation measure at peak of Faraday depth spectrum.

        Public method. Finds peak location and returns phi value at that location.
        Uses maybe_compute so that dask arrays are realized before argmax (avoids
        wrong peak from chunked argmax or lazy evaluation).

        Args:
            fd_data: Faraday depth spectrum (numpy or dask)

        Returns:
            Rotation measure (rad/m²)
        """
        from ...utils.array_utils import maybe_compute
        fd_abs = np.asarray(maybe_compute(np.abs(fd_data)))
        idx = int(np.argmax(fd_abs))
        phi = np.asarray(maybe_compute(self.parameter.phi))
        return float(phi[idx])

    def reconstruct(self):
        """
        Run FISTA reconstruction.

        Public method. Performs full reconstruction pipeline:
        1. Flag data (if flagger set)
        2. Compute dirty map and statistics
        3. Optimize ChiSquared + L1 with FISTA
        4. Compute model, residual, restored maps and statistics

        Sets attributes: fd_dirty, rm_dirty, fd_model, rm_model, fd_residual,
        fd_restored, rm_restored, and error estimates.
        """
        if self.flagger:
            self.flag_dataset()

        fd_dirty = self.get_dirty_faraday_depth()
        dirty_noise = self.calculate_fd_signal_noise(
            fd_dirty, self.parameter.phi, self.parameter.max_faraday_depth
        )

        self.fd_dirty = fd_dirty
        # Optimizer works in Jy/phi_pixel; start from complex zeros (not from dirty).
        n_phi = self.parameter.n
        self.parameter.data = np.zeros(n_phi, dtype=np.complex64)
        self.rm_dirty = self.get_rm(fd_dirty)
        self.rm_dirty_error = self.calculate_sigma_phi_peak(
            self.parameter.rmtf_fwhm, np.max(np.abs(fd_dirty)), dirty_noise
        )
        (
            self.rm_dirty_quadratic_interpolation,
            self.dirty_peak_quadratic_interpolation,
        ) = self.estimate_peak_quadratic_interpolation(fd_dirty, self.parameter.cellsize)
        self.rm_dirty_quadratic_interpolation_error = self.calculate_sigma_phi_peak(
            self.parameter.rmtf_fwhm, self.dirty_peak_quadratic_interpolation, dirty_noise
        )
        # When using wavelets, optimize in coefficient space; otherwise complex Faraday throughout
        if self.wavelet is not None:
            self.parameter.data = self.wavelet.decompose_complex(self.parameter.data)
        # Faraday depth kept as complex; no real stacking

        if self.lambda_l_norm is None:
            self.lambda_l_norm = 0.0  # No L1; set lambda_l_norm explicitly for sparse reconstruction

        # Use same operator as dirty map so scaling matches and backtracking finds reasonable steps
        chi_squared = ChiSquared(
            measurement_operator=self.measurement_operator, wavelet=self.wavelet
        )
        l1 = L1(reg=self.lambda_l_norm)

        F_func = [chi_squared, l1]
        F_obj = OFunction(F_func, persist_gradient=True)

        fista_kw = dict(
            guess_param=self.parameter,
            F_obj=F_obj,
            verbose=self.fista_verbose,
            monotonic=self.fista_monotonic,
        )
        if self.fista_maxiter is not None:
            fista_kw["maxiter"] = self.fista_maxiter
        if self.fista_tol is not None:
            fista_kw["tol"] = self.fista_tol
        if self.fista_step is not None:
            fista_kw["step"] = self.fista_step
        opt = FISTA(**fista_kw)

        obj, X = opt.run()

        self.coefficients = X.data
        if self.wavelet is not None:
            X.data = self.wavelet.reconstruct_complex(X.data)

        # Model and dirty/residual in same Faraday-depth space
        self.fd_model = X.data
        self.rm_model = self.get_rm(self.fd_model)
        self.second_moment = self.calculate_second_moment()

        # Residual: dirty of (data - model_data), in Jy/rmtf (same as dirty)
        self.fd_residual = self.measurement_operator.dirty_spectrum(
            self.dataset.data - self.dataset.model_data
        )

        # Model is Jy/phi_pixel; dirty and residual are Jy/rmtf. Same restoration as CG:
        # scale convolved map (in Jy/rmtf) so its peak matches dirty peak, then add residual.
        conv_model = self.parameter.convolve(x=self.fd_model)
        pixels_per_rmtf = self.parameter.rmtf_fwhm / self.parameter.cellsize

        from ...utils.array_utils import maybe_compute
        def _peak(a):
            a = np.asarray(maybe_compute(a))
            return float(np.max(np.abs(a)))
        conv_Jy_rmtf = conv_model * pixels_per_rmtf
        conv_peak = _peak(conv_Jy_rmtf)
        dirty_peak = _peak(self.fd_dirty)
        amp_scale = (dirty_peak / conv_peak) if conv_peak > 1e-30 else 1.0
        self.fd_restored = conv_Jy_rmtf * amp_scale + self.fd_residual
        # self.fd_restored = conv_Jy_rmtf + self.fd_residual

        restored_noise = self.calculate_fd_signal_noise(
            self.fd_restored, self.parameter.phi, self.parameter.max_faraday_depth
        )

        self.rm_restored = self.get_rm(self.fd_restored)
        self.rm_restored_error = self.calculate_sigma_phi_peak(
            self.parameter.rmtf_fwhm, np.max(np.abs(self.fd_restored)), restored_noise
        )
        (
            self.rm_restored_quadratic_interpolation,
            self.restored_peak_quadratic_interpolation,
        ) = self.estimate_peak_quadratic_interpolation(self.fd_restored, self.parameter.cellsize)
        self.rm_restored_quadratic_interpolation_error = self.calculate_sigma_phi_peak(
            self.parameter.rmtf_fwhm, self.restored_peak_quadratic_interpolation, restored_noise
        )

    def calculate_second_moment(self) -> float:
        """
        Calculate second moment of model (width measure).

        Public method. Computes weighted second moment around first moment.

        Returns:
            Second moment (rad²/m⁴)
        """
        phi_nonzero_positions = np.abs(self.fd_model) != 0
        phi_nonzero = self.parameter.phi[phi_nonzero_positions]
        fd_model_nonzero = self.fd_model[phi_nonzero_positions]

        fd_model_abs = np.abs(fd_model_nonzero)
        k_parameter = np.sum(fd_model_abs)
        if k_parameter == 0 or phi_nonzero.size == 0:
            return 0.0
        first_moment = np.sum(phi_nonzero * fd_model_abs) / k_parameter
        second_moment = (np.sum(fd_model_abs * (phi_nonzero - first_moment)**2) / k_parameter)
        return second_moment
