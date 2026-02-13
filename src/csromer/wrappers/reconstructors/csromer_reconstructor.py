"""
CS-ROMER reconstructor: FISTA-based sparse reconstruction with L1 regularization.

Minimizes ChiSquared + L1 using FISTA optimizer. Supports wavelet transforms for
sparse representation. Computes dirty map, model, residual, and restored maps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.stats import sigma_clipped_stats

from ...dictionaries import Wavelet
from ...objectivefunction import L1, TSV, TV, ChiSquared, OFunction
from ...optimization import FISTA
from ...reconstruction import Parameter
from ...transformers.dfts import NDFT1D, NUFFT1D
from ...transformers.flaggers.flagger import Flagger
from .faraday_reconstructor import FaradayReconstructorWrapper


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
        dft: Direct Fourier transform operator (for dirty map)
        nufft: NUFFT operator (for optimization)
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
        lambda_l_norm: L1 regularization factor (auto-computed if None)
        calculate_l2_zero: Whether to compute l2_ref (default: False)
    """
    parameter: Parameter = field(init=False)
    flagger: Flagger = None
    dft: NDFT1D = field(init=False)
    nufft: NUFFT1D = field(init=False)
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
    lambda_l_norm: float = None
    calculate_l2_zero: bool = None

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
        index_0 = np.argmax(np.abs(fd_signal))

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
        Configure Fourier transform operators.
        
        Public method. Called during initialization. Sets up DFT (for dirty map)
        and NUFFT (for optimization).
        """
        self.dft = NDFT1D(dataset=self.dataset, parameter=self.parameter)
        self.nufft = NUFFT1D(dataset=self.dataset, parameter=self.parameter, solve=True)

    def get_dirty_faraday_depth(self) -> np.ndarray:
        """
        Compute dirty Faraday depth spectrum.
        
        Public method. Returns A^H(weighted data) / K.
        
        Returns:
            Dirty Faraday depth spectrum (n_phi,)
        """
        return self.dft.dirty_spectrum(self.dataset.data)

    def get_rmtf(self) -> np.ndarray:
        """
        Get Rotation Measure Transfer Function.
        
        Public method.
        
        Returns:
            RMTF array (n_phi,)
        """
        return self.dft.RMTF()

    def get_rm(self, fd_data: np.ndarray) -> float:
        """
        Get rotation measure at peak of Faraday depth spectrum.
        
        Public method. Finds peak location and returns phi value at that location.
        
        Args:
            fd_data: Faraday depth spectrum
            
        Returns:
            Rotation measure (rad/m²)
        """
        rm_at_peak = self.parameter.phi[np.argmax(np.abs(fd_data))]
        return rm_at_peak

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
        self.parameter.data = fd_dirty
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
            self.parameter.data = self.wavelet.decompose_complex(fd_dirty)
        # Faraday depth kept as complex; no real stacking

        if self.lambda_l_norm is None:
            if self.wavelet is not None:
                self.lambda_l_norm = (
                    np.sqrt(self.dataset.m + 2 * np.sqrt(self.dataset.m)) * 2.0 * np.sqrt(2) *
                    np.mean(self.dataset.sigma)
                )
            else:
                self.lambda_l_norm = (
                    np.sqrt(self.dataset.m + 2 * np.sqrt(self.dataset.m)) * np.sqrt(2) *
                    np.mean(self.dataset.sigma)
                )

        chi_squared = ChiSquared(measurement_operator=self.nufft, wavelet=self.wavelet)
        l1 = L1(reg=self.lambda_l_norm)

        F_func = [chi_squared, l1]
        F_obj = OFunction(F_func, persist_gradient=True)

        if self.wavelet is not None:
            opt_noise = 2.0 * self.dataset.theo_noise
        else:
            opt_noise = self.dataset.theo_noise

        opt = FISTA(
            guess_param=self.parameter,
            F_obj=F_obj,
            noise=opt_noise,
            verbose=True,
        )

        obj, X = opt.run()

        self.coefficients = X.data
        if self.wavelet is not None:
            X.data = self.wavelet.reconstruct_complex(X.data)

        self.fd_model = X.data
        self.rm_model = self.get_rm(self.fd_model)
        self.second_moment = self.calculate_second_moment()

        self.fd_residual = self.dft.backward(self.dataset.data - self.dataset.model_data)

        self.fd_restored = X.convolve() + self.fd_residual
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
