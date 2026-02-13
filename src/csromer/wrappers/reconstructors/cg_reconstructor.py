"""
CG-based Faraday depth reconstructor: minimizes ChiSquared (smooth) with non-linear
Conjugate Gradient. No L1 regularization; same Parameter/Dataset/DFT setup as
CSROMERReconstructorWrapper.
"""
from dataclasses import dataclass, field

import numpy as np

from ...objectivefunction import ChiSquared, OFunction
from ...optimization import PolakRibiere
from ...reconstruction import Parameter
from ...transformers.dfts import NDFT1D, NUFFT1D
from ...transformers.flaggers.flagger import Flagger
from .csromer_reconstructor import CSROMERReconstructorWrapper


@dataclass(init=True, repr=True)
class CGReconstructorWrapper(CSROMERReconstructorWrapper):
    """
    Reconstructor that minimizes ChiSquared using non-linear Conjugate Gradient.
    Uses PolakRibiere by default; no L1/wavelet. Same dirty/restored/residual
    outputs as CSROMERReconstructorWrapper.
    """

    cg_maxiter: int = 500
    cg_tol: float = 1e-6
    cg_verbose: bool = True

    def __post_init__(self):
        self.wavelet = None
        self.lambda_l_norm = None
        super().__post_init__()

    def reconstruct(self):
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
        # Faraday depth kept as complex (n_phi,) throughout; no real stacking

        chi_squared = ChiSquared(measurement_operator=self.nufft, wavelet=self.wavelet)
        F_obj = OFunction([chi_squared])

        opt = PolakRibiere(
            guess_param=self.parameter,
            F_obj=F_obj,
            grad_fun=chi_squared.calculate_gradient,
            maxiter=self.cg_maxiter,
            tol=self.cg_tol,
            verbose=self.cg_verbose,
        )
        obj, X = opt.run()

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
        ) = self.estimate_peak_quadratic_interpolation(
            self.fd_restored, self.parameter.cellsize
        )
        self.rm_restored_quadratic_interpolation_error = self.calculate_sigma_phi_peak(
            self.parameter.rmtf_fwhm,
            self.restored_peak_quadratic_interpolation,
            restored_noise,
        )
