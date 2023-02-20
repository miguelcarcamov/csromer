from dataclasses import dataclass, field

import numpy as np

from ...dictionaries import Wavelet
from ...objectivefunction import L1, TSV, TV, Chi2, OFunction
from ...optimization import FISTA
from ...reconstruction import Parameter
from ...transformers.dfts import NDFT1D, NUFFT1D
from ...transformers.flaggers.flagger import Flagger
from .faraday_reconstructor import FaradayReconstructorWrapper


@dataclass(init=True, repr=True)
class CSROMERReconstructorWrapper(FaradayReconstructorWrapper):
    parameter: Parameter = field(init=False)
    flagger: Flagger = None
    dft: NDFT1D = field(init=False)
    nufft: NUFFT1D = field(init=False)
    wavelet: Wavelet = None
    coefficients: np.ndarray = field(init=False)
    fd_restored: np.ndarray = field(init=False)
    rm_restored: float = field(init=False)
    restored_peak_quadratic_interpolation: float = field(init=False)
    rm_restored_quadratic_interpolation: float = field(init=False)
    fd_model: np.ndarray = field(init=False)
    rm_model: float = field(init=False)
    fd_residual: np.ndarray = field(init=False)
    fd_dirty: np.ndarray = field(init=False)
    rm_dirty: float = field(init=False)
    dirty_peak_quadratic_interpolation: float = field(init=False)
    rm_dirty_quadratic_interpolation: float = field(init=False)
    second_moment: float = field(init=False)
    cellsize: float = None
    oversampling: float = None
    lambda_l_norm: float = None

    def __post_init__(self):
        if self.oversampling is None:
            self.oversampling = 7.0
        self.parameter = Parameter()
        self.config_fd_space(self.cellsize, self.oversampling)
        self.config_fourier_transforms()

    def flag_dataset(self, flagger: Flagger = None):

        if flagger is None:
            idxs, outliers_idxs = self.flagger.run()
        else:
            idxs, outliers_idxs = flagger.run()

        return idxs, outliers_idxs

    def config_fd_space(self, cellsize: float = None, oversampling: float = None):
        if cellsize is not None and oversampling is not None:
            self.parameter.calculate_cellsize(dataset=self.dataset, cellsize=cellsize)
        elif cellsize is None and oversampling is not None:
            self.parameter.calculate_cellsize(dataset=self.dataset, oversampling=oversampling)
        elif cellsize is not None and oversampling is None:
            self.parameter.calculate_cellsize(dataset=self.dataset, cellsize=cellsize)
        else:
            raise ValueError("Either cellsize or oversampling cannot be Nonetype values")

    def config_fourier_transforms(self):
        self.dft = NDFT1D(dataset=self.dataset, parameter=self.parameter)
        self.nufft = NUFFT1D(dataset=self.dataset, parameter=self.parameter, solve=True)

    def get_dirty_faraday_depth(self):
        return self.dft.backward(self.dataset.data)

    def get_rmtf(self):
        return self.dft.RMTF()

    def get_rm(self, fd_data):
        return self.parameter.phi[np.argmax(np.abs(fd_data))]

    def reconstruct(self):
        if self.flagger:
            self.flag_dataset()

        fd_dirty = self.get_dirty_faraday_depth()
        self.fd_dirty = fd_dirty
        self.parameter.data = fd_dirty
        self.rm_dirty = self.get_rm(fd_dirty)
        (
            self.rm_dirty_quadratic_interpolation,
            self.dirty_peak_quadratic_interpolation,
        ) = self.estimate_peak_quadratic_interpolation(fd_dirty, self.parameter.cellsize)
        self.parameter.complex_data_to_real()

        if self.wavelet is not None:
            lambda_l_norm = (
                np.sqrt(self.dataset.m + 2 * np.sqrt(self.dataset.m)) * 2.0 * np.sqrt(2) *
                np.mean(self.dataset.sigma)
            )
        else:
            lambda_l_norm = (
                np.sqrt(self.dataset.m + 2 * np.sqrt(self.dataset.m)) * np.sqrt(2) *
                np.mean(self.dataset.sigma)
            )

        chi2 = Chi2(dft_obj=self.nufft, wavelet=self.wavelet)
        l1 = L1(reg=lambda_l_norm)

        F_func = [chi2, l1]
        f_func = [chi2]
        g_func = [l1]

        F_obj = OFunction(F_func)
        f_obj = OFunction(f_func)
        g_obj = OFunction(g_func)

        if self.wavelet is not None:
            opt_noise = 2.0 * self.dataset.theo_noise
        else:
            opt_noise = self.dataset.theo_noise

        opt = FISTA(
            guess_param=self.parameter,
            F_obj=F_obj,
            fx=chi2,
            gx=g_obj,
            noise=opt_noise,
            verbose=True,
        )

        obj, X = opt.run()

        if self.wavelet is not None:
            self.coefficients = X.data
            X.data = self.wavelet.reconstruct(X.data)

        X.real_data_to_complex()

        self.fd_model = X.data
        self.rm_model = self.get_rm(self.fd_model)
        self.second_moment = self.calculate_second_moment()

        self.fd_residual = self.dft.backward(self.dataset.data - self.dataset.model_data)

        self.fd_restored = X.convolve() + self.fd_residual
        self.rm_restored = self.get_rm(self.fd_restored)
        (
            self.rm_restored_quadratic_interpolation,
            self.restored_peak_quadratic_interpolation,
        ) = self.estimate_peak_quadratic_interpolation(self.fd_restored, self.parameter.cellsize)

    def calculate_second_moment(self):
        phi_nonzero_positions = np.abs(self.fd_model) != 0
        phi_nonzero = self.parameter.phi[phi_nonzero_positions]
        fd_model_nonzero = self.fd_model[phi_nonzero_positions]

        fd_model_abs = np.abs(fd_model_nonzero)
        k_parameter = np.sum(fd_model_abs)
        first_moment = np.sum(phi_nonzero * fd_model_abs) / k_parameter

        second_moment = (np.sum(fd_model_abs * (phi_nonzero - first_moment)**2) / k_parameter)

        return second_moment
