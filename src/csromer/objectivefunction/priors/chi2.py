from dataclasses import dataclass

import numpy as np

from ...transformers.dfts import FT
from ...utils.utilities import complex_to_real, real_to_complex
from ..fi import Fi


@dataclass(init=True, repr=True)
class Chi2(Fi):
    dft_obj: FT = None
    F_dirty: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()

        if self.dft_obj is not None and self.F_dirty is None:
            self.F_dirty = self.dft_obj.backward(self.dft_obj.dataset.data)

    def evaluate(self, x):
        if self.wavelet is not None:
            x_ = self.wavelet.reconstruct(x.copy())
        else:
            x_ = x.copy()
        x_complex = real_to_complex(x_) * self.norm_factor
        model_data = self.dft_obj.forward_normalized(x_complex)
        self.dft_obj.dataset.model_data = model_data
        res = self.dft_obj.dataset.residual
        chi2_vector = self.dft_obj.dataset.w * (res.real**2 + res.imag**2)
        val = 0.5 * np.sum(chi2_vector)
        return val

    def calculate_gradient(self, x):
        if self.wavelet is not None:
            x_ = self.wavelet.reconstruct(x.copy())
        else:
            x_ = x.copy()
        x_complex = real_to_complex(x_) * self.norm_factor
        val = x_complex - self.F_dirty
        return complex_to_real(val)

    def calculate_gradient_fista(self, x):
        if self.wavelet is not None:
            x_ = self.wavelet.reconstruct(x.copy())
        else:
            x_ = x.copy()
        x_complex = real_to_complex(x_) * self.norm_factor
        model_data = self.dft_obj.forward_normalized(x_complex)
        self.dft_obj.dataset.model_data = model_data
        res = -self.dft_obj.dataset.residual
        val = self.dft_obj.backward(res)
        ret_val = complex_to_real(val)
        if self.wavelet is not None:
            ret_val = self.wavelet.decompose(ret_val)
        return ret_val

    def calculate_prox(self, x, nu=0):
        a_transpose_b = complex_to_real(self.F_dirty)
        lambda_plus_one = self.reg + 1.0
        one_over_lambda1 = 1.0 / lambda_plus_one
        return one_over_lambda1 * (self.reg * a_transpose_b + nu)
