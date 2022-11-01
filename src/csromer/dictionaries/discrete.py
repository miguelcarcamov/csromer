from dataclasses import dataclass

import numpy as np
import pywt

from .wavelet import Wavelet


@dataclass(init=True, repr=True)
class DiscreteWavelet(Wavelet):

    def __post_init__(self):
        super().__post_init__()

        if self.wavelet_name is not None and self.wavelet_name in pywt.wavelist(kind="discrete"):
            self.wavelet = pywt.Wavelet(self.wavelet_name)
        else:
            raise NotImplementedError("The wavelet has not been implemented by pywavelets")

    def calculate_ncoeffs(self, x):
        n = len(x)
        return pywt.dwt_coeff_len(n, self.wavelet, mode=self.mode)

    def calculate_max_level(self, x):
        n = len(x)
        return pywt.dwt_max_level(n, self.wavelet.dec_len)

    def decompose(self, x):
        if self.wavelet_level is not None:
            if self.wavelet_level > self.calculate_max_level(x):
                raise ValueError(
                    "You are trying to decompose into more levels than the maximum level expected"
                )

        self.n = len(x)
        # Return coefficients
        coeffs = pywt.wavedec(
            data=x, wavelet=self.wavelet, mode=self.mode, level=self.wavelet_level
        )
        coeffs_arr, self.coeff_slices, self.coeff_shapes = pywt.ravel_coeffs(coeffs=coeffs)

        if self.append_signal:
            coeffs_arr = np.concatenate([x, coeffs_arr])
        return coeffs_arr

    def decompose_complex(self, x):
        if self.wavelet_level is not None and self.wavelet_level > self.calculate_max_level(x.real):
            raise ValueError(
                "You are trying to decompose into more levels than the maximum level expected"
            )

        # Return coefficients
        coeffs_re = pywt.wavedec(
            data=x.real, wavelet=self.wavelet, mode=self.mode, level=self.wavelet_level
        )
        coeffs_im = pywt.wavedec(
            data=x.imag, wavelet=self.wavelet, mode=self.mode, level=self.wavelet_level
        )

        coeffs_arr_re, coeffs_slices_re = pywt.coeffs_to_array(coeffs=coeffs_re)
        coeffs_arr_im, coeffs_slices_im = pywt.coeffs_to_array(coeffs=coeffs_im)

        if self.append_signal:
            coeffs_arr_re = np.concatenate([x.real, coeffs_arr_re])
            coeffs_arr_im = np.concatenate([x.imag, coeffs_arr_im])

        self.coeff_slices = coeffs_slices_re, coeffs_slices_im
        return coeffs_arr_re + 1j * coeffs_arr_im

    def reconstruct(self, input_coeffs):
        if self.append_signal:
            coeffs = input_coeffs.copy()
            signal = coeffs[0:self.n]
            coeffs_ = pywt.unravel_coeffs(
                arr=coeffs[self.n:len(coeffs)],
                coeff_slices=self.coeff_slices,
                coeff_shapes=self.coeff_shapes,
                output_format="wavedec",
            )
            signal_from_coeffs = pywt.waverec(coeffs=coeffs_, wavelet=self.wavelet, mode=self.mode)
            signal += signal_from_coeffs
        else:
            coeffs = pywt.unravel_coeffs(
                arr=input_coeffs,
                coeff_slices=self.coeff_slices,
                coeff_shapes=self.coeff_shapes,
                output_format="wavedec",
            )
            signal = pywt.waverec(coeffs=coeffs, wavelet=self.wavelet, mode=self.mode)
        # Return signal
        return signal

    def reconstruct_complex(self, input_coeffs):
        if self.append_signal:
            coeffs = input_coeffs.copy()
            signal = coeffs[0:self.n]
            coeffs_ = coeffs[self.n:len(input_coeffs)]
            coeffs_re = pywt.array_to_coeffs(
                arr=coeffs_.real,
                coeff_slices=self.coeff_slices[0],
                output_format="wavedec",
            )
            coeffs_im = pywt.array_to_coeffs(
                arr=coeffs_.imag,
                coeff_slices=self.coeff_slices[1],
                output_format="wavedec",
            )
            signal_re_from_coeffs = pywt.waverec(
                coeffs=coeffs_re, wavelet=self.wavelet, mode=self.mode
            )
            signal_im_from_coeffs = pywt.waverec(
                coeffs=coeffs_im, wavelet=self.wavelet, mode=self.mode
            )
            signal_from_coeffs = signal_re_from_coeffs + 1j * signal_im_from_coeffs
            signal += signal_from_coeffs
        else:
            coeffs_re = pywt.array_to_coeffs(
                arr=input_coeffs.real,
                coeff_slices=self.coeff_slices[0],
                output_format="wavedec",
            )
            coeffs_im = pywt.array_to_coeffs(
                arr=input_coeffs.imag,
                coeff_slices=self.coeff_slices[1],
                output_format="wavedec",
            )
            signal_re = pywt.waverec(coeffs=coeffs_re, wavelet=self.wavelet, mode=self.mode)
            signal_im = pywt.waverec(coeffs=coeffs_im, wavelet=self.wavelet, mode=self.mode)
            signal = signal_re + 1j * signal_im
        return signal
