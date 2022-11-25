from dataclasses import dataclass

import numpy as np
import pywt

from ..utils import next_power_2
from .wavelet import Wavelet


@dataclass(init=True, repr=True)
class UndecimatedWavelet(Wavelet):
    trim_approx: bool = None
    norm: bool = None

    def __post_init__(self):
        super().__post_init__()

        if self.wavelet_name is not None and self.wavelet_name in pywt.wavelist(kind="all"):
            self.wavelet = pywt.Wavelet(self.wavelet_name)
        elif self.wavelet_name == "IUWT":
            h = list(np.divide([1.0, 4.0, 6.0, 4.0, 1.0], 16))
            g = list(np.divide([-1.0, -4.0, 10.0, -4.0, -1.0], 16))
            delta = list(np.divide([0.0, 0.0, 16.0, 0.0, 0.0], 16))
            filter_bank = [g, h, delta, delta]
            self.wavelet = pywt.Wavelet("IUWT", filter_bank=filter_bank)
        else:
            raise NotImplementedError("The wavelet has not been implemented by pywavelets")

        if self.trim_approx is None:
            self.trim_approx = True

        if self.norm is None:
            self.norm = False

        if self.wavelet_level is not None:
            self.array_size = 2**self.wavelet_level

        self.pad_width = None

    @staticmethod
    def calculate_max_level(x):
        n = len(x)
        return pywt.swt_max_level(n)

    def decompose(self, x):
        if self.wavelet_level is not None:
            if self.wavelet_level > self.calculate_max_level(x):
                raise ValueError(
                    "You are trying to decompose into more levels than the maximum level expected"
                )

        if self.wavelet_level is None:
            max_level = self.calculate_max_level(x)
            array_size = 2**max_level
        else:
            array_size = 2**self.wavelet_level

        signal_size = len(x)
        self.n = signal_size
        x_copy = x.copy()

        if signal_size and (signal_size % array_size) != 0:
            print(
                "Your signal length is not multiple of 2**" + str(self.wavelet_level) +
                ". Padding array..."
            )
            padded_size = next_power_2(signal_size)
            self.pad_width = padded_size - signal_size

            if self.mode is None:
                x_copy = np.pad(x_copy, (0, self.pad_width))
            else:
                x_copy = pywt.pad(x=x_copy, pad_widths=(0, self.pad_width), mode=self.mode)

        coeffs = pywt.swt(
            data=x_copy,
            wavelet=self.wavelet,
            level=self.wavelet_level,
            trim_approx=self.trim_approx,
            norm=self.norm,
        )
        coeffs_arr, self.coeff_slices, self.coeff_shapes = pywt.ravel_coeffs(coeffs)

        if self.append_signal:
            coeffs_arr = np.concatenate([x, coeffs_arr])
        return coeffs_arr

    def decompose_complex(self, x):
        if self.wavelet_level is not None and self.wavelet_level > self.calculate_max_level(x.real):
            raise ValueError(
                "You are trying to decompose into more levels than the maximum level expected"
            )

        if self.wavelet_level is None:
            array_size = 2**self.calculate_max_level(x.real)
        else:
            array_size = 2**self.wavelet_level

        signal_size = len(x)
        self.n = signal_size
        x_copy = x.copy()
        if signal_size and (signal_size % array_size) != 0:
            print(
                "Your signal length is not multiple of 2**" + str(self.wavelet_level) +
                ". Padding array..."
            )
            # padded_size = int(array_size * round(float(signal_size) / array_size))
            padded_size = next_power_2(signal_size)
            self.pad_width = padded_size - signal_size
            if self.mode is None:
                x_copy = np.pad(x_copy, (0, self.pad_width))
            else:
                x_copy = pywt.pad(x=x_copy, pad_widths=(0, self.pad_width), mode=self.mode)

        # Return coefficients
        coeffs_re = pywt.swt(
            data=x_copy.real,
            wavelet=self.wavelet,
            level=self.wavelet_level,
            trim_approx=self.trim_approx,
            norm=self.norm,
        )
        coeffs_im = pywt.swt(
            data=x_copy.imag,
            wavelet=self.wavelet,
            level=self.wavelet_level,
            trim_approx=self.trim_approx,
            norm=self.norm,
        )

        coeffs_arr_re, coeffs_slices_re = pywt.coeffs_to_array(coeffs_re)
        coeffs_arr_im, coeffs_slices_im = pywt.coeffs_to_array(coeffs_im)

        self.coeff_slices = coeffs_slices_re, coeffs_slices_im

        if self.append_signal:
            coeffs_arr_re = np.concatenate([x.real, coeffs_arr_re])
            coeffs_arr_im = np.concatenate([x.imag, coeffs_arr_im])
        return coeffs_arr_re + 1j * coeffs_arr_im

    def reconstruct(self, input_coeffs):

        if self.append_signal:
            signal = input_coeffs[0:self.n].copy()
            coeffs = pywt.unravel_coeffs(
                arr=input_coeffs[self.n:len(input_coeffs)],
                coeff_slices=self.coeff_slices,
                coeff_shapes=self.coeff_shapes,
                output_format="wavedec",
            )
        else:
            coeffs = pywt.unravel_coeffs(
                arr=input_coeffs,
                coeff_slices=self.coeff_slices,
                coeff_shapes=self.coeff_shapes,
                output_format="wavedec",
            )

        signal_from_coeffs = pywt.iswt(coeffs, self.wavelet, self.norm)

        if self.pad_width is not None:
            # Undo padding
            signal_from_coeffs = signal_from_coeffs[0:len(signal_from_coeffs) - self.pad_width]
            self.pad_width = None

        if self.append_signal:
            signal += signal_from_coeffs
        else:
            signal = signal_from_coeffs

        return signal

    def reconstruct_complex(self, input_coeffs):

        if self.append_signal:
            signal = input_coeffs[0:self.n].copy()
            coeffs = input_coeffs[self.n:len(input_coeffs)]
            coeffs_re = pywt.array_to_coeffs(
                coeffs.real, self.coeff_slices[0], output_format="wavedec"
            )
            coeffs_im = pywt.array_to_coeffs(
                coeffs.imag, self.coeff_slices[1], output_format="wavedec"
            )

        else:
            coeffs_re = pywt.array_to_coeffs(
                input_coeffs.real, self.coeff_slices[0], output_format="wavedec"
            )
            coeffs_im = pywt.array_to_coeffs(
                input_coeffs.imag, self.coeff_slices[1], output_format="wavedec"
            )

        signal_re = pywt.iswt(coeffs_re, self.wavelet, self.norm)
        signal_im = pywt.iswt(coeffs_im, self.wavelet, self.norm)

        if self.pad_width is not None:
            # Undo padding
            signal_re = signal_re[0:len(signal_re) - self.pad_width]
            signal_im = signal_im[0:len(signal_im) - self.pad_width]
            self.pad_width = None

        signal_from_coeffs = signal_re + 1.0j * signal_im

        if self.append_signal:
            signal += signal_from_coeffs
        else:
            signal = signal_from_coeffs

        return signal
