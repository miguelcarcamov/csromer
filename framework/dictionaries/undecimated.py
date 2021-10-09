from .wavelet import Wavelet
import pywt
import numpy as np


class UndecimatedWavelet(Wavelet):
    def __init__(self, trim_approx: bool = None, norm: bool = None, **kwargs):
        super().__init__(**kwargs)

        if self.wavelet_name is not None and self.wavelet_name in pywt.wavelist(kind="all"):
            self.wavelet = pywt.Wavelet(self.wavelet_name)
        elif self.wavelet_name == "IUWT":
            h = list(np.divide([1., 4., 6., 4., 1.], 16))
            g = list(np.divide([-1., -4., 10., -4., -1.], 16))
            delta = list(np.divide([0., 0., 16., 0., 0.], 16))
            filter_bank = [g, h, delta, delta]
            self.wavelet = pywt.Wavelet('IUWT', filter_bank=filter_bank)
        else:
            raise ValidationError("The wavelet does not exist")

        if trim_approx is None:
            self.trim_approx = True
        else:
            self.trim_approx = trim_approx

        if norm is None:
            self.norm = False
        else:
            self.norm = norm

        if self.level is not None:
            self.array_size = 2 ** self.level

        self.pad_width = None

    def calculate_max_level(self, x):
        n = len(x)
        return pywt.swt_max_level(n)

    def decompose(self, x):
        if self.level is not None:
            if self.level > self.calculate_max_level(x):
                raise ValueError("You are trying to decompose into more levels than the maximum level expected")

        if self.level is None:
            array_size = 2 ** self.calculate_max_level(x)
        else:
            array_size = 2 ** self.level

        signal_size = len(x)
        if signal_size and (array_size % signal_size) == 0:
            print("Your signal length is not multiple of 2**" + str(self.level) + ". Padding array...")
            padded_size = int(array_size * round(float(signal_size) / array_size))
            self.pad_width = padded_size - signal_size
            x = np.pad(x, (0, self.pad_width))

        coeffs = pywt.swt(data=x, wavelet=self.wavelet, level=self.level, trim_approx=self.trim_approx, norm=self.norm)
        coeffs_arr, self.coeff_slices = pywt.coeffs_to_array(coeffs)
        return coeffs_arr

    def decompose_complex(self, x):
        if self.level is not None and self.level > self.calculate_max_level(x.real):
            raise ValueError("You are trying to decompose into more levels than the maximum level expected")

        if self.level is None:
            array_size = 2 ** self.calculate_max_level(x.real)
        else:
            array_size = 2 ** self.level

        signal_size = len(x)
        if signal_size and (array_size % signal_size) == 0:
            print("Your signal length is not multiple of 2**" + str(self.level) + ". Padding array...")
            padded_size = int(array_size * round(float(signal_size) / array_size))
            self.pad_width = padded_size - signal_size
            x = np.pad(x, (0, self.pad_width))

        # Return coefficients
        coeffs_re = pywt.swt(data=x.real, wavelet=self.wavelet, level=self.level, trim_approx=self.trim_approx,
                             norm=self.norm)
        coeffs_im = pywt.swt(data=x.imag, wavelet=self.wavelet, level=self.level, trim_approx=self.trim_approx,
                             norm=self.norm)

        coeffs_arr_re, coeffs_slices_re = pywt.coeffs_to_array(coeffs_re)
        coeffs_arr_im, coeffs_slices_im = pywt.coeffs_to_array(coeffs_im)

        self.coeff_slices = coeffs_slices_re, coeffs_slices_im
        return coeffs_arr_re + 1j * coeffs_arr_im

    def reconstruct(self, input_coeffs):
        coeffs = pywt.array_to_coeffs(input_coeffs, self.coeff_slices, output_format='wavedec')
        signal = pywt.iswt(coeffs, self.wavelet, self.norm)

        if self.pad_width is not None:
            # Undo padding
            signal = signal[0: len(signal) - self.pad_width]
            self.pad_width = None

        return signal

    def reconstruct_complex(self, input_coeffs):
        coeffs_re = pywt.array_to_coeffs(input_coeffs.real, self.coeff_slices[0], output_format='wavedec')
        coeffs_im = pywt.array_to_coeffs(input_coeffs.imag, self.coeff_slices[1], output_format='wavedec')

        signal_re = pywt.iswt(coeffs_re, self.wavelet, self.norm)
        signal_im = pywt.iswt(coeffs_im, self.wavelet, self.norm)

        if self.pad_width is not None:
            # Undo padding
            signal_re = signal_re[0: len(signal_re) - self.pad_width]
            signal_im = signal_im[0: len(signal_im) - self.pad_width]
            self.pad_width = None

        return signal_re + 1j * signal_im
