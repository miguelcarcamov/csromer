from .wavelet import Wavelet
import pywt


class DiscreteWavelet(Wavelet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.wavelet_name is not None and self.wavelet_name in pywt.wavelist(kind="discrete"):
            self.wavelet = pywt.Wavelet(self.wavelet_name)
        else:
            raise ValueError("The wavelet is not discrete")

    def calculate_ncoeffs(self, x):
        n = len(x)
        return pywt.dwt_coeff_len(n, self.wavelet, mode=mode)

    def calculate_max_level(self, x):
        n = len(x)
        return pywt.dwt_max_level(n, self.wavelet.dec_len)

    def decompose(self, x):
        if self.level is not None:
            if self.level > self.calculate_max_level(x):
                raise ValueError("You are trying to decompose into more levels than the maximum level expected")

        # Return coefficients
        coeffs = pywt.wavedec(data=x, wavelet=self.wavelet, mode=self.mode, level=self.level)
        coeffs_arr, self.coeff_slices = pywt.coeffs_to_array(coeffs)
        return coeffs_arr

    def decompose_complex(self, x):
        if self.level is not None and self.level > self.calculate_max_level(x.real):
            raise ValueError("You are trying to decompose into more levels than the maximum level expected")

        # Return coefficients
        coeffs_re = pywt.wavedec(data=x.real, wavelet=self.wavelet, mode=self.mode, level=self.level)
        coeffs_im = pywt.wavedec(data=x.imag, wavelet=self.wavelet, mode=self.mode, level=self.level)

        coeffs_arr_re, coeffs_slices_re = pywt.coeffs_to_array(coeffs_re)
        coeffs_arr_im, coeffs_slices_im = pywt.coeffs_to_array(coeffs_im)

        self.coeff_slices = coeffs_slices_re, coeffs_slices_im
        return coeffs_arr_re + 1j * coeffs_arr_im

    def reconstruct(self, input_coeffs):
        coeffs = pywt.array_to_coeffs(input_coeffs, self.coeff_slices, output_format='wavedec')

        # Return signal
        return pywt.waverec(coeffs, self.wavelet, self.mode)

    def reconstruct_complex(self, input_coeffs):
        coeffs_re = pywt.array_to_coeffs(input_coeffs.real, self.coeff_slices[0], output_format='wavedec')
        coeffs_im = pywt.array_to_coeffs(input_coeffs.imag, self.coeff_slices[1], output_format='wavedec')

        # Return signal
        signal_re = pywt.waverec(coeffs_re, self.wavelet, self.mode)
        signal_im = pywt.waverec(coeffs_im, self.wavelet, self.mode)
        return signal_re + 1j * signal_im
