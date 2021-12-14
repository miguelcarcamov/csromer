import numpy as np
import sys
import astropy.units as un
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import matplotlib.pyplot as plt

from csromer.io import Reader, filter_cubes
from csromer.utils import calculate_noise
from csromer.base import Dataset
from csromer.reconstruction import Parameter
from csromer.transformers import DFT1D, NUFFT1D
from csromer.objectivefunction import OFunction
from csromer.objectivefunction import TSV, L1, Chi2
from csromer.dictionaries import DiscreteWavelet, UndecimatedWavelet
from csromer.optimization import FISTA
from csromer.transformers import MeanFlagger
from csromer.faraday_sky import FaradaySky

cubes = sys.argv[1]
spc_idx = float(sys.argv[2])
use_wavelet = eval(sys.argv[3])

reader = Reader()
IQUV_header, IQUV = reader.readCube(cubes)
I, Q, U, nu = filter_cubes(IQUV[0], IQUV[1], IQUV[2], IQUV_header)
Q = np.flipud(Q)
U = np.flipud(U)

coord = SkyCoord(ra=149.60414 * un.deg, dec=2.4470554 * un.deg, frame=IQUV_header["RADESYS"].lower())
wcs = WCS(IQUV_header, naxis=2)

x, y = coord.to_pixel(wcs, origin=0)
x = int(x)
y = int(y)

sigma_Q_cube = calculate_noise(image=Q, xn=300, yn=300, nsigma=3.0, use_sigma_clipped_stats=True)
sigma_U_cube = calculate_noise(image=U, xn=300, yn=300, nsigma=3.0, use_sigma_clipped_stats=True)

sigma_qu = 0.5 * (sigma_Q_cube + sigma_U_cube)

data = Q[:, y, x] + 1j * U[:, y, x]

# Constructing dataset
measurements = Dataset(nu=nu, data=data, sigma=sigma_qu, spectral_idx=spc_idx)

f_sky = FaradaySky(filename="/raid/scratch/carcamo/repos/csromer/faradaysky/faraday2020v2.hdf5")
gal_mean, gal_std = f_sky.galactic_rm(coord.ra, coord.dec, frame=IQUV_header["RADESYS"].lower())

measurements.subtract_galacticrm(gal_mean.value)

## Flagging dataset ##
normal_flagger = MeanFlagger(data=measurements, nsigma=5.0, delete_channels=True)
idxs, outliers_idxs = normal_flagger.run()

## Setting parameter space ##
parameter = Parameter()
parameter.calculate_cellsize(dataset=measurements, oversampling=8)

nufft = NUFFT1D(dataset=measurements, parameter=parameter, solve=True)
dft = DFT1D(dataset=measurements, parameter=parameter)

## We will start with the dirty as a guess ##

F_dirty = dft.backward(measurements.data)

noise = 0.5 * 0.5 * (np.std(F_dirty.real[np.abs(parameter.phi) > parameter.max_faraday_depth / 1.5]) + np.std(
    F_dirty.imag[np.abs(parameter.phi) > parameter.max_faraday_depth / 1.5]))

parameter.data = F_dirty
parameter.complex_data_to_real()

if use_wavelet:
    wav = DiscreteWavelet(wavelet_name="rbio5.5", mode="periodization")
    # wav = UndecimatedWavelet(wavelet_name="haar")
else:
    wav = None

lambda_l1 = np.sqrt(2 * len(measurements.data) + np.sqrt(4 * len(measurements.data))) * noise
lambda_tsv = 0.0
chi2 = Chi2(dft_obj=nufft, wavelet=wav)
l1 = L1(reg=lambda_l1)
tsv = TSV(reg=lambda_tsv)
# F_func = [chi2(P, dft, W), L1(lambda_l1)]
F_func = [chi2, l1, tsv]
f_func = [chi2]
g_func = [l1, tsv]

F_obj = OFunction(F_func)
f_obj = OFunction(f_func)
g_obj = OFunction(g_func)

opt = FISTA(guess_param=parameter, F_obj=F_obj, fx=chi2, gx=g_obj, noise=noise, verbose=True)
obj, X = opt.run()

if wav is not None:
    coeffs = X.data.copy()
    X.data = wav.reconstruct(X.data)
else:
    coeffs = None

X.real_data_to_complex()

F_residual = nufft.backward(measurements.residual)
F_restored = X.convolve(normalized=True) + F_residual

phi_idx = np.argmax(np.abs(F_restored))
print("Rotation Measure at peak {0:.2f}".format(parameter.phi[phi_idx]))
print("Rotated intensity {0:.2f}".format(np.max(np.abs(F_restored))))

if wav is not None:
    parameter_ncols = 5
else:
    parameter_ncols = 4

fig1, ax1 = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
fig2, ax2 = plt.subplots(nrows=1, ncols=parameter_ncols, figsize=(18, 5))

# Data
ax1[0].plot(measurements.lambda2, measurements.data.real, 'k.', label=r"Stokes $Q$")
ax1[0].plot(measurements.lambda2, measurements.data.imag, 'c.', label=r"Stokes $U$")
ax1[0].plot(measurements.lambda2, np.abs(measurements.data), 'g.', label=r"$|P|$")
ax1[0].set_xlabel(r'$\lambda^2$[m$^{2}$]')
ax1[0].set_ylabel(r'Jy/beam')
ax1[0].title.set_text("Data")
ax1[0].legend()

ax2[0].plot(parameter.phi, F_dirty.real, 'c--', label=r"Stokes $Q$")
ax2[0].plot(parameter.phi, F_dirty.imag, 'c:', label=r"Stokes $U$")
ax2[0].plot(parameter.phi, np.abs(F_dirty), 'k-', label=r"|P|")
ax2[0].set_xlabel(r'$\phi$[rad m$^{-2}$]')
ax2[0].set_ylabel(r'Jy/beam m$^2$ rad$^{-1}$ rmtf$^{-1}$')
ax2[0].set_xlim([-1000, 1000])
ax2[0].legend()

# Model
ax1[1].plot(measurements.lambda2, measurements.model_data.real, 'k.', label=r"Stokes $Q$")
ax1[1].plot(measurements.lambda2, measurements.model_data.imag, 'c.', label=r"Stokes $U$")
ax1[1].plot(measurements.lambda2, np.abs(measurements.model_data), 'g.', label=r"$|P|$")
ax1[1].set_xlabel(r'$\lambda^2$[m$^{2}$]')
ax1[1].set_ylabel(r'Jy/beam')
ax1[1].title.set_text("Model")
ax1[1].legend()

ax2[1].plot(parameter.phi, X.data.real, 'c--', label=r"Stokes $Q$")
ax2[1].plot(parameter.phi, X.data.imag, 'c:', label=r"Stokes $U$")
ax2[1].plot(parameter.phi, np.abs(X.data), 'k-', label=r"$|P|$")
ax2[1].set_xlabel(r'$\phi$[rad m$^{-2}$]')
ax2[1].set_ylabel(r'Jy/beam m$^2$ rad$^{-1}$ pix$^{-1}$')
ax2[1].set_xlim([-1000, 1000])
ax2[1].legend()

# Residual
ax1[2].plot(measurements.lambda2, measurements.residual.real, 'k.', label=r"Stokes $Q$")
ax1[2].plot(measurements.lambda2, measurements.residual.imag, 'c.', label=r"Stokes $U$")
ax1[2].plot(measurements.lambda2, np.abs(measurements.residual), 'g.', label=r"$|P|$")
ax1[2].set_xlabel(r'$\lambda^2$[m$^{2}$]')
ax1[2].set_ylabel(r'Jy/beam')
ax1[2].title.set_text("Residual")
ax1[2].legend()

ax2[2].plot(parameter.phi, F_residual.real, 'c--', label=r"Stokes $Q$")
ax2[2].plot(parameter.phi, F_residual.imag, 'c:', label=r"Stokes $U$")
ax2[2].plot(parameter.phi, np.abs(F_residual), 'k-', label=r"$|P|$")
ax2[2].set_xlabel(r'$\phi$[rad m$^{-2}$]')
ax2[2].set_ylabel(r'Jy/beam m$^2$ rad$^{-1}$ rmtf$^{-1}$')
ax2[2].set_xlim([-1000, 1000])
ax2[2].legend()


# Restored & Coeffs if any

ax2[3].plot(parameter.phi, F_restored.real, 'c--', label=r"Stokes $Q$")
ax2[3].plot(parameter.phi, F_restored.imag, 'c:', label=r"Stokes $U$")
ax2[3].plot(parameter.phi, np.abs(F_restored), 'k-', label=r"$|P|$")
ax2[3].set_xlim([-1000, 1000])
ax2[3].set_xlabel(r'$\phi$[rad m$^{-2}$]')
ax2[3].set_ylabel(r'Jy/beam m$^2$ rad$^{-1}$ rmtf$^{-1}$')
ax2[3].title.set_text("Restored")
ax2[3].legend()

if coeffs is not None:
    ax2[4].plot(coeffs)

fig1.tight_layout()
fig1.savefig("single_los_l2_sq.pdf", dpi=600)

fig2.tight_layout()
fig2.savefig("single_los_phi.pdf", dpi=600)
