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
from csromer.objectivefunction import TSV, TV, L1, Chi2
from csromer.optimization import FISTA
from csromer.transformers import MeanFlagger
from csromer.faraday_sky import FaradaySky

cubes = sys.argv[1]
spc_idx = float(sys.argv[2])

reader = Reader()
IQUV_header, IQUV = reader.readCube(cubes)
I, Q, U, nu = filter_cubes(IQUV[0], IQUV[1], IQUV[2], IQUV_header)
Q = np.flipud(Q)
U = np.flipud(U)

coord = SkyCoord(ra=173.571 * un.deg, dec=49.151 * un.deg, frame=IQUV_header["RADESYS"])
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
gal_mean, gal_std = f_sky.galactic_rm(coord.ra, coord.dec, frame="fk5")

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

lambda_l1 = np.sqrt(2 * len(measurements.data) + np.sqrt(4 * len(measurements.data))) * noise
lambda_tsv = 0.0
chi2 = Chi2(dft_obj=nufft)
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

X.real_data_to_complex()

F_residual = nufft.backward(measurements.residual)
F_restored = X.convolve(normalized=True) + F_residual

phi_idx = np.argmax(np.abs(F_restored))
print("Rotation Measure at peak {0:.2f}".format(parameter.phi[phi_idx]))
print("Rotated intensity {0:.2f}".format(np.max(np.abs(F_restored))))

fig, axs = plt.subplots(2, 4)

# l2-squared data
axs[0, 0].plot(measurements.lambda2, measurements.data.real, 'k.', label=r"Stokes $Q$")
axs[0, 0].plot(measurements.lambda2, measurements.data.imag, 'c.', label=r"Stokes $U$")
axs[0, 0].plot(measurements.lambda2, np.abs(measurements.data), 'g.', label=r"$|P|$")
axs[0, 0].xlabel(r'$\lambda^2$[m$^{2}$]')
axs[0, 0].ylabel(r'Jy/beam')

# l2-squared model
axs[0, 1].plot(measurements.lambda2, measurements.model_data.real, 'k.', label=r"Stokes $Q$")
axs[0, 1].plot(measurements.lambda2, measurements.model_data.imag, 'c.', label=r"Stokes $U$")
axs[0, 1].plot(measurements.lambda2, np.abs(measurements.model_data), 'g.', label=r"$|P|$")
axs[0, 1].xlabel(r'$\lambda^2$[m$^{2}$]')
axs[0, 1].ylabel(r'Jy/beam')

# l2-squared residuals
axs[0, 1].plot(measurements.lambda2, measurements.residual.real, 'k.', label=r"Stokes $Q$")
axs[0, 1].plot(measurements.lambda2, measurements.residual.imag, 'c.', label=r"Stokes $U$")
axs[0, 1].plot(measurements.lambda2, np.abs(measurements.residual), 'g.', label=r"$|P|$")
axs[0, 1].xlabel(r'$\lambda^2$[m$^{2}$]')
axs[0, 1].ylabel(r'Jy/beam')

# Faraday dirty
axs[1, 0].plot(parameter.phi, F_dirty.real, 'c--', label=r"Stokes $Q$")
axs[1, 0].plot(parameter.phi, F_dirty.imag, 'c:', label=r"Stokes $U$")
axs[1, 0].plot(parameter.phi, np.abs(F_dirty), 'k-', label=r"$|P|$")
axs[1, 0].set_xlim([-1000, 1000])
axs[1, 0].set_xlabel(r'$\phi$[rad m$^{-2}$]')
axs[1, 0].set_ylabel(r'Jy/beam m$^2$ rad$^{-1}$ rmtf$^{-1}$')
axs[1, 0].legend(loc='upper right')
# Faraday model
axs[1, 1].plot(parameter.phi, X.data.real, 'c--', label=r"Stokes $Q$")
axs[1, 1].plot(parameter.phi, X.data.imag, 'c:', label=r"Stokes $U$")
axs[1, 1].plot(parameter.phi, np.abs(X.data), 'k-', label=r"$|P|$")
axs[1, 1].set_xlim([-1000, 1000])
axs[1, 1].set_xlabel(r'$\phi$[rad m$^{-2}$]')
axs[1, 1].set_ylabel(r'Jy/beam m$^2$ rad$^{-1}$ pix$^{-1}$')
axs[1, 1].legend(loc='upper right')
# Faraday residuals
axs[1, 2].plot(parameter.phi, F_residual.real, 'c--', label=r"Stokes $Q$")
axs[1, 2].plot(parameter.phi, F_residual.imag, 'c:', label=r"Stokes $U$")
axs[1, 2].plot(parameter.phi, np.abs(F_residual), 'k-', label=r"$|P|$")
axs[1, 2].set_xlim([-1000, 1000])
axs[1, 2].set_xlabel(r'$\phi$[rad m$^{-2}$]')
axs[1, 2].set_ylabel(r'Jy/beam m$^2$ rad$^{-1}$ rmtf$^{-1}$')
axs[1, 2].legend(loc='upper right')
# Faraday restored
axs[1, 3].plot(parameter.phi, F_restored.real, 'c--', label=r"Stokes $Q$")
axs[1, 3].plot(parameter.phi, F_restored.imag, 'c:', label=r"Stokes $U$")
axs[1, 3].plot(parameter.phi, np.abs(F_restored), 'k-', label=r"$|P|$")
axs[1, 3].set_xlim([-1000, 1000])
axs[1, 3].set_xlabel(r'$\phi$[rad m$^{-2}$]')
axs[1, 3].set_ylabel(r'Jy/beam m$^2$ rad$^{-1}$ rmtf$^{-1}$')
axs[1, 3].legend(loc='upper right')

plt.tight_layout()
