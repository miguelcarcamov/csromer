import argparse
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, load, dump
from astropy.io import fits
import shutil

from csromer.io import Reader, Writer
from csromer.utils import calculate_noise, make_mask_faraday
from csromer.base import Dataset
from csromer.reconstruction import Parameter
from csromer.transformers import DFT1D, NUFFT1D
from csromer.objectivefunction import OFunction
from csromer.objectivefunction import TSV, TV, L1, Chi2
from csromer.optimization import FISTA, ADMM, SDMM, GradientBasedMethod
from csromer.dictionaries.discrete import DiscreteWavelet
from csromer.dictionaries.undecimated import UndecimatedWavelet
from csromer.transformers import MeanFlagger
from csromer.faraday_sky import FaradaySky


def getopt():
    # initiate the parser
    parser = argparse.ArgumentParser(
        description='This is a program to blablabla')
    parser.add_argument("-V", "--version",
                        help="show program version", action="store_true")
    parser.add_argument("-v", "--verbose",
                        help="Print output", action="store_true")
    parser.add_argument("-c", "--cubes", nargs=2,
                        help="Input Cubes Stokes polarized images (Q,U) separated by a space", required=True)
    parser.add_argument("-m", "--mfs", nargs=3,
                        help="Input MFS Stokes polarized images (I,Q,U) separated by a space", required=True)
    parser.add_argument("-a", "--spectral_idx",
                        help="Input Spectral Index Image", required=False)
    parser.add_argument("-f", "--freq-file",
                        help="Text file with frequency values", required=True)
    parser.add_argument("-s", "--sigmas", nargs=2,
                        help="Number of sigmas in total intensity (I) and polarized intensity (P) above on which "
                             "calculation is done",
                        required=True,
                        type=float)
    parser.add_argument("-l", "--lambdas", nargs='*',
                        help="Regularization parameters separated by space")
    parser.add_argument("-e", "--eta", nargs='?',
                        help="Eta factor to increase or decrease L1 regularization", default=1.0, const=float)
    parser.add_argument("-o", "--output", nargs="*",
                        help="Path/s and/or name/s of the output file/s in FITS/npy format", required=True)
    parser.add_argument("-I", "--index", nargs='?',
                        help="Selected index of the pixel on where the minimization is done", const=int, default=0)

    # read arguments from the command line
    args = parser.parse_args()

    cubes = vars(args)['cubes']
    mfs_images = vars(args)['mfs']
    spec_idx = vars(args)['spectral_idx']
    freq_f = vars(args)['freq_file']
    reg_terms = vars(args)['lambdas']
    eta_term = vars(args)['eta']
    output = vars(args)['output']
    index = vars(args)['index']
    nsigmas = vars(args)['sigmas']
    verbose = vars(args)['verbose']
    # check for --version or -V

    if args.version:
        print("this is myprogram version 0.1")
        sys.exit(1)

    return cubes, mfs_images, spec_idx, freq_f, reg_terms, eta_term, output, index, nsigmas, verbose


def reconstruct_cube(F=None, data=None, sigma=None, nu=None, spectral_idx=None, noise=None,
                     mask_idxs=None, idx=None, eta=1.0, use_wavelet=True):
    i = mask_idxs[0][idx]
    j = mask_idxs[1][idx]

    dataset = Dataset(nu=nu, sigma=sigma, data=data[:, i, j], spectral_idx=spectral_idx[i, j])
    parameter = Parameter()
    parameter.calculate_cellsize(dataset=dataset, oversampling=8, verbose=False)

    dft = DFT1D(dataset=dataset, parameter=parameter)
    nufft = NUFFT1D(dataset=dataset, parameter=parameter, solve=True)

    F_dirty = dft.backward(dataset.data)

    if noise is None:
        edges_idx = np.where(np.abs(parameter.phi) > parameter.max_faraday_depth / 1.5)
        noise = eta * 0.5 * (np.std(F_dirty[edges_idx].real) + np.std(F_dirty[edges_idx].imag))

    F[0, :, i, j] = F_dirty

    if use_wavelet:
        wav = UndecimatedWavelet(wavelet_name="coif2")
    else:
        wav = None

    lambda_l1 = np.sqrt(2 * len(dataset.data) + np.sqrt(4 * len(dataset.data))) * noise
    lambda_tsv = 0.0
    chi2 = Chi2(dft_obj=nufft, wavelet=wav)
    l1 = L1(reg=lambda_l1)
    tsv = TSV(reg=lambda_tsv)
    F_func = [chi2, l1, tsv]
    f_func = [chi2]
    g_func = [l1, tsv]

    F_obj = OFunction(F_func)
    g_obj = OFunction(g_func)

    parameter.data = F_dirty
    parameter.complex_data_to_real()

    if use_wavelet:
        parameter.data = wav.decompose(parameter.data)

    opt = FISTA(guess_param=parameter, F_obj=F_obj, fx=chi2, gx=g_obj, noise=noise, verbose=False)
    obj, X = opt.run()

    if use_wavelet:
        X.data = wav.reconstruct(X.data)

    X.real_data_to_complex()
    F_residual = nufft.backward(dataset.residual)
    F[1, :, i, j] = X.data
    F[2, :, i, j] = X.convolve(normalized=True) + F_residual
    F[3, :, i, j] = F_residual


def main():
    cubes, mfs_images, spectral_idx, freq_f, lambda_reg, eta, output, index, nsigmas, verbose = getopt()
    index = int(index)
    mfs_counter = len(mfs_images)
    cube_counter = len(cubes)

    if cube_counter > 1:
        print("Reading images")
        reader = Reader(stokes_I_name=mfs_images[0], stokes_Q_name=mfs_images[1], stokes_U_name=mfs_images[2],
                        Q_cube_name=cubes[0], U_cube_name=cubes[1], freq_file_name=freq_f)
        Q, U, QU_header = reader.readQU()
        Q = np.flipud(Q)
        U = np.flipud(U)
        M = Q.shape[1]
        N = Q.shape[2]
    else:
        reader = Reader(freq_file_name=freq_f, numpy_file=images[0])
        Q, U = reader.readNumpyFile()
        Q = np.flipud(Q)
        U = np.flipud(U)

    I_header, I_mfs = reader.readImage(stokes="I")
    Q_header, Q_mfs = reader.readImage(stokes="Q")
    U_header, U_mfs = reader.readImage(stokes="U")

    if spectral_idx is None:
        spectral_idx = np.zeros_like(I_mfs)
    else:
        alpha_header, alpha_mfs = reader.readImage(name=spectral_idx)
        spectral_idx = alpha_mfs

    sigma_I = calculate_noise(image=I_mfs, xn=300, yn=300, nsigma=3.0, use_sigma_clipped_stats=True)
    sigma_Q = calculate_noise(image=Q_mfs, xn=300, yn=300, nsigma=3.0, use_sigma_clipped_stats=True)
    sigma_U = calculate_noise(image=U_mfs, xn=300, yn=300, nsigma=3.0, use_sigma_clipped_stats=True)
    sigma_Q_cube = calculate_noise(image=Q, xn=300, yn=300, nsigma=3.0, use_sigma_clipped_stats=True)
    sigma_U_cube = calculate_noise(image=U, xn=300, yn=300, nsigma=3.0, use_sigma_clipped_stats=True)

    print("Sigma MFS I: ", sigma_I)
    print("Sigma MFS Q: ", sigma_Q)
    print("Sigma MFS U: ", sigma_U)

    sigma_P = 0.5 * (sigma_Q + sigma_U)

    print("Sigma MFS P: ", sigma_P)

    print("I shape: ", I_mfs.shape)

    P_mfs = np.sqrt(Q_mfs ** 2 + U_mfs ** 2)
    pol_fraction = P_mfs / I_mfs
    workers_idxs, masked_idxs = make_mask_faraday(I_mfs, P_mfs, Q, U, spectral_idx, nsigmas[0] * sigma_I,
                                                  nsigmas[1] * sigma_P)

    nu = reader.readFreqsNumpyFile()

    sigma = 0.5 * (sigma_Q_cube + sigma_U_cube)

    global_dataset = Dataset(nu=nu, sigma=sigma)

    # Get Milky-way RM contribution
    f_sky = FaradaySky(filename="/raid/scratch/carcamo/repos/csromer/faradaysky/faraday2020v2.hdf5")

    mean_sky, std_sky = f_sky.galactic_rm_image(I_header, use_bilinear_interpolation=True)

    # Subtract Milky-way RM contribution
    P = Q + 1j * U
    P *= np.exp(-2j * mean_sky.value[np.newaxis, :, :] * (
            global_dataset.lambda2[:, np.newaxis, np.newaxis] - global_dataset.l2_ref))

    # Flagging #
    # First round using normal flagging
    normal_flagger = MeanFlagger(data=global_dataset, nsigma=5.0, delete_channels=True)
    idxs, outliers_idxs = normal_flagger.run()

    sigma = global_dataset.sigma
    nu = global_dataset.nu[::-1]

    data = P[idxs]
    noise = 1.0 / np.sqrt(np.sum(global_dataset.w))

    global_parameter = Parameter()
    global_parameter.calculate_cellsize(dataset=global_dataset, oversampling=8)

    folder = './joblib_mmap'
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
    except FileExistsError:
        pass

    output_file_mmap = os.path.join(folder, 'output_mmap')

    F = np.memmap(output_file_mmap, dtype=np.complex64, shape=(4, global_parameter.n, M, N), mode='w+')

    total_pixels = len(workers_idxs[0])
    print("LOS to reconstruct: ", total_pixels)
    print("Masked LOS: ", len(masked_idxs[0]))

    del global_dataset

    Parallel(n_jobs=-3, backend="multiprocessing", verbose=10)(delayed(reconstruct_cube)(
        F, data, sigma, nu, spectral_idx, None, workers_idxs, i, eta, False) for i in range(0, total_pixels))

    results_folder = "recon_l1_testW/"
    os.makedirs(results_folder, exist_ok=True)

    phi = global_parameter.phi
    phi_output_idx = np.where((phi > -1000) & (phi < 1000))
    phi = phi[phi_output_idx]
    dirty_F = F[0, phi_output_idx].squeeze()
    model_F = F[1, phi_output_idx].squeeze()
    restored_F = F[2, phi_output_idx].squeeze()
    residual_F = F[3, phi_output_idx].squeeze()

    abs_F = np.abs(restored_F)

    max_rotated_intensity = np.amax(abs_F, axis=0)
    max_faraday_depth_pos = np.argmax(abs_F, axis=0)
    max_rotated_intensity_image = np.where((I_mfs >= nsigmas[0] * sigma_I) & (P_mfs >= nsigmas[1] * sigma_P),
                                           max_rotated_intensity, np.nan)
    max_faraday_depth = np.where((I_mfs >= nsigmas[0] * sigma_I) & (P_mfs >= nsigmas[1] * sigma_P),
                                 phi[max_faraday_depth_pos], np.nan)
    # masked_pol_fraction = np.where((I_mfs >= nsigmas[0] * sigma_I) & (P_mfs >= nsigmas[1] * sigma_P), pol_fraction,
    #                               np.nan)
    P_from_faraday = np.sqrt(max_rotated_intensity ** 2 - (2.3 * sigma_P ** 2))
    Pfraction_from_faraday = P_from_faraday / I_mfs

    writer = Writer()

    writer.writeFITS(data=max_rotated_intensity_image, header=I_header,
                     output=results_folder + "max_rotated_intensity.fits")

    writer.writeFITS(data=max_faraday_depth, header=I_header, output=results_folder + "max_faraday_depth.fits")

    writer.writeFITS(data=Pfraction_from_faraday, header=I_header, output=results_folder + "polarization_fraction.fits")

    dirty_F[:, masked_idxs[0], masked_idxs[1]] = np.nan
    model_F[:, masked_idxs[0], masked_idxs[1]] = np.nan
    restored_F[:, masked_idxs[0], masked_idxs[1]] = np.nan
    residual_F[:, masked_idxs[0], masked_idxs[1]] = np.nan

    writer.writeFITSCube(dirty_F, I_header, len(phi), phi, np.abs(phi[1] - phi[0]),
                         output=results_folder + "faraday_dirty.fits")

    writer.writeFITSCube(model_F, I_header, len(phi), phi, np.abs(phi[1] - phi[0]),
                         output=results_folder + "faraday_model.fits")

    writer.writeFITSCube(restored_F, I_header, len(phi), phi, np.abs(phi[1] - phi[0]),
                         output=results_folder + "faraday_restored.fits")

    writer.writeFITSCube(residual_F, I_header, len(phi), phi, np.abs(phi[1] - phi[0]),
                         output=results_folder + "faraday_residual.fits")

    del global_parameter

    try:
        shutil.rmtree(folder)
    except:  # noqa
        print('Could not clean-up automatically.')


if __name__ == '__main__':
    main()
