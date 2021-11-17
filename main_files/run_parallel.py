import argparse
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, load, dump
from astropy.io import fits
import shutil

sys.path.insert(0, './../')
from framework.io import Reader, Writer
from framework.utils import calculate_noise, make_mask_faraday
from framework.base import Dataset
from framework.reconstruction import Parameter
from framework.transformers import DFT1D, NUFFT1D
from framework.objectivefunction import OFunction
from framework.objectivefunction import TSV, TV, L1, Chi2
from framework.optimization import FISTA, ADMM, SDMM, GradientBasedMethod
from framework.dictionaries.discrete import DiscreteWavelet
from framework.dictionaries.undecimated import UndecimatedWavelet


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
    output = vars(args)['output']
    index = vars(args)['index']
    nsigmas = vars(args)['sigmas']
    verbose = vars(args)['verbose']
    # check for --version or -V

    if args.version:
        print("this is myprogram version 0.1")
        sys.exit(1)

    return cubes, mfs_images, spec_idx, freq_f, reg_terms, output, index, nsigmas, verbose


def reconstruct_cube(F=None, data=None, sigma=None, nu=None, spectral_idx=None,
                     mask_idxs=None,
                     idx=None):
    i = mask_idxs[0][idx]
    j = mask_idxs[1][idx]

    dataset = Dataset(nu=nu, sigma=sigma, data=data[:, i, j], spectral_idx=spectral_idx[i, j])
    parameter = Parameter()
    parameter.calculate_cellsize(dataset=dataset, oversampling=8, verbose=False)

    dft = DFT1D(dataset=dataset, parameter=parameter)
    nufft = NUFFT1D(dataset=dataset, parameter=parameter, solve=True)

    F_dirty = dft.backward(dataset.data)
    idx_noise = np.where(np.abs(parameter.phi) > parameter.max_faraday_depth / 1.5)
    noise_F = np.std(0.5 * (F_dirty[idx_noise].real + F_dirty[idx_noise].imag))
    F[0, :, i, j] = F_dirty
    wav = UndecimatedWavelet(wavelet_name="coif2")

    lambda_l1 = np.sqrt(2 * len(dataset.data) + 4 * np.sqrt(len(dataset.data))) * noise_F * np.sqrt(0.5)
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
    parameter.data = wav.decompose(parameter.data)

    opt = FISTA(guess_param=parameter, F_obj=F_obj, fx=chi2, gx=g_obj, noise=noise_F, verbose=False)
    obj, X = opt.run()

    X.data = wav.reconstruct(X.data)
    X.real_data_to_complex()
    F_residual = nufft.backward(dataset.residual)
    F[1, :, i, j] = X.data
    F[2, :, i, j] = X.convolve(normalized=True) + F_residual
    F[3, :, i, j] = F_residual




def main():
    cubes, mfs_images, spectral_idx, freq_f, lambda_reg, output, index, nsigmas, verbose = getopt()
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

    nu = reader.readFreqsNumpyFile()

    sigma_I = calculate_noise(image=I_mfs, x0=0, xn=197, y0=0, yn=184)
    sigma_Q = calculate_noise(image=Q_mfs, x0=0, xn=197, y0=0, yn=184)
    sigma_U = calculate_noise(image=U_mfs, x0=0, xn=197, y0=0, yn=184)
    sigma_Q_cube = calculate_noise(image=Q, x0=0, xn=197, y0=0, yn=184)
    sigma_U_cube = calculate_noise(image=U, x0=0, xn=197, y0=0, yn=184)

    print("Sigma MFS I: ", sigma_I)
    print("Sigma MFS Q: ", sigma_Q)
    print("Sigma MFS U: ", sigma_U)

    sigma_P = 0.5 * (sigma_Q + sigma_U)

    print("Sigma MFS P: ", sigma_P)

    print("I shape: ", I_mfs.shape)

    P_mfs = np.sqrt(Q_mfs ** 2 + U_mfs ** 2)
    pol_fraction = P_mfs / I_mfs
    workers_idxs, masked_idxs = make_mask_faraday(I_mfs, P_mfs, Q, U, spectral_idx, nsigmas[0] * sigma_I, nsigmas[1] * sigma_P)

    sigma = np.sqrt((sigma_Q_cube ** 2 + sigma_U_cube ** 2) / 2)

    data = Q + 1j * U

    global_dataset = Dataset(nu=nu, sigma=sigma)

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
        F, data, sigma, nu, spectral_idx,
        workers_idxs, i) for i in range(0, total_pixels))

    results_folder = "recon/"
    os.makedirs(results_folder, exist_ok=True)

    phi = global_parameter.phi
    phi_output_idx = np.where((phi > -1000) & (phi < 1000))
    phi = phi[phi_output_idx]
    dirty_F = F[0, phi_output_idx].squeeze()
    model_F = F[1, phi_output_idx].squeeze()
    restored_F = F[2, phi_output_idx].squeeze()
    residual_F = F[3, phi_output_idx].squeeze()
    print(residual_F.shape)

    abs_F = np.abs(restored_F)

    max_rotated_intensity = np.amax(abs_F, axis=0)
    max_faraday_depth_pos = np.argmax(abs_F, axis=0)
    max_rotated_intensity_image = np.where((I_mfs >= nsigmas[0] * sigma_I) & (P_mfs >= nsigmas[1] * sigma_P),
                                           max_rotated_intensity, np.nan)
    max_faraday_depth = np.where((I_mfs >= nsigmas[0] * sigma_I) & (P_mfs >= nsigmas[1] * sigma_P),
                                 phi[max_faraday_depth_pos], np.nan)
    masked_pol_fraction = np.where((I_mfs >= nsigmas[0] * sigma_I) & (P_mfs >= nsigmas[1] * sigma_P), pol_fraction,
                                   np.nan)

    F_x, F_y = np.indices((M, N))

    F_at_peak = np.where((I_mfs >= nsigmas[0] * sigma_I) & (P_mfs >= nsigmas[1] * sigma_P),
                         restored_F[max_faraday_depth_pos, F_x, F_y], np.nan)
    abs_F_at_peak = np.abs(F_at_peak)
    P_from_faraday = np.sqrt(abs_F_at_peak ** 2 - (2.3 * sigma_P ** 2))
    Pfraction_from_faraday = P_from_faraday / I_mfs

    writer = Writer()

    writer.writeFITS(data=max_rotated_intensity_image, header=I_header,
                     output=results_folder + "max_rotated_intensity.fits")

    writer.writeFITS(data=max_faraday_depth, header=I_header, output=results_folder + "max_faraday_depth.fits")

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
