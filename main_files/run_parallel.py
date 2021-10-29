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
from framework.utils import calculate_noise, make_mask
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
    parser.add_argument("-s", "--sigmas", nargs=2,
                        help="Number of sigmas in total intensity (I) and polarized intensity (P) above on which "
                             "calculation is done",
                        required=True,
                        type=float)
    parser.add_argument("-f", "--freq-file",
                        help="Text file with frequency values", required=True)
    parser.add_argument("-a", "--spectral_idx",
                        help="Input Spectral Index Image", required=False)
    parser.add_argument("-o", "--output", nargs="*",
                        help="Path/s and/or name/s of the output file/s in FITS/npy format", required=True)
    parser.add_argument("-l", "--lambdas", nargs='*',
                        help="Regularization parameters separated by space")
    parser.add_argument("-I", "--index", nargs='?',
                        help="Selected index of the pixel on where the minimization is done", const=int, default=0)
    parser = argparse.ArgumentParser(
        description='This is a program to blablabla')

    # read arguments from the command line
    args = parser.parse_args()

    reg_terms = vars(args)['lambdas']
    cubes = vars(args)['cubes']
    mfs_images = vars(args)['mfs']
    freq_f = vars(args)['freq_file']
    output = vars(args)['output']
    index = vars(args)['index']
    verbose = vars(args)['verbose']
    nsigmas = vars(args)['sigmas']

    images = vars(args)['images']
    spec_idx = vars(args)['spectral_idx']
    pol_fraction = vars(args)['pol_fraction']
    freq_f = vars(args)['freq_file']
    output = vars(args)['output']
    index = vars(args)['index']
    verbose = vars(args)['verbose']
    nsigma = vars(args)['sigmas']
    lambda_reg = vars(args)['lambda']
    # check for --version or -V
    if args.version:
        print("this is myprogram version 0.1")
        sys.exit(1)

    return cubes, mfs_images, freq_f, reg_terms, output, index, nsigmas, verbose
    return images, pol_fraction, spec_idx, freq_f, output, index, nsigma, lambda_reg, verbose


def reconstruct_cube(faraday_depth_cube=None, data=None, sigma=None, nu=None, spectral_idx=None, mask_idxs=None,
                     idx=None):
    i = mask_idxs[0][idx]
    j = mask_idxs[1][idx]

    dataset = Dataset(nu=nu, sigma=sigma, data=data[:, i, j], spectral_idx=spectral_idx[i, j])
    parameter = Parameter()
    parameter.calculate_cellsize(dataset=dataset, verbose=False)

    dft = DFT1D(dataset=dataset, parameter=parameter)
    nufft = NUFFT1D(dataset=dataset, parameter=parameter, solve=True)

    F_dirty = dft.backward(dataset.data)
    idx_noise = np.where(np.abs(parameter.phi) > parameter.max_faraday_depth / 1.5)
    noise_F = np.std(0.5 * (F_dirty[idx_noise].real + F_dirty[idx_noise].imag))

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
    faraday_depth_cube[:, i, j] = X.data


def main():
    images, pol_fraction, spectral_idx, freq_f, output, index, nsigma, lambda_reg, verbose = getopt()
    index = int(index)
    imag_counter = len(images)

    if imag_counter > 1:
        print("Reading images")
        reader = Reader(stokes_I_name=images[0], Q_cube_name=images[1], U_cube_name=images[2], freq_file_name=freq_f)
        Q, U, QU_header = reader.readQU(memmap=True)
        Q = np.flipud(Q)
        U = np.flipud(U)
        M = Q.shape[1]
        N = Q.shape[2]
    else:
        reader = Reader(freq_file_name=freq_f, numpy_file=images[0])
        Q, U = reader.readNumpyFile()
        Q = np.flipud(Q)
        U = np.flipud(U)

    I_header, I = reader.readImage(stokes="I")

    if spectral_idx is None:
        spectral_idx = np.zeros_like(I)

    pol_fraction_header, pol_fraction_data = reader.readImage(name=pol_fraction)
    nu = reader.readFreqsNumpyFile()

    sigma_I = calculate_noise(image=I, x0=0, xn=197, y0=0, yn=184)
    sigma_Q = calculate_noise(image=Q, x0=0, xn=197, y0=0, yn=184)
    sigma_U = calculate_noise(image=U, x0=0, xn=197, y0=0, yn=184)

    mask_idx, masked_values = make_mask(I, nsigma * sigma_I)

    sigma = np.sqrt((sigma_Q ** 2 + sigma_U ** 2) / 2)
    data = Q + 1j * U

    global_dataset = Dataset(nu=nu, sigma=sigma)

    global_parameter = Parameter()
    global_parameter.calculate_cellsize(dataset=global_dataset)

    folder = './joblib_mmap'
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    output_file_mmap = os.path.join(folder, 'output_mmap')
    F = np.memmap(output_file_mmap, dtype=np.complex128, shape=(global_parameter.n, M, N), mode='w+')

    total_pixels = len(mask_idx[0])
    print("LOS to reconstruct: ", total_pixels)

    del global_dataset

    Parallel(n_jobs=-3, backend="multiprocessing", verbose=10)(delayed(reconstruct_cube)(
        F, data, sigma, nu, spectral_idx, mask_idx, i) for i in range(0, total_pixels))

    writer = Writer()
    writer.writeFITSCube(F.real, I_header, global_parameter.n, global_parameter.phi, global_parameter.cellsize,
                         output="model_Q.fits")

    writer.writeFITSCube(F.imag, I_header, global_parameter.n, global_parameter.phi, global_parameter.cellsize,
                         output="model_U.fits")

    try:
        shutil.rmtree(folder)
    except:  # noqa
        print('Could not clean-up automatically.')


if __name__ == '__main__':
    main()
