#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:45:38 2019

@author: miguel
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:26:28 2019

@author: miguel
"""

import argparse
import os
import numpy as np
from io_functions import Reader, Writer
import sys
from pre_processing import PreProcessor
from dfts import DFT1D
import matplotlib.pyplot as plt
from ofunction import OFunction
from priors import TV, L1, Chi2
from optimizer import FISTA, ADMM, SDMM, GradientBasedMethod
from utilities import real_to_complex, complex_to_real, find_pixel, make_mask
from analytical_functions import Gaussian
from animations import create_animation
from joblib import Parallel, delayed, load, dump
from astropy.io import fits
import shutil
from scipy import signal


def getopt():
    # initiate the parser
    parser = argparse.ArgumentParser(
        description='This is a program to blablabla')
    parser.add_argument("-V", "--version",
                        help="show program version", action="store_true")
    parser.add_argument("-v", "--verbose",
                        help="Print output", action="store_true")
    parser.add_argument("-i", "--images", nargs=3,
                        help="Input Stokes polarized images (I,Q,U FITS images) separated by a space", required=True)
    parser.add_argument("-p", "--pol_fraction",
                        help="Input polarization fraction image", required=False)
    parser.add_argument("-s", "--sigmas",
                        help="Number of sigmas above on which calculation is done", required=False, default=8.0,
                        type=float)
    parser.add_argument("-f", "--freq-file",
                        help="Text file with frequency values")
    parser.add_argument("-o", "--output", nargs="*",
                        help="Path/s and/or name/s of the output file/s in FITS/npy format", required=True)
    parser.add_argument("-l", "--lambdas", nargs='*',
                        help="Regularization parameters separated by space")
    parser.add_argument("-I", "--index", nargs='?',
                        help="Selected index of the pixel on where the minimization is done", const=int, default=0)

    # read arguments from the command line
    args = parser.parse_args()

    reg_terms = vars(args)['lambdas']
    images = vars(args)['images']
    pol_fraction = vars(args)['pol_fraction']
    freq_f = vars(args)['freq_file']
    output = vars(args)['output']
    index = vars(args)['index']
    verbose = vars(args)['verbose']
    nsigma = vars(args)['sigmas']
    # check for --version or -V
    if args.version:
        print("this is myprogram version 0.1")
        sys.exit(1)
    return images, pol_fraction, freq_f, reg_terms, output, index, nsigma, verbose


def main():
    images, pol_fraction, freq_f, reg_terms, output, index, nsigma, verbose = getopt()
    index = int(index)
    imag_counter = len(images)

    if imag_counter > 1:
        print("Reading images")
        reader = Reader(stokes_I_name=images[0], Q_cube_name=images[1], U_cube_name=images[2], freq_file_name=freq_f)
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

    I_header, I = reader.readImage(stokes="I")
    # pol_fraction_header, pol_fraction_data = reader.readImage(name=pol_fraction)
    freqs = reader.readFreqsNumpyFile()
    pre_proc = PreProcessor(freqs=freqs)
    """
    if imag_counter > 1:
       i, j = find_pixel(M, N, index)
       Q = Q[:,i,j]
       U = U[:,i,j]
    else:
       Q = Q[index,:,0]
       U = U[index,:,1]
    """
    sigma_I = pre_proc.calculate_sigma(image=I, x0=0, xn=197, y0=0, yn=184)
    sigma_Q = pre_proc.calculate_sigmas_cube(image=Q, x0=0, xn=197, y0=0, yn=184)
    sigma_U = pre_proc.calculate_sigmas_cube(image=U, x0=0, xn=197, y0=0, yn=184)

    print("SigmaI: ", sigma_I)
    print("I shape: ", I.shape)
    mask_idx, masked_values = make_mask(I, nsigma * sigma_I)

    sigma = np.sqrt((sigma_Q ** 2 + sigma_U ** 2) / 2)

    np.save("sigmas.npy", sigma)

    sigma_noise = np.std(sigma)

    W, K = pre_proc.calculate_W_K(sigma=sigma)
    # W, K = pre_proc.calculate_W_K()

    lambda2, lambda2_ref, phi, phi_r = pre_proc.calculate_phi(W, K, times=7, verbose=True)

    gaussian_rmtf = Gaussian(x=phi, fwhm=pre_proc.rmtf_fwhm)
    clean_beam = gaussian_rmtf.run()

    print("Max I: ", pre_proc.calculate_max(I))
    # Strongest source
    # pix_source = 525,901
    # South source
    # pix_source = 233,543
    # South extended
    # pix_source = 270,583
    # North source
    # pix_source = 567,521

    # P = P[:,567,521]
    # I = I[:,568,521]

    P = Q[:, 525, 900] + 1j * U[:, 525, 900]

    np.save("polarized_emission.npy", P)

    dft = DFT1D(W, K, lambda2, lambda2_ref, phi)

    F = dft.backward(P)
    R = dft.RMTF()

    max_faraday_depth_pos = np.argmax(np.abs(F))
    max_faraday_depth = phi[max_faraday_depth_pos]

    print("Maximum Faraday Depth for this LOS: {0:.3f}".format(max_faraday_depth))
    F_real = complex_to_real(F)

    # Test different sigmas formulas
    lambda_test1 = np.sqrt(2 * len(P) + 4 * np.sqrt(len(P))) * sigma_noise
    lambda_test2 = np.sqrt(len(P)) * sigma_noise
    # print(lambda_test1)
    # print(lambda_test2)
    lambda_l1 = 5.5e-4
    # lambda_l1 = lambda_test2
    lambda_tv = 0.05

    chi2 = Chi2(b=P, dft_obj=dft, w=W)
    tv = TV(reg=lambda_tv)
    l1 = L1(reg=lambda_l1)
    # F_func = [chi2(P, dft, W), L1(lambda_l1)]
    F_func = [chi2, l1]
    f_func = [chi2]
    g_func = [l1]
    # g_func = [L1(lambda_l1)]

    F_obj = OFunction(F_func)
    f_obj = OFunction(f_func)
    g_obj = OFunction(g_func)

    # print("Optimizing objetive function...")
    opt = FISTA(F_obj=F_obj, i_guess=F_real, maxiter=400, verbose=True, fx=f_obj, gx=g_obj)
    # opt = GradientBasedMethod(F_obj=F_obj, i_guess=F_real, maxiter=10, verbose=verbose)
    obj, X = opt.run()

    # print(F_obj.getValues())
    # print("Obj final: {0:0.5f}".format(obj))

    X = real_to_complex(X)

    P_hat = dft.forward_normalized(X) * W
    # P_hat = W*P
    # res = P - P_hat

    # X_res = dft.backward(res)

    # recon = (signal.convolve(X, clean_beam, mode='same', method='auto') / np.sum(clean_beam)) + X_res

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, axes = plt.subplots(2, 2)

    # plt.axvline(x=50, color='darkgrey', linestyle='-')
    axes[0, 0].plot(lambda2, W * P.real, 'k.', label=r"Stokes $Q$")
    axes[0, 0].plot(lambda2, W * P.imag, 'c.', label=r"Stokes $U$")
    axes[0, 0].plot(lambda2, W * np.abs(P), 'g.', label=r"$|P|$")

    # plt.plot(lambda2, I, 'r.', label=r"Stokes $I$")
    axes[0, 0].set_xlabel(r'$\lambda^2$[m$^{2}$]')
    axes[0, 0].set_ylabel(r'Jy/beam')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].tick_params(axis='x', labelsize=12)
    axes[0, 0].tick_params(axis='y', labelsize=12)
    # axes[0, 0].tight_layout()
    # plt.savefig("stokes.eps", bbox_inches="tight")
    # plt.xlim([-500, 500])
    # plt.ylim([-0.75, 1.25])

    # plt.figure(2)

    # plt.axvline(x=50, color='darkgrey', linestyle='-')
    axes[1, 1].plot(phi, X.real, 'c--', label=r"Real part")
    axes[1, 1].plot(phi, X.imag, 'c:', label=r"Imaginary part")
    axes[1, 1].plot(phi, np.abs(X), 'k-', label=r"Amplitude")
    # plt.plot(lambda2, I, 'r.', label=r"Stokes $I$")
    axes[1, 1].set_xlabel(r'$\phi$[rad m$^{-2}$]')
    axes[1, 1].set_ylabel(r'Jy m$^2$ rad$^{-1}$')
    axes[1, 1].legend(loc='upper right')
    # axes[1, 1].tight_layout()
    # axes[1, 1].savefig("X.eps", bbox_inches="tight")
    axes[1, 1].set_xlim([-1000, 1000])
    # plt.ylim([-0.75, 1.25])

    # plt.figure(3)

    # plt.axvline(x=50, color='darkgrey', linestyle='-')
    axes[1, 0].plot(phi, F.real, 'c--', label=r"Real part")
    axes[1, 0].plot(phi, F.imag, 'c:', label=r"Imaginary part")
    axes[1, 0].plot(phi, np.abs(F), 'k-', label=r"Amplitude")
    axes[1, 0].set_xlabel(r'$\phi$[rad m$^{-2}$]')
    axes[1, 0].set_ylabel(r'Jy m$^2$ rad$^{-1}$')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].tick_params(axis='x', labelsize=12)
    axes[1, 0].tick_params(axis='y', labelsize=12)
    axes[1, 0].set_xlim([-1000, 1000])
    # plt.tight_layout()
    # plt.savefig("FDS.eps", bbox_inches="tight")
    # plt.ylim([-0.75, 1.25])

    # plt.figure(5)

    # plt.axvline(x=50, color='darkgrey', linestyle='-')
    axes[0, 1].plot(lambda2, P_hat.real, 'k.', label=r"Stokes $Q$")
    axes[0, 1].plot(lambda2, P_hat.imag, 'c.', label=r"Stokes $U$")
    axes[0, 1].plot(lambda2, np.abs(P_hat), 'g.', label=r"$|P|$")
    # plt.plot(lambda2, I, 'r.', label=r"Stokes $I$")
    axes[0, 1].set_xlabel(r'$\lambda^2$[m$^{2}$]')
    axes[0, 1].set_ylabel(r'Jy/beam')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].tick_params(axis='x', labelsize=12)
    axes[0, 1].tick_params(axis='y', labelsize=12)
    # axes[0, 1].tight_layout()
    # plt.savefig("stokes.eps", bbox_inches="tight")
    # plt.xlim([-500, 500])
    # plt.ylim([-0.75, 1.25])

    plt.figure(2)
    # plt.axvline(x=50, color='darkgrey', linestyle='-')
    plt.plot(lambda2, W * P.real, 'k.', label=r"Stokes $Q$")
    plt.plot(lambda2, W * P.imag, 'c.', label=r"Stokes $U$")
    plt.plot(lambda2, W * np.abs(P), 'g.', label=r"$|P|$")
    plt.plot(lambda2, P_hat.real, 'k.', label=r"Reconstructed Stokes $Q$")
    plt.plot(lambda2, P_hat.imag, 'c.', label=r"Reconstructed Stokes $U$")
    plt.plot(lambda2, np.abs(P_hat), 'g.', label=r"Reconstructed $|P|$")
    # plt.plot(lambda2, I, 'r.', label=r"Stokes $I$")
    plt.xlabel(r'$\lambda^2$[m$^{2}$]')
    plt.ylabel(r'Jy/beam')
    plt.legend(loc='upper right')
    plt.xticks(size=12)
    plt.yticks(size=12)
    # axes[0, 0].tight_layout()
    # plt.savefig("stokes.eps", bbox_inches="tight")
    # plt.xlim([-500, 500])
    # plt.ylim([-0.75, 1.25])

    plt.figure(3)

    # plt.axvline(x=50, color='darkgrey', linestyle='-')
    plt.plot(phi, R.real, 'c--', label=r"Real part")
    plt.plot(phi, R.imag, 'c:', label=r"Imaginary part")
    plt.plot(phi, np.abs(R), 'k-', label=r"Amplitude")
    plt.plot(phi, clean_beam, 'r-', label=r"Fit")
    plt.xlabel(r'$\phi$[rad m$^{-2}$]')
    plt.ylabel(r'Jy m$^2$ rad$^{-1}$')
    plt.legend(loc='upper right')
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlim([-1000, 1000])
    plt.tight_layout()
    plt.savefig("R.eps", bbox_inches="tight")
    # plt.ylim([-0.75, 1.25])
    """

    # plt.axvline(x=50, color='darkgrey', linestyle='-')
    plt.plot(phi, recon.real, 'c--', label=r"Real part")
    plt.plot(phi, recon.imag, 'c:', label=r"Imaginary part")
    plt.plot(phi, np.abs(recon), 'k-', label=r"Amplitude")
    plt.xlabel(r'$\phi$[rad m$^{-2}$]')
    plt.ylabel(r'Jy m$^2$ rad$^{-1}$')
    plt.legend(loc='upper right')
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlim([-1000, 1000])
    plt.tight_layout()
    plt.savefig("R.eps", bbox_inches="tight")
    # plt.ylim([-0.75, 1.25])
    """

    plt.show()


if __name__ == '__main__':
    main()
