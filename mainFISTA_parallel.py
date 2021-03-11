#!/usr/bin/env python3
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
from priors import TV, L1, chi2
from optimizer import Optimizer
from utilities import real_to_complex, complex_to_real, find_pixel, make_mask
from animations import create_animation
from joblib import Parallel, delayed, load, dump
from astropy.io import fits
import shutil


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
                        help="Input polarization fraction image", required=True)
    parser.add_argument("-s", "--sigmas",
                        help="Number of sigmas above on which calculation is done", required=False, default=8.0, type=float)
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

def calculateF(dftObject=None, W=np.array([]), F=np.array([]), P=np.array([]), idx_array=np.array([]), idx=0):
    i = idx_array[0][idx]
    j = idx_array[1][idx]
    F[:,i,j] = dftObject.backward(P[:,i,j])
    F_real = complex_to_real(F[:,i,j])

    lambda_l1 = 1e-5
    F_func = [chi2(b=P[:,i,j], dft_obj=dftObject, w=W), TV(0.0), L1(lambda_l1)]
    f_func = [chi2(b=P[:,i,j], dft_obj=dftObject, w=W)]
    g_func = [TV(0.0), L1(lambda_l1)]

    F_obj = OFunction(F_func)
    f_obj = OFunction(f_func)
    g_obj = OFunction(g_func)

    opt = Optimizer(F_obj.evaluate, F_obj.calculate_gradient,
                    F_real, maxiter=100, verbose=False)

    obj, X = opt.FISTA(f_obj.evaluate, g_obj.evaluate,
                       f_obj.calculate_gradient, g_obj, 0.5)

    F[:,i,j] = real_to_complex(X)

def main():

    images, pol_fraction, freq_f, reg_terms, output, index, nsigma, verbose = getopt()
    index = int(index)
    imag_counter = len(images)


    if imag_counter > 1:
        print("Reading images")
        reader = Reader(images[0], images[1], images[2], freq_f)
        Q,U = reader.readQU(memmap=True)
        Q = np.flipud(Q)
        U = np.flipud(U)
        M = Q.shape[1]
        N = Q.shape[2]
    else:
        reader = Reader(freq_file_name=freq_f, numpy_file=images[0])
        Q,U = reader.readNumpyFile()
        Q = np.flipud(Q)
        U = np.flipud(U)

    I_header, I = reader.readImage()
    pol_fraction_header, pol_fraction_data = reader.readImage(name=pol_fraction)
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
    mask_idx = make_mask(I, nsigma*sigma_I)

    sigma = np.sqrt((sigma_Q**2 + sigma_U**2)/2)
    W, K = pre_proc.calculate_W_K(sigma)

    lambda2, lambda2_ref, phi, phi_r = pre_proc.calculate_phi(W, K, times=8)

    print("Max I: ", pre_proc.calculate_max(I))

    P = Q + 1j * U

    dft = DFT1D(W, K, lambda2, lambda2_ref, phi)

    folder = './joblib_mmap'
    try:
       os.mkdir(folder)
    except FileExistsError:
       pass
    #data_file_mmap = os.path.join(folder, 'data_mmap')
    #dump(P, data_file_mmap)

    output_file_mmap = os.path.join(folder, 'output_mmap')
    #P = load(data_file_mmap, mmap_mode="r")
    #F = np.zeros((len(phi), M, N)) + 1j * np.zeros((len(phi), M, N))
    F = np.memmap(output_file_mmap, dtype=np.complex128, shape=(len(phi), M, N), mode='w+')

    #chi_degrees = np.arctan2(P.imag, P.real) * 180.0 / np.pi

    #arrays = np.array([lambda2, I, sigma_I, P.real, sigma_Q, P.imag, sigma_U])
    #arrays = np.transpose(arrays)
    #np.savetxt('A1314_A.txt', arrays, delimiter=' ', newline=os.linesep)

    #np.savetxt('A1314_south.txt', arrays, delimiter=' ', newline=os.linesep)

    #F = dft.backward(P)
    total_pixels = len(mask_idx[0])
    print("Pixels: ", total_pixels)

    #lambda_l1 = 1e-5
    #lambda_tv = 0.0
    #F_func = [chi2(P, dft, W), TV(lambda_tv), L1(lambda_l1)]
    #F_func = [chi2(dft_obj=dft, w=W), TV(lambda_tv), L1(lambda_l1)]
    #f_func = [chi2(dft_obj=dft, w=W)]
    #g_func = [TV(lambda_tv), L1(lambda_l1)]
    #g_func = [TV(lambda_tv), L1(lambda_l1)]

    #F_obj = OFunction(F_func)
    #f_obj = OFunction(f_func)
    #g_obj = OFunction(g_func)
    Parallel(n_jobs=-3, backend="multiprocessing", verbose=10)(delayed(calculateF)(dft, W, F, P, mask_idx, i) for i in range(0,total_pixels))
    """
    F_max = np.argmax(np.abs(F))
    print("Max RM: ", phi[F_max], "rad/m^2")
    R = dft.RMTF()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.figure(1)

    #plt.axvline(x=50, color='darkgrey', linestyle='-')
    plt.plot(lambda2, P.real, 'k.', label=r"Stokes $Q$")
    plt.plot(lambda2, P.imag, 'c.', label=r"Stokes $U$")
    plt.plot(lambda2, np.abs(P), 'g.', label=r"$|P|$")
    #plt.plot(lambda2, I, 'r.', label=r"Stokes $I$")
    plt.xlabel(r'$\lambda^2$[m$^{2}$]')
    plt.ylabel(r'Jy/beam')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("stokes.eps", bbox_inches ="tight")
    #plt.xlim([-500, 500])
    #plt.ylim([-0.75, 1.25])

    plt.figure(2)

    #plt.axvline(x=50, color='darkgrey', linestyle='-')
    plt.plot(phi, F.real, 'c--', label=r"Real part")
    plt.plot(phi, F.imag, 'c:', label=r"Imaginary part")
    plt.plot(phi, np.abs(F), 'k-', label=r"Amplitude")
    plt.xlabel(r'$\phi$[rad m$^{-2}$]')
    plt.ylabel(r'Jy m$^2$ rad$^{-1}$')
    plt.legend(loc='upper right')
    plt.xlim([-1000, 1000])
    plt.tight_layout()
    plt.savefig("FDS.eps", bbox_inches ="tight")
    #plt.ylim([-0.75, 1.25])

    plt.figure(3)

    #plt.axvline(x=50, color='darkgrey', linestyle='-')
    plt.plot(phi, R.real, 'c--', label=r"Real part")
    plt.plot(phi, R.imag, 'c:', label=r"Imaginary part")
    plt.plot(phi, np.abs(R), 'k-', label=r"Amplitude")
    plt.xlabel(r'$\phi$[rad m$^{-2}$]')
    plt.ylabel(r'RMTF')
    plt.legend(loc='upper right')
    plt.xlim([-1000, 1000])
    plt.tight_layout()
    plt.savefig("rmtf.eps", bbox_inches ="tight")
    #plt.ylim([-0.75, 1.25])


    plt.figure(4)

    plt.plot(lambda2, sigma_I, 'k.', label=r"$\sigma_I$")
    plt.plot(lambda2, sigma_Q, 'k.', label=r"$\sigma_Q$")
    plt.plot(lambda2, sigma_U, 'c.', label=r"$\sigma_U$")
    plt.xlabel(r'$\lambda^2$[m$^{2}$]')
    plt.ylabel(r'Jy/beam')
    plt.legend(loc='upper right')
    plt.tight_layout()
    #plt.xlim([-500, 500])
    #plt.ylim([-0.75, 1.25])

    plt.figure(5)
    #plt.axvline(x=50, color='darkgrey', linestyle='-')
    plt.plot(lambda2, W, 'k.', label=r"$W$")
    #plt.plot(lambda2, W.imag, 'c.', label=r"$W_U$")
    #plt.plot(lambda2, np.abs(P), 'k.', label=r"Amplitude")
    plt.xlabel(r'$\lambda^2$[m$^{2}$]')
    plt.ylabel(r'$(Jy^2/beam)^{-1}$')
    plt.legend(loc='upper right')
    plt.tight_layout()
    #plt.xlim([-500, 500])
    #plt.ylim([-0.75, 1.25])

    plt.figure(6)
    #plt.axvline(x=50, color='darkgrey', linestyle='-')
    plt.plot(lambda2, chi_degrees, 'k.', label=r"$\chi$")
    #plt.plot(lambda2, W.imag, 'c.', label=r"$W_U$")
    #plt.plot(lambda2, np.abs(P), 'k.', label=r"Amplitude")
    plt.xlabel(r'$\lambda^2$[m$^{2}$]')
    plt.ylabel(r'Polarization angle (degrees)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    #plt.xlim([-500, 500])
    #plt.ylim([-0.75, 1.25])

    plt.figure(7)

    #plt.axvline(x=50, color='darkgrey', linestyle='-')
    plt.plot(lambda2, I, 'k.', label=r"Stokes $I$")
    plt.xlabel(r'$\lambda^2$[m$^{2}$]')
    plt.ylabel(r'Jy/beam')
    plt.legend(loc='upper right')
    plt.tight_layout()
    #plt.xlim([-500, 500])
    #plt.ylim([-0.75, 1.25])



    plt.show()
    """
    phi_output_idx = np.where((phi>-1000) & (phi<1000))
    phi = phi[phi_output_idx]
    F = F[phi_output_idx]


    results_folder = "FISTA/"
    os.makedirs(results_folder, exist_ok=True)
    # Plot pixels of interest
    # Plot pixels of interest
    #============ Center-Source============
    #(568, 521)
    #=============Value===================
    #78.852776
    #============ North-West Source============
    #(772, 765)
    #=============Value===================
    #31.632423
    #============ Center-West Source============
    #(478, 901)
    #=============Value===================
    #66.87205
    #============ South-East Source============
    #(261, 343)
    #=============Value===================
    #27.568705
    #============ Center-South Extended Source============
    #(231, 542)
    #=============Value===================
    #105.13219
    #============ Center-West Extended Source============
    #(525, 900)
    #=============Value===================
    #353.85233
    x_pix = [521,765,901,343,542,900]
    y_pix = [568,772,478,261,231,525]
    file_names = ["center-source", "north-west-source", "center-west-source", "south-east-source", "center-south extended-source", "center-west extended-source"]
    names = ["Center Source", "North West Source", "Center West Source", "South East Source", "Center-South Extended Source", "Center-West Extended Source"]
    for i in range(0, len(names)):
        F_plot = F[:,y_pix[i], x_pix[i]]
        plt.figure()
        #plt.axvline(x=50, color='darkgrey', linestyle='-')
        plt.plot(phi, F_plot.real, 'c--', label=r"Real part")
        plt.plot(phi, F_plot.imag, 'c:', label=r"Imaginary part")
        plt.plot(phi, np.abs(F_plot), 'k-', label=r"Amplitude")
        plt.xlabel(r'$\phi$[rad m$^{-2}$]')
        plt.ylabel(r'Jy/beam m$^2$ rad$^{-1}$')
        plt.legend(loc='upper right')
        plt.xlim([-1000, 1000])
        plt.title(names[i])
        plt.tight_layout()
        plt.savefig(results_folder+file_names[i]+"_faradayrecon.eps", bbox_inches ="tight")
        #plt.ylim([-0.75, 1.25])

        plt.figure()
        #plt.axvline(x=50, color='darkgrey', linestyle='-')
        plt.plot(lambda2, P[:,y_pix[i], x_pix[i]].real, 'c--', label=r"Real part")
        plt.plot(lambda2, P[:,y_pix[i], x_pix[i]].imag, 'c:', label=r"Imaginary part")
        plt.plot(lambda2, np.abs(P[:, y_pix[i], x_pix[i]]), 'k-', label=r"Amplitude")
        plt.xlabel(r'$\lambda^2$[m$^{2}$]')
        plt.ylabel(r'Jy/beam')
        plt.legend(loc='upper right')
        plt.title(names[i])
        plt.tight_layout()
        plt.savefig(results_folder+file_names[i]+"_polintensity.eps", bbox_inches ="tight")

    header = reader.readHeader()
    writer = Writer()
    abs_F = np.abs(F)

    max_rotated_intensity = np.amax(abs_F, axis=0)
    max_faraday_depth_pos = np.argmax(abs_F, axis=0)
    max_faraday_depth = np.where(I>=nsigma*sigma_I, phi[max_faraday_depth_pos], np.nan)
    masked_pol_fraction = np.where(I>=nsigma*sigma_I, pol_fraction_data, np.nan)

    SNR_image = I/sigma_I
    SNR_image_vector = SNR_image[np.where(I>=nsigma*sigma_I)].flatten()
    pol_fraction_data_vector = pol_fraction_data[np.where(I>=nsigma*sigma_I)].flatten()

    plt.figure()
    #plt.plot(SNR_image_vector, 'c.', label="SNR")
    #plt.plot(pol_fraction_data_vector, 'k.', label="Polarization fraction")
    plt.scatter(pol_fraction_data_vector, SNR_image_vector)
    plt.xlabel("Polarization fraction")
    plt.ylabel("Signal-to-noise ratio")
    plt.tight_layout()
    plt.savefig(results_folder+"SNRvsPolFraction.png", bbox_inches ="tight", dpi=100)

    #SNRvsPol = np.where(I>=nsigma*sigma_I, SNR_image/pol_fraction_data, np.nan)
    writer.writeFITS(data=np.where(I>=nsigma*sigma_I, SNR_image, np.nan), header=pol_fraction_header, output=results_folder+"SNR.fits")
    #writer.writeFITS(data=SNRvsPol, header=pol_fraction_header, output="SNRvsPolFraction.fits")
    writer.writeFITS(data=masked_pol_fraction, header=pol_fraction_header, output=results_folder+"masked_pol_fraction.fits")
    writer.writeFITS(data=np.where(I>=nsigma*sigma_I, max_rotated_intensity/I, np.nan), header=header, output=results_folder+"leakage_map.fits")
    writer.writeFITS(data=max_faraday_depth, header=header, output=results_folder+"max_faraday_depth.fits")

    vmax = np.amax(np.amax(abs_F, axis=0))
    vmin = np.amin(np.amin(abs_F, axis=0))

    abs_F[:, masked_values[0], masked_values[1]] = np.nan
    writer.writeFITSCube(abs_F, header, len(phi), phi, np.abs(phi[1]-phi[0]), output=results_folder+"abs_F.fits")
    create_animation(header=header, cube_axis=phi, cube=abs_F, title='Faraday Depth Spectrum at {0:.4f} rad/m^2'.format(phi[0]), xlabel="Offset (degrees)", ylabel="Offset (degrees)", cblabel="Jy/beam", vmin=vmin, vmax=vmax, output_video=results_folder+"animation.mp4", repeat=True)

    try:
        shutil.rmtree(folder)
    except:
        print("Could not clean")

if __name__ == '__main__':
    main()
