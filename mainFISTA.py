#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:45:38 2019

@author: miguel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:26:28 2019

@author: miguel
"""




import argparse
import numpy as np
from io_functions import Read, Write
import sys
from pre_processing import PreProcessor
from dfts import DFT1D
import matplotlib.pyplot as plt
from ofunction import OFunction
from priors import chi2, TV, L1
from optimizer import Optimizer
from utilities import real_to_complex, complex_to_real, find_pixel
def getopt():
     # initiate the parser
    parser = argparse.ArgumentParser(
        description='This is a program to blablabla')
    parser.add_argument("-V", "--version",
                        help="show program version", action="store_true")
    parser.add_argument("-v", "--verbose",
                        help="Print output", action="store_true")
    parser.add_argument("-i", "--images", nargs='+',
                        help="Input Stokes polarized images (Q,U FITS images) separated by a space", required=True)
    parser.add_argument("-f", "--freq-file",
                        help="Text file with frequency values")
    parser.add_argument("-o", "--output", nargs="+",
                        help="Path/s and/or name/s of the output file/s in FITS/npy format", required=True)
    parser.add_argument("-l", "--lambdas", nargs='*',
                        help="Regularization parameters separated by space")
    parser.add_argument("-I", "--index", nargs='?',
                        help="Selected index of the pixel on where the minimization is done", const=int, default=0)

    # read arguments from the command line
    args = parser.parse_args()

    reg_terms = vars(args)['lambdas']
    images = vars(args)['images']
    freq_f = vars(args)['freq_file']
    output = vars(args)['output']
    index = vars(args)['index']
    verbose = vars(args)['verbose']
    # check for --version or -V
    if args.version:
        print("this is myprogram version 0.1")
        sys.exit(1)
    return images, freq_f, reg_terms, output, index, verbose


def main():

    images, freq_f, reg_terms, output, index, verbose = getopt()
    index = int(index)
    imag_counter = len(images)

    """
    if imag_counter > 1:
        reader = Read(images[0], images[1], freq_f)
        Q,U = reader.readCube()
        Q = np.flipud(Q)
        U = np.flipud(U)
        M = Q.shape[1]
        N = Q.shape[2]
    else:
        reader = Read(freq_file_name=freq_f, numpy_file=images[0])
        Q,U = reader.readNumpyFile()
        Q = np.flipud(Q)
        U = np.flipud(U)
    """
    file = images[0]
    r_file = np.loadtxt(file)
    lambda2 = r_file[:, 0]
    Q = r_file[:, 1]
    sigma = r_file[:, 2]
    U = r_file[:, 3]

    #freqs = reader.getFileNFrequencies()
    pre_proc = PreProcessor(lambda2=lambda2)
    #pre_proc = PreProcessor(freqs)
    """
    if imag_counter > 1:
       i, j = find_pixel(M, N, index)
       Q = Q[:,i,j]
       U = U[:,i,j]
    else:
       Q = Q[index,:,0]
       U = U[index,:,1]
    """
    W, K = pre_proc.calculate_W_K(sigma)

    lambda2, lambda2_ref, phi, phi_r = pre_proc.calculate_phi(W, K)

    #W, K = pre_proc.calculate_W_K()

    P = Q + 1j * U

    dft = DFT1D(W, K, lambda2, lambda2_ref, phi)

    F = dft.backward(P)

    F_real = complex_to_real(F)

    lambda_l1 = 0.5
    lambda_tv = 1e-4
    F_func = [chi2(P, dft, W), TV(lambda_tv), L1(lambda_l1)]
    f_func = [chi2(P, dft, W)]
    g_func = [TV(lambda_tv), L1(lambda_l1)]

    F_obj = OFunction(F_func)
    f_obj = OFunction(f_func)
    g_obj = OFunction(g_func)

    print("Optimizing objetive function...")
    opt = Optimizer(F_obj.evaluate, F_obj.calculate_gradient,
                    F_real, maxiter=100, verbose=verbose)

    obj, X = opt.FISTA(f_obj.evaluate, g_obj.evaluate,
                       f_obj.calculate_gradient, g_obj, 0.5)
    print("Obj final: {0:0.5f}".format(obj))

    X = real_to_complex(X)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    #plt.axvline(x=50, color='darkgrey', linestyle='-')
    plt.plot(lambda2, P.real, 'c.', label=r"Real part")
    plt.plot(lambda2, P.imag, 'c.', label=r"Imaginary part")
    plt.plot(lambda2, np.abs(P), 'k.', label=r"Amplitude")
    plt.xlabel(r'$\phi$[rad m$^{-2}$]')
    plt.ylabel(r'Jy m$^2$ rad$^{-1}$')
    plt.legend(loc='upper right')
    #plt.xlim([-500, 500])
    #plt.ylim([-0.75, 1.25])

    plt.figure(2)

    #plt.axvline(x=50, color='darkgrey', linestyle='-')
    plt.plot(phi, F.real, 'c-', label=r"Real part")
    plt.plot(phi, F.imag, 'c-.', label=r"Imaginary part")
    plt.plot(phi, np.abs(F), 'k-', label=r"Amplitude")
    plt.xlabel(r'$\phi$[rad m$^{-2}$]')
    plt.ylabel(r'Jy m$^2$ rad$^{-1}$')
    plt.legend(loc='upper right')
    plt.xlim([-500, 500])
    plt.ylim([-0.75, 1.25])

    plt.figure(3)
    plt.plot(phi, X.real, 'c-', label=r"Real part")
    plt.plot(phi, X.imag, 'c-.', label=r"Imaginary part")
    plt.plot(phi, np.abs(X), 'k-', label=r"Amplitude")
    plt.xlabel(r'$\phi$[rad m$^{-2}$]')
    plt.ylabel(r'Jy m$^2$ rad$^{-1}$')
    plt.legend(loc='upper right')
    plt.xlim([-500, 500])
    plt.ylim([-0.75, 1.25])

    plt.show()


if __name__ == '__main__':
    main()
