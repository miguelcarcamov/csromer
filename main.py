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
from priors import TV, L1
from optimizer import Optimizer
from utilities import real_to_complex, complex_to_real, find_pixel

def getopt():
     # initiate the parser
    parser = argparse.ArgumentParser(description='This is a program to blablabla')
    parser.add_argument("-V", "--version", help="show program version", action="store_true")
    parser.add_argument("-i", "--images", nargs='*', help="Input Stokes polarized images (Q,U FITS images) separated by a space", required=True)
    parser.add_argument("-f", "--freq-file", help="Text file with frequency values", required=True)
    parser.add_argument("-o", "--output", nargs="*", help="Path/s and/or name/s of the output file/s in FITS/npy format", required=True)
    parser.add_argument("-l", "--lambdas", nargs='*', help="Regularization parameters separated by space")
    parser.add_argument("-I", "--index", nargs='?', help="Selected index of the pixel on where the minimization is done", const=int, default=0)
    
    # read arguments from the command line
    args = parser.parse_args()
    
    reg_terms = vars(args)['lambdas']
    images = vars(args)['images']
    freq_f = vars(args)['freq_file']
    output = vars(args)['output']
    index = vars(args)['index']
    # check for --version or -V
    if args.version:
        print("this is myprogram version 0.1")
        sys.exit(1)
    return images, freq_f, reg_terms, output, index

def main():
   
    images, freq_f, reg_terms, output, index = getopt()
    index = int(index)
    imag_counter = len(images)
    
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
        
    freqs = reader.getFileNFrequencies()
    pre_proc = PreProcessor(freqs)
    
    if imag_counter > 1:
       i, j = find_pixel(M, N, index)
       Q = Q[:,i,j]
       U = U[:,i,j]
    else:
       Q = Q[index,:,0]
       U = U[index,:,1]
    
    lambda2, lambda2_ref, phi, phi_r = pre_proc.calculate_lambda2_phi()
    
    W, K = pre_proc.calculate_W_K()
    
    P = Q + 1j * U
    
    dft = DFT1D(W, K, lambda2, lambda2_ref, phi)
    
    F = dft.backward_dirty(P)
    
    F_real = complex_to_real(F)
    
    priors = [L1(0.001), TV(0.01)]
    
    objf = OFunction(P, dft, priors)
    
    opt = Optimizer(objf.evaluate, objf.calculate_gradient, F_real, 10000, method='CG', verbose=False)
    
    res = opt.optimize()
    
    X = real_to_complex(res.x)
    
    plt.figure(1)
    plt.plot(phi,np.abs(X))
    plt.plot(phi,X.real)
    plt.plot(phi,X.imag)
    plt.xlabel(r'$\phi$[rad m$^{-2}$]')
    plt.ylabel(r'Jy m$^2$ rad$^{-1}$')
    plt.legend((r"Amplitude", r"Real part", r"Imaginary part"), loc='upper right')
    plt.xlim([-500, 500])

    
    plt.figure(2)
    plt.plot(phi,np.abs(F))
    plt.plot(phi,F.real)
    plt.plot(phi,F.imag)
    plt.xlabel(r'$\phi$[rad m$^{-2}$]')
    plt.ylabel(r'Jy m$^2$ rad$^{-1}$')
    plt.legend((r"Amplitude", r"Real part", r"Imaginary part"), loc='upper right')
    plt.xlim([-500, 500])
    plt.show()
    
    
    
if __name__ == '__main__':
    main()