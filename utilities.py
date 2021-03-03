#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:33:53 2019

@author: miguel
"""
import numpy as np
from astropy.wcs import WCS


def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z) // 2] + 1j * z[len(z) // 2:]


def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


def find_pixel(M, N, contiguous_id):
    for i in range(M):
        for j in range(N):
            if contiguous_id == N * i + j:
                return i, j

def make_mask(I=np.array([]), sigma=0.0):
    indexes = np.where(I >= sigma)
    return indexes
