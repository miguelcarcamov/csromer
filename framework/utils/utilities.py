#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:33:53 2019

@author: miguel
"""
import numpy as np
from astropy.wcs import WCS


def real_to_complex(z):  # real vector of length 2n -> complex of length n
    return z[:len(z) // 2] + 1j * z[len(z) // 2:]


def complex_to_real(z):  # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


def make_mask(I=np.array([]), sigma=0.0):
    indexes = np.where(I >= sigma)
    masked_values = np.where(I < sigma)
    return indexes, masked_values


def make_mask_faraday(I=np.array([]), P=np.array([]), sigma_I=0.0, sigma_P=0.0):
    indexes = np.where((I >= sigma_I) & (P >= sigma_P))
    masked_values = np.where((I < sigma_I) & (P < sigma_P))
    return indexes, masked_values


def calculate_noise(image=np.array([]), x0=0, xn=0, y0=0, yn=0):
    if image.ndim > 2:
        sigma = np.std(image[:, y0:yn, x0:xn], axis=(1, 2))
    else:
        sigma = np.std(image[y0:yn, x0:xn])

    return sigma
