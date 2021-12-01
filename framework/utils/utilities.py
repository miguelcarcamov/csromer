#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:33:53 2019

@author: miguel
"""
import numpy as np
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats


def real_to_complex(z):  # real vector of length 2n -> complex of length n
    return z[:len(z) // 2] + 1j * z[len(z) // 2:]


def complex_to_real(z):  # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


def make_mask(I=np.array([]), sigma=0.0):
    indexes = np.where(I >= sigma)
    masked_idxs = np.where(I < sigma)
    return indexes, masked_idxs


def make_mask_faraday(I=np.array([]), P=np.array([]), cube_Q=None, cube_U=None, spectral_idx=None,
                      sigma_I=0.0, sigma_P=0.0):
    if cube_Q is not None:
        Q_nan = np.isnan(cube_Q).any(axis=0)

    if cube_U is not None:
        U_nan = np.isnan(cube_U).any(axis=0)

    if spectral_idx is None:
        arr = (I >= sigma_I) & (P >= sigma_P) & ~Q_nan & ~U_nan
        indexes = np.where(arr)
        masked_idxs = np.where(np.logical_not(arr))
    else:
        isnan = np.isnan(spectral_idx)
        arr = (I >= sigma_I) & (P >= sigma_P) & ~isnan & ~Q_nan & ~U_nan
        indexes = np.where(arr)
        masked_idxs = np.where(np.logical_not(arr))
    return indexes, masked_idxs


def calculate_noise(image=np.array([]), x0=0, xn=0, y0=0, yn=0, nsigma=3, use_sigma_clipped_stats=False):
    if use_sigma_clipped_stats:
        if image.ndim > 2:
            sigma = np.nanstd(image[:, y0:yn, x0:xn], axis=(1, 2))
        else:
            sigma = np.nanstd(image[y0:yn, x0:xn])
    else:
        if image.ndim > 2:
            mean, median, std = sigma_clipped_stats(image, sigma=nsigma, axis=(1, 2))
        else:
            mean, median, std = sigma_clipped_stats(image, sigma=nsigma)
        sigma = std

    return sigma
