#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:33:53 2019

@author: miguel
"""
import numpy as np
from astropy.stats import sigma_clipped_stats


def next_power_2(n):
    count = 0
    # First n in the below
    # condition is for the
    # case where n is 0
    if n and not (n & (n - 1)):
        return n

    while n != 0:
        n >>= 1
        count += 1

    return 1 << count


def real_to_complex(z):  # real vector of length 2n -> complex of length n
    return z[:len(z) // 2] + 1j * z[len(z) // 2:]


def complex_to_real(z):  # complex vector of length n -> real of length 2n
    return np.concatenate([z.real, z.imag])


def make_mask(I=np.array([]), sigma=0.0):
    indexes = np.where(I >= sigma)
    masked_idxs = np.where(I < sigma)
    return indexes, masked_idxs


def make_mask_faraday(
    I=np.array([]),
    P=np.array([]),
    cube_Q=None,
    cube_U=None,
    spectral_idx=None,
    sigma_I=0.0,
    sigma_P=0.0,
):
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


def calculate_noise(
    image=np.array([]),
    x0=None,
    xn=None,
    y0=None,
    yn=None,
    nsigma=3,
    cenfunc='mean',
    stdfunc='mad_std',
    use_sigma_clipped_stats=False,
):
    if x0 is None:
        x0 = 0

    if y0 is None:
        y0 = 0

    if xn is None:
        if image.ndim > 2:
            xn = image.shape[2]
        else:
            xn = image.shape[1]

    if yn is None:
        if image.ndim > 2:
            yn = image.shape[1]
        else:
            yn = image.shape[0]

    if not use_sigma_clipped_stats:
        if image.ndim > 2:
            sigma = np.nanstd(image[:, y0:yn, x0:xn], axis=(1, 2))
        else:
            sigma = np.nanstd(image[y0:yn, x0:xn])
    else:
        if image.ndim > 2:
            _, _, sigma = sigma_clipped_stats(
                image[:, y0:yn, x0:xn], sigma=nsigma, cenfunc=cenfunc, stdfunc=stdfunc, axis=(1, 2)
            )
        else:
            _, _, sigma = sigma_clipped_stats(image[y0:yn, x0:xn], cenfunc=cenfunc, stdfunc=stdfunc)

    return sigma
