#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:12:45 2019

@author: miguel
"""
from scipy.constants import pi
from scipy.constants import speed_of_light as c
import numpy as np


class PreProcessor:
    def __init__(self, m=0, freqs=[], lambda2=[]):
        self.freqs = freqs

        if len(lambda2):
            self.m = len(lambda2)
            self.lambda2 = lambda2
        else:
            self.m = len(freqs)
            self.nu_to_l2()

        self.rmtf_fwhm = 0.0
        self.max_recovered_width = 0.0
        self.max_faraday_depth = 0.0

    def nu_to_l2(self):
        self.lambda2 = (c / self.freqs) ** 2
        self.lambda2 = self.lambda2[::-1]

    def calculate_max(self, image=None):
        max = np.amax(image)
        where_max = np.unravel_index(np.argmax(image, axis=None), image.shape)
        return max, where_max

    def calculate_min(self, image=None, axis=0):
        min = np.amin(image)
        where_min = np.unravel_index(np.argmin(image, axis=None), image.shape)
        return min, where_min

    def calculate_sigmas_cube(self, image=None, x0=0, xn=0, y0=0, yn=0):
        sigmas = np.zeros(len(self.freqs))

        for i in range(len(self.freqs)):
            sigmas[i] = np.sqrt(np.mean(image[i, y0:yn, x0:xn] ** 2))

        return sigmas

    def calculate_sigma(self, image=None, x0=0, xn=0, y0=0, yn=0):
        sigma = np.sqrt(np.mean(image[y0:yn, x0:xn] ** 2))

        return sigma

    def calculate_l2ref(self, W, K):
        return (1. / K) * np.sum(W * self.lambda2)

    def calculate_phi(self, W, K, times=4, verbose=False):

        l2 = self.lambda2

        l2_min = np.min(l2)
        l2_max = np.max(l2)

        l2_ref = self.calculate_l2ref(W, K)
        delta_l2_min = np.min(np.abs(np.diff(l2)))
        delta_l2_mean = np.mean(np.abs(np.diff(l2)))
        delta_l2_max = np.max(np.abs(np.diff(l2)))

        delta_phi_fwhm = 2.0 * np.sqrt(3.0) / (l2_max - l2_min)  # FWHM of the FPSF
        delta_phi_theo = pi / l2_min

        delta_phi = min(delta_phi_fwhm, delta_phi_theo)

        phi_max = np.sqrt(3) / delta_l2_mean
        phi_max = max(phi_max, delta_phi_fwhm * 10.0)

        self.rmtf_fwhm = delta_phi_fwhm
        self.max_recovered_width = delta_phi_theo
        self.max_faraday_depth = phi_max
        if verbose:
            print("Minimum Lambda-squared: {0:.3f} m^2".format(l2_min))
            print("Maximum Lambda-squared: {0:.3f} m^2".format(l2_max))
            print("delta Lambda-squared min: {0:.3e} m^2".format(delta_l2_min))
            print("delta Lambda-squared max: {0:.3f} m^2".format(delta_l2_max))
            print("delta Lambda-squared mean: {0:.3e} m^2".format(delta_l2_mean))
            print("FWHM of the main peak of the RMTF: {0:.3f} rad/m^2".format(delta_phi_fwhm))
            print("Maximum recovered width structure: {0:.3f} rad/m^2".format(delta_phi_theo))
            print("Maximum Faraday Depth to which one has more than 50% sensitivity: {0:.3f}".format(phi_max))

        phi_r = delta_phi / times

        temp = np.int32(np.floor(2 * phi_max / phi_r))
        n = int(temp - np.mod(temp, 32))

        phi_r = 2 * phi_max / n

        phi = phi_r * np.arange(-(n / 2), (n / 2), 1)

        return l2, l2_ref, phi, phi_r

    def calculate_W_K(self, sigma=np.array([])):

        if not len(sigma):
            w = np.ones(self.m)
        else:
            w = 1.0 / (sigma ** 2)

        k = np.sum(w)

        return w, k
