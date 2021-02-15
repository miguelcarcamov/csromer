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
            self.lambda2 = (c / self.freqs)**2
            self.lambda2 = self.lambda2[::-1]

    def calculate_max(self, image=None):
        max = np.amax(image)
        where_max = np.unravel_index(np.argmax(image, axis=None), image.shape)
        return max, where_max

    def calculate_min(self, image=None, axis=0):
        min = np.amin(image)
        where_min = np.unravel_index(np.argmax(image, axis=None), image.shape)
        return min, where_min

    def calculate_sigmas_cube(self, image=None, x0=0, xn=0, y0=0, yn=0):
        sigmas = np.zeros(len(self.freqs))

        for i in range(len(self.freqs)):
            sigmas[i] = np.sqrt(np.mean(image[i, y0:yn, x0:xn]**2))

        return sigmas

    def calculate_sigma(self, image=None, x0=0, xn=0, y0=0, yn=0):
        sigma = np.sqrt(np.mean(image[y0:yn, x0:xn]**2))

        return sigma

    def calculate_phi(self, W, K, times=4):

        l2 = self.lambda2

        l2_max = l2[-1]
        l2_min = l2[0]

        l2_ref = (1./K) * np.sum(W * self.lambda2)

        delta_l2 = np.abs(l2[1] - l2[0])

        delta_phi_fwhm = 2.0 * np.sqrt(3.0) / \
            (l2_max - l2_min)  # FWHM of the FPSF
        delta_phi_theo = pi / l2_min

        delta_phi = min(delta_phi_fwhm, delta_phi_theo)

        phi_max = np.sqrt(3) / (delta_l2)

        phi_r = delta_phi / times

        temp = np.int(np.floor(2 * phi_max / phi_r))
        n = int(temp - np.mod(temp, 32))

        phi_r = 2 * phi_max / n

        phi = phi_r * np.arange(-(n / 2), (n / 2), 1)

        return l2, l2_ref, phi, phi_r


    def calculate_W_K(self, sigma=np.array([])):

        if not len(sigma):
            W = np.ones(self.m)
        else:
            W = 1.0 /(sigma**2)

        K = np.sum(W)

        return W, K
