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
    m = 0
    freqs=[]
    lambda2=[]

    def __init__(self, freqs=[], lambda2 = []):
       self.freqs = freqs
       
       if len(lambda2):
           self.m = len(lambda2)
           self.lambda2 = lambda2
       else:
           self.m = len(freqs)
           self.lambda2 = (c/self.freqs)**2
           self.lambda2 = self.lambda2[::-1]
          
    def calculate_phi(self, times=4):

        l2 = self.lambda2

        l2_max = l2[-1]
        l2_min = l2[0]

        l2_ref = (l2_max+l2_min)/2.0

        delta_l2 = np.abs(l2[1]-l2[0])

        delta_phi_fwhm = 2.0*np.sqrt(3.0)/(l2_max-l2_min) #FWHM of the FPSF
        delta_phi_theo = pi/l2_min

        delta_phi = min(delta_phi_fwhm, delta_phi_theo)

        phi_max = np.sqrt(3)/(delta_l2)

        phi_r = delta_phi/times

        temp = np.int(np.floor(2*phi_max/phi_r))
        n = int(temp-np.mod(temp,32))

        phi_r = 2*phi_max/n;

        phi = phi_r*np.arange(-(n/2),(n/2), 1)

        return l2, l2_ref, phi, phi_r

    def calculate_W_K(self, sigma=[]):
        if not len(sigma):
            W = np.ones(self.m)
        else:
            W = np.array(1/(sigma**2))

        K = 1.0 / np.sum(W)

        return W, K
