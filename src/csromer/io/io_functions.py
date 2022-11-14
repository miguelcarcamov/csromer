#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:45:42 2019

@author: miguel
"""
import sys

import dask.array as da
import numpy as np
from astropy.io import fits


def filter_cubes(data_I, data_Q, data_U, header, additional_outlier_idxs=None):
    init_freq = header["CRVAL3"]
    nfreqs = header["NAXIS3"]
    step_freq = header["CDELT3"]
    nu = init_freq + np.arange(0, nfreqs) * step_freq
    sum_I = np.nansum(data_I, axis=(1, 2))
    sum_Q = np.nansum(data_Q, axis=(1, 2))
    sum_U = np.nansum(data_U, axis=(1, 2))
    correct_freqs = np.where((sum_I != 0.0) | (sum_Q != 0.0) | (sum_U != 0.0))[0]
    if additional_outlier_idxs:
        correct_freqs = np.setxor1d(correct_freqs, additional_outlier_idxs)
    filtered_data = 100.0 * (nfreqs - len(correct_freqs)) / nfreqs
    print("Filtering {0:.2f}% of the total data".format(filtered_data))
    return (
        data_I[correct_freqs],
        data_Q[correct_freqs],
        data_U[correct_freqs],
        nu[correct_freqs],
    )


class Reader:

    def __init__(
        self,
        stokes_I_name=None,
        stokes_Q_name=None,
        stokes_U_name=None,
        Q_cube_name=None,
        U_cube_name=None,
        freq_file_name=None,
        numpy_file=None,
    ):
        self.stokes_I_name = stokes_I_name
        self.stokes_Q_name = stokes_Q_name
        self.stokes_U_name = stokes_U_name
        self.Q_cube_name = Q_cube_name
        self.U_cube_name = U_cube_name
        self.freq_file_name = freq_file_name
        self.numpy_file = numpy_file

    # this has to be separated

    def readCube(self, file=None, stokes=None, memmap=True):
        if stokes is not None and file is None:
            if stokes == "Q":
                file = self.Q_cube_name
            else:
                file = self.U_cube_name
        try:
            hdu = fits.open(file, memmap=memmap)
        except FileNotFoundError:
            print("FileNotFoundError: The FITS file cannot be found")
            sys.exit(1)

        print("FITS shape: ", hdu[0].data.squeeze().shape)

        image = hdu[0].data.squeeze()
        header = hdu[0].header
        hdu.close()
        return header, image

    def readQU(self, memmap=True):
        files = [self.Q_cube_name, self.U_cube_name]
        IQU = []
        for file in files:
            header, QU = self.readCube(file=file, memmap=memmap)
            IQU.append(QU)

        return IQU[0], IQU[1], header

    def readImage(self, name=None, stokes=None):
        filename = ""
        if name is None and stokes is not None:
            if stokes == "I":
                filename = self.stokes_I_name
            elif stokes == "Q":
                filename = self.stokes_Q_name
            else:
                filename = self.stokes_U_name
        else:
            filename = name

        hdul = fits.open(name=filename)
        header = hdul[0].header
        data = np.squeeze(hdul[0].data)
        hdul.close()
        return header, data

    def readNumpyFile(self):
        try:
            np_array = np.load(self.numpy_file)
        except FileNotFoundError:
            print("FileNotFoundError: The numpy file cannot be found")
            sys.exit(1)

        Q = np_array[:, :, 0]
        U = np_array[:, :, 1]

        return Q, U

    def readHeader(self, name=None):
        if name is None:
            f_filename = self.Q_cube_name
        else:
            f_filename = name
        hdul_image = fits.open(name=f_filename)
        header = hdul_image[0].header
        hdul_image.close()
        return header

    def getFileNFrequencies(self):
        f_filename = self.freq_file_name
        try:
            with open(f_filename, "r") as f:
                freqs = f.readlines()
                freqs[:] = [freq.rstrip("\n") for freq in freqs]
                freqs[:] = [float(freq) for freq in freqs]
                f.close()
        except:
            print("Cannot open file")
            sys.exit(1)
        freqs = np.array(freqs)
        return freqs

    def readFreqsNumpyFile(self):
        filename = self.freq_file_name
        freqs = np.load(filename)
        return freqs


class Writer:

    def __init__(self, output=""):
        self.output = output

    def writeFITSCube(self, cube, header, nphi, phi, dphi, output=None, overwrite=True):
        header["NAXIS"] = 4
        header["NAXIS3"] = (nphi, "Length of Faraday depth axis")
        header["NAXIS4"] = (2, "Real and imaginary")
        header["CTYPE3"] = "Phi"
        header["CDELT3"] = dphi
        header["CUNIT3"] = "rad/m/m"
        header["CRVAL3"] = phi[0]

        if output is None:
            if cube.dtype == np.complex64 or cube.dtype == np.complex128:
                real_part = da.from_array(cube.real, chunks="auto")
                imag_part = da.from_array(cube.imag, chunks="auto")
                concatenated_cube = da.stack([real_part, imag_part], axis=0)

                fits.writeto(
                    self.output,
                    data=concatenated_cube,
                    header=header,
                    overwrite=overwrite,
                    output_verify="silentfix",
                )

            else:
                fits.writeto(
                    self.output,
                    data=cube,
                    header=header,
                    overwrite=overwrite,
                    output_verify="silentfix",
                )
        else:
            if cube.dtype == np.complex64 or cube.dtype == np.complex128:
                real_part = da.from_array(cube.real, chunks="auto")
                imag_part = da.from_array(cube.imag, chunks="auto")
                concatenated_cube = da.stack([real_part, imag_part], axis=0)

                fits.writeto(
                    output,
                    data=concatenated_cube,
                    header=header,
                    overwrite=overwrite,
                    output_verify="silentfix",
                )

            else:
                fits.writeto(
                    output,
                    data=cube,
                    header=header,
                    overwrite=overwrite,
                    output_verify="silentfix",
                )

    def writeNPCube(self, cube, output=None):
        if output is None:
            np.save(self.output, cube)
        else:
            np.save(output, cube)

        np.save(output, cube)

    def writeFITS(self, data=None, header=None, output=None, overwrite=True):
        hdu = fits.PrimaryHDU(data, header)
        hdul = fits.HDUList([hdu])
        if output is None:
            hdul.writeto(self.output, overwrite=overwrite)
        else:
            hdul.writeto(output, overwrite=overwrite)
