import cupy
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
from typing import Union, List
import astropy.units as un
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView
from astropy.units import Quantity
from astropy.visualization import make_lupton_rgb
import numpy as np


def astroquery_search(center_coordinate: SkyCoord = None, radius: Quantity = None,
                      surveys: Union[List[str], str] = None, coordinates: str = None,
                      scaling: str = None, pixels: str = None, sampler: str = None):
    if radius is None:
        radius = 0.0 * un.deg
    if surveys is not None and center_coordinate is not None:
        images = []
        center_coordinates_string = center_coordinate.to_string('hmsdms')
        if isinstance(surveys, str):
            surveys = surveys.split(",")
        for survey in surveys:
            images.append(SkyView.get_images(position=center_coordinates_string, radius=radius, scaling=scaling,
                                             coordinates=coordinates, pixels=pixels, sampler=sampler, survey=survey))

        if len(surveys) > 1:
            return images[0]
        else:
            return tuple(images)

    else:
        raise ValueError("Survey List or center coordinate cannot be Nonetype objects")


class Plotter:
    def __init__(self, optical_image: fits.HDUList = None,
                 radio_image: fits.HDUList = None,
                 additional_image: fits.HDUList = None, use_latex: bool = None):
        self.optical_image = optical_image
        self.radio_image = radio_image
        self.additional_image = additional_image
        self.use_latex = use_latex

    def get_optical(self, radio_image: fits.HDUList = None, rgb: bool = False, filename: str = None):
        if radio_image is None:
            radio_image = self.radio_image
        if radio_image is not None and self.optical_image is None:
            header = radio_image[0].header
            m = header["NAXIS1"]
            n = header["NAXIS2"]
            cdelt1 = header["CDELT1"] * un.deg
            cdelt2 = header["CDELT2"] * un.deg
            crpix1 = header["CRPIX1"]
            crpix2 = header["CRPIX2"]
            crval1 = header["CRVAL1"] * un.deg
            crval2 = header["CRVAL2"]
            radesys = header["RADESYS"]
            pixels = str(m) + "," + str(n)
            coord = SkyCoord(crval1, crval2, frame=radesys)
            radius = 2 * (m - crpix1) * -cdelt1

            if rgb:
                surveys = ["SDSSg", "SDSSr", "SDSSi"]
                hdu_g, hdu_r, hdu_i = astroquery_search(center_coordinate=coord, radius=radius, pixels=pixels,
                                                        coordinates="J2000", sampler="LI", scaling="Log",
                                                        surveys=surveys)
                self.optical_image = make_lupton_rgb(hdu_i.data, hdu_r.data, hdu_g.data, filename=filename)
            else:
                self.optical_image = astroquery_search(center_coordinate=coord, radius=radius, pixels=pixels,
                                                       coordinates="J2000", sampler="LI", scaling="Log",
                                                       surveys="DSS")

    def get_additional(self, radio_image: fits.HDUList = None, surveys: Union[List[str], str] = None):
        if radio_image is None:
            radio_image = self.radio_image
        if radio_image is not None and self.optical_image is None:
            header = radio_image[0].header
            m = header["NAXIS1"]
            n = header["NAXIS2"]
            cdelt1 = header["CDELT1"] * un.deg
            cdelt2 = header["CDELT2"] * un.deg
            crpix1 = header["CRPIX1"]
            crpix2 = header["CRPIX2"]
            crval1 = header["CRVAL1"] * un.deg
            crval2 = header["CRVAL2"]
            radesys = header["RADESYS"]
            pixels = str(m) + "," + str(n)
            coord = SkyCoord(crval1, crval2, frame=radesys)
            radius = 2 * (m - crpix1) * -cdelt1

            self.additional_image = astroquery_search(center_coordinate=coord, radius=radius, pixels=pixels,
                                                      coordinates="J2000", surveys=surveys)

    def plot_overlay(self):

