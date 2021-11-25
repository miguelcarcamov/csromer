import cupy
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
from astropy.stats import sigma_clip, sigma_clipped_stats
from typing import Union, List
import astropy.units as un
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView
from astropy.units import Quantity
from astropy.visualization import make_lupton_rgb
from astropy.cosmology import Planck18
import numpy as np

SMALL_SIZE = 6
MEDIUM_SIZE = 7
BIGGER_SIZE = 8

SKYVIEW_LIST = list(np.concatenate(list(SkyView.survey_dict.values())).flat)


def normalize_data(data: np.ndarray = None, _min: float = None, _max: float = None):
    if data is not None:
        if _min is None:
            _min = 0.0

        if _max is None:
            _max = 0.0
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        return _min + (data - data_min)/(data_max - data_min) * (_max - _min)
    else:
        raise ValueError("No data to normalize")


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

        check_surveys = all(item in SKYVIEW_LIST for item in surveys)
        if check_surveys:
            for survey in surveys:
                images.append(SkyView.get_images(position=center_coordinates_string, radius=radius, scaling=scaling,
                                                 coordinates=coordinates, pixels=pixels, sampler=sampler,
                                                 survey=survey)[0][
                                  0])

            if len(surveys) == 1:
                return images[0]
            else:
                return images
        else:
            raise ValueError("One of the surveys you are querying is not in SkyView")

    else:
        raise ValueError("Survey List or center coordinate cannot be Nonetype objects")


def get_sigma_from_image(image: np.ndarray = None, nsigma=5.0):
    if image is not None:
        sigma = sigma_clipped_stats(image, sigma=nsigma)[2]
        return sigma
    else:
        raise ValueError("Cannot get error if image is a Nonetype object")


class Plotter:
    def __init__(self, optical_image: fits.HDUList = None,
                 radio_image: Union[fits.HDUList, str] = None,
                 xray_image: Union[fits.HDUList, str] = None,
                 additional_image: fits.HDUList = None, z: np.float32 = None, dist: np.float32 = None,
                 scalebar_length_kpc: Quantity = None, use_latex: bool = None):
        self.optical_image = optical_image
        self.radio_image = radio_image
        self.xray_image = xray_image
        self.additional_image = additional_image
        self.use_latex = use_latex
        self.z = z
        self.scale_length_kpc = scalebar_length_kpc

        if self.scale_length_kpc is None:
            self.scale_length_kpc = 500 * un.kpc

        if isinstance(self.radio_image, str):
            self.radio_image = fits.open(self.radio_image)

        if isinstance(self.xray_image, str):
            self.xray_image = fits.open(self.xray_image)

        if dist is None and self.z is not None:
            self.dist = Planck18.comoving_distance(self.z)  # distance in MPc
            scale = Planck18.arcsec_per_kpc_comoving(self.z)  # scale in arcsec per kpc
            self.sb_length_arcsec = scale * scalebar_length_kpc

    def get_optical(self, radio_image: fits.HDUList = None, rgb: bool = False, filename: str = None):
        if radio_image is None:
            radio_image = self.radio_image
        if radio_image is not None:
            header = radio_image[0].header
            m = header["NAXIS1"]
            n = header["NAXIS2"]
            cdelt1 = header["CDELT1"] * un.deg
            cdelt2 = header["CDELT2"] * un.deg
            crpix1 = header["CRPIX1"]
            crpix2 = header["CRPIX2"]
            crval1 = header["CRVAL1"] * un.deg
            crval2 = header["CRVAL2"] * un.deg
            radesys = header["RADESYS"].lower()
            pixels = str(m) + "," + str(n)
            coord = SkyCoord(crval1, crval2, frame=radesys)
            radius = 2 * (m - crpix1) * -cdelt1

            if rgb:
                surveys = ["SDSSg", "SDSSr", "SDSSi"]
                hdu_g, hdu_r, hdu_i = astroquery_search(center_coordinate=coord, radius=radius, pixels=pixels,
                                                        coordinates="J2000", sampler="LI", scaling="Log",
                                                        surveys=surveys)
                data = make_lupton_rgb(hdu_i.data, hdu_r.data, hdu_g.data, filename=filename)
                hdu_i.data = data
                self.optical_image = hdu_i
            else:
                self.optical_image = astroquery_search(center_coordinate=coord, radius=radius, pixels=pixels,
                                                       coordinates="J2000", sampler="LI", scaling="Log",
                                                       surveys="DSS")

    def get_additional(self, radio_image: fits.HDUList = None, surveys: Union[List[str], str] = None):
        if radio_image is None:
            radio_image = self.radio_image
        if radio_image is not None:
            header = radio_image[0].header
            m = header["NAXIS1"]
            n = header["NAXIS2"]
            cdelt1 = header["CDELT1"] * un.deg
            cdelt2 = header["CDELT2"] * un.deg
            crpix1 = header["CRPIX1"]
            crpix2 = header["CRPIX2"]
            crval1 = header["CRVAL1"] * un.deg
            crval2 = header["CRVAL2"] * un.deg
            radesys = header["RADESYS"].lower()
            pixels = str(m) + "," + str(n)
            coord = SkyCoord(crval1, crval2, frame=radesys)
            radius = 2 * (m - crpix1) * -cdelt1

            self.additional_image = astroquery_search(center_coordinate=coord, radius=radius, pixels=pixels,
                                                      coordinates="J2000", surveys=surveys)

    def plot_overlay(self, dpi: np.int32 = 600, xlabel: str = None, ylabel: str = None, sigma_list: list = None):
        plt.figure(dpi=dpi)
        if self.use_latex:
            plt.rcParams['mathtext.fontset'] = 'cm'
            plt.rcParams['font.family'] = 'cmu serif'

        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize

        # get colour maps and set bad values to be transparent
        bluecmap = plt.cm.Blues
        bluecmap.set_bad('white', 0)
        redcmap = plt.cm.Reds
        redcmap.set_bad('white', 0)
        yellowcmap = plt.cm.YlOrBr_r
        yellowcmap.set_bad('white', 0)
        purplecmap = plt.cm.Purples
        purplecmap.set_bad('white', 0)
        greyscale = plt.cm.gray_r
        greyscale.set_bad('white', 0)

        # perceptively uniform cmap that will be used for contours
        twilightcmap = plt.cm.get_cmap('twilight')

        # contour levels
        # the step parameter is the factor of 2^step each contour goes up by
        # so use step=1 for contours which double each time
        contourexps = np.arange(start=0, stop=32, step=0.5)
        contourmults = np.power(2, contourexps)

        if sigma_list is None:
            raise ValueError("Cannot draw contours without a sigma_list")

        # Plot for optical data
        if self.radio_image is not None:
            wcs = WCS(self.radio_image[0].header, naxis=2)
            ax = plt.subplot(projection=wcs)

        if self.optical_image is not None:
            wcs_optical = WCS(self.optical_image.header, naxis=2)
            ax.imshow(self.optical_image.data, transform=ax.get_transform(wcs_optical), origin='lower',
                      cmap=greyscale)

        # Contours for X-ray data
        if self.xray_image is not None:
            wcs_xray = WCS(self.xray_image[0].header)
            xray_image_sigma = sigma_list[0] * get_sigma_from_image(self.xray_image[0].data)
            purple_contours = [xray_image_sigma * i for i in contourmults]
            purple_data = self.xray_image[0].data
            purple_norm = normalize_data(purple_data, 0, 1)
            purple_alphas = np.where(purple_data >= xray_image_sigma, purple_norm, 0.0)
            purple_color = twilightcmap(0.4)
            ax.contour(purple_data, transform=ax.get_transform(wcs_xray), colors=[purple_color],
                       levels=purple_contours, linewidths=0.2)
            ax.imshow(purple_data, origin='lower', transform=ax.get_transform(wcs_xray),
                      cmap=purplecmap, alpha=purple_alphas, vmax=np.nanmax(purple_data)/5000)
            ax.plot(0, 0, '-', c=purple_color, label="XMM", linewidth=3)

            # Contours for additional data
        if self.additional_image is not None:
            wcs_additional = WCS(self.additional_image.header)
            add_image_sigma = sigma_list[1] * get_sigma_from_image(self.additional_image.data)
            blue_contours = [add_image_sigma * i for i in contourmults]
            blue_data = self.additional_image.data
            blue_norm = normalize_data(blue_data, 0, 1)
            blue_color = twilightcmap(0.2)
            blue_alphas = np.where(blue_data >= add_image_sigma, blue_norm, 0.0)
            ax.contour(blue_data, transform=ax.get_transform(wcs_additional),
                       colors=[blue_color], levels=blue_contours, linewidths=0.2)
            ax3 = plt.imshow(blue_data, origin='lower',
                             transform=ax.get_transform(wcs_additional),
                             cmap=bluecmap, alpha=blue_alphas)
            ax.plot(0, 0, '-', c=blue_color, label="WISE", linewidth=3)

        # Contours for radio continuum
        if self.radio_image is not None:
            radio_image_sigma = sigma_list[2] * get_sigma_from_image(self.radio_image[0].data.squeeze())
            red_contours = [radio_image_sigma * i for i in contourmults]
            red_data = self.radio_image[0].data.squeeze()
            red_norm = normalize_data(red_data, 0, 1)
            red_color = twilightcmap(0.75)
            red_alphas = np.where(red_data >= radio_image_sigma, red_norm, 0.0)
            ax.contour(red_data, transform=ax.get_transform(wcs), colors=[red_color],
                       levels=red_contours, linewidths=0.2)
            ax4 = plt.imshow(red_data, origin='lower', transform=ax.get_transform(wcs),
                             vmax=np.nanmax(red_data)/1000,
                             cmap=redcmap, alpha=red_alphas)
            ax.plot(0, 0, '-', c=red_color, label="JVLA", linewidth=3)

        # if self.z is not None or self.dist is not None:

        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.gca().set_aspect("equal")
        plt.legend(loc=2)
        plt.show()
