from typing import List, Union

import astropy.units as un
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.units import Quantity
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from astroquery.skyview import SkyView
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, PathPatch, Rectangle
from matplotlib.text import TextPath
from matplotlib.transforms import IdentityTransform

from ..utils.utilities import calculate_noise

SMALL_SIZE = 6
MEDIUM_SIZE = 7
BIGGER_SIZE = 8

SKYVIEW_LIST = list(np.concatenate(list(SkyView.survey_dict.values())).flat)


def add_number(
    ax: plt.Axes = None,
    ra: Quantity = None,
    dec: Quantity = None,
    frame: str = "fk5",
    wcs: WCS = None,
    fp: FontProperties = None,
    text: str = None,
):
    source = SkyCoord(ra=ra, dec=dec, frame=frame)
    source_pix = source.to_pixel(wcs, origin=0)
    source_text = TextPath((source_pix[0] - 45, source_pix[1]), text, prop=fp, size=30)
    source_p1 = PathPatch(source_text, ec="w", lw=1, fc="w", alpha=0.75)
    source_p2 = PathPatch(source_text, ec="none", fc="k")
    ax.add_artist(source_p1)
    ax.add_artist(source_p2)


def normalize_data(data: np.ndarray = None, _min: float = None, _max: float = None):
    if data is not None:
        if _min is None:
            _min = 0.0

        if _max is None:
            _max = 0.0
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        return _min + (data - data_min) / (data_max - data_min) * (_max - _min)
    else:
        raise ValueError("No data to normalize")


def astroquery_search(
    center_coordinate: SkyCoord = None,
    radius: Quantity = None,
    surveys: Union[List[str], str] = None,
    coordinates: str = None,
    scaling: str = None,
    pixels: str = None,
    sampler: str = None,
):
    if radius is None:
        radius = 0.0 * un.deg
    if surveys is not None and center_coordinate is not None:
        images = []
        center_coordinates_string = center_coordinate.to_string("hmsdms")
        if isinstance(surveys, str):
            surveys = surveys.split(",")

        check_surveys = all(item in SKYVIEW_LIST for item in surveys)
        if check_surveys:
            for survey in surveys:
                images.append(
                    SkyView.get_images(
                        position=center_coordinates_string,
                        radius=radius,
                        scaling=scaling,
                        coordinates=coordinates,
                        pixels=pixels,
                        sampler=sampler,
                        survey=survey,
                    )[0][0]
                )

            if len(surveys) == 1:
                return images[0]
            else:
                return images
        else:
            raise ValueError("One of the surveys you are querying is not in SkyView")

    else:
        raise ValueError("Survey List or center coordinate cannot be Nonetype objects")


class Plotter:

    def __init__(
        self,
        optical_image: fits.HDUList = None,
        radio_image: Union[fits.PrimaryHDU, fits.HDUList, str] = None,
        xray_image: Union[fits.HDUList, str] = None,
        additional_image: fits.HDUList = None,
        z: np.float32 = None,
        dist: np.float32 = None,
        scalebar_length_kpc: Quantity = None,
        use_latex: bool = None,
    ):
        self.optical_image = optical_image
        self.radio_image = radio_image
        self.xray_image = xray_image
        self.additional_image = additional_image
        self.use_latex = use_latex
        self.z = z
        self.scale_length_kpc = scalebar_length_kpc
        self.sb_length_arcsec = None

        if self.scale_length_kpc is None:
            self.scale_length_kpc = 500 * un.kpc

        if isinstance(self.radio_image, str):
            self.radio_image = fits.open(self.radio_image)

        if isinstance(self.xray_image, str):
            self.xray_image = fits.open(self.xray_image)

        if dist is None and self.z is not None:
            # self.dist = Planck18.comoving_distance(self.z)  # distance in MPc
            scale = Planck18.arcsec_per_kpc_comoving(self.z)  # scale in arcsec per kpc
            self.sb_length_arcsec = scale * self.scale_length_kpc

    def get_optical(
        self,
        radio_image: Union[fits.Header, fits.PrimaryHDU, fits.HDUList] = None,
        rgb: bool = False,
        filename: str = None,
    ):
        if radio_image is None:
            radio_image = self.radio_image

        if isinstance(radio_image, fits.HDUList):
            header = radio_image[0].header
        elif isinstance(radio_image, fits.PrimaryHDU):
            header = radio_image.header
        elif isinstance(radio_image, fits.Header):
            header = radio_image
        else:
            raise TypeError("Radio image is not a header, PrimaryHDU or HDUList")

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
            hdu_g, hdu_r, hdu_i = astroquery_search(
                center_coordinate=coord,
                radius=radius,
                pixels=pixels,
                coordinates="J2000",
                sampler="LI",
                scaling="Log",
                surveys=surveys,
            )
            data = make_lupton_rgb(hdu_i.data, hdu_r.data, hdu_g.data, filename=filename)
            hdu_i.data = data
            self.optical_image = hdu_i
        else:
            self.optical_image = astroquery_search(
                center_coordinate=coord,
                radius=radius,
                pixels=pixels,
                coordinates="J2000",
                sampler="LI",
                scaling="Log",
                surveys="DSS",
            )

    def get_additional(
        self,
        radio_image: Union[fits.Header, fits.PrimaryHDU, fits.HDUList] = None,
        surveys: Union[List[str], str] = None,
    ):
        if radio_image is None:
            radio_image = self.radio_image
        if isinstance(radio_image, fits.HDUList):
            header = radio_image[0].header
        elif isinstance(radio_image, fits.PrimaryHDU):
            header = radio_image.header
        elif isinstance(radio_image, fits.Header):
            header = radio_image
        else:
            raise TypeError("Radio image is not a header, PrimaryHDU or HDUList")

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

        self.additional_image = astroquery_search(
            center_coordinate=coord,
            radius=radius,
            pixels=pixels,
            coordinates="J2000",
            surveys=surveys,
        )

    def plot_overlay(
        self,
        dpi: np.int32 = 600,
        xlabel: str = None,
        ylabel: str = None,
        sigma_list: list = None,
        savefig: bool = False,
        save_path: str = "./plot_overlay",
        file_format: str = "pdf",
    ):
        fig = plt.figure(dpi=dpi)
        if self.use_latex:
            plt.rcParams.update(
                {
                    "font.family": "serif",
                    "text.usetex": True,
                    "pgf.rcfonts": False,
                    "pgf.texsystem": "pdflatex",  # default is xetex
                }
            )

        plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize

        # get colour maps and set bad values to be transparent
        bluecmap = plt.cm.get_cmap("Blues").copy()
        bluecmap.set_bad("white", 0)
        redcmap = plt.cm.get_cmap("Reds").copy()
        redcmap.set_bad("white", 0)
        yellowcmap = plt.cm.get_cmap("YlOrBr_r").copy()
        yellowcmap.set_bad("white", 0)
        purplecmap = plt.cm.get_cmap("Purples").copy()
        purplecmap.set_bad("white", 0)
        greyscale = plt.cm.get_cmap("gray_r").copy()
        greyscale.set_bad("white", 0)

        # perceptively uniform cmap that will be used for contours
        twilightcmap = plt.cm.get_cmap("twilight")

        # contour levels
        # the step parameter is the factor of 2^step each contour goes up by
        # so use step=1 for contours which double each time
        contourexps = np.arange(start=0, stop=32, step=0.5)
        contourmults = np.power(2, contourexps)

        if sigma_list is None:
            raise ValueError("Cannot draw contours without a sigma_list")

        # Plot for optical data
        if self.radio_image is not None:
            if isinstance(self.radio_image, fits.HDUList):
                header = self.radio_image[0].header
                radio_data = self.radio_image[0].data.squeeze()
            elif isinstance(self.radio_image, fits.PrimaryHDU):
                header = self.radio_image.header
                radio_data = self.radio_image.data.squeeze()

            wcs = WCS(header, naxis=2)
            ax = plt.subplot(projection=wcs)

        if self.optical_image is not None:
            wcs_optical = WCS(self.optical_image.header, naxis=2)
            ax.imshow(
                self.optical_image.data,
                transform=ax.get_transform(wcs_optical),
                origin="lower",
                cmap=greyscale,
            )

        # Contours for X-ray data
        if self.xray_image is not None:
            wcs_xray = WCS(self.xray_image[0].header, naxis=2)
            xray_image_sigma = sigma_list[0] * calculate_noise(
                self.xray_image[0].data, use_sigma_clipped_stats=True
            )
            purple_contours = [xray_image_sigma * i for i in contourmults]
            purple_data = self.xray_image[0].data
            purple_norm = normalize_data(purple_data, 0, 1)
            purple_alphas = np.where(purple_data >= xray_image_sigma, purple_norm, 0.0)
            purple_color = twilightcmap(0.4)
            ax.contour(
                purple_data,
                transform=ax.get_transform(wcs_xray),
                colors=[purple_color],
                levels=purple_contours,
                linewidths=0.2,
            )
            ax.imshow(
                purple_data,
                origin="lower",
                transform=ax.get_transform(wcs_xray),
                cmap=purplecmap,
                alpha=purple_alphas,
                vmax=np.nanmax(purple_data) / 5000,
            )
            ax.plot(0, 0, "-", c=purple_color, label="XMM", linewidth=3)

            # Contours for additional data
        if self.additional_image is not None:
            wcs_additional = WCS(self.additional_image.header, naxis=2)
            add_image_sigma = sigma_list[1] * calculate_noise(
                self.additional_image.data, use_sigma_clipped_stats=True
            )
            blue_contours = [add_image_sigma * i for i in contourmults]
            blue_data = self.additional_image.data
            blue_norm = normalize_data(blue_data, 0, 1)
            blue_color = twilightcmap(0.2)
            blue_alphas = np.where(blue_data >= add_image_sigma, blue_norm, 0.0)
            ax.contour(
                blue_data,
                transform=ax.get_transform(wcs_additional),
                colors=[blue_color],
                levels=blue_contours,
                linewidths=0.2,
            )
            ax3 = plt.imshow(
                blue_data,
                origin="lower",
                transform=ax.get_transform(wcs_additional),
                cmap=bluecmap,
                alpha=blue_alphas,
            )
            ax.plot(0, 0, "-", c=blue_color, label="WISE", linewidth=3)

        # Contours for radio continuum
        if self.radio_image is not None:
            radio_image_sigma = sigma_list[2] * calculate_noise(
                radio_data.squeeze(), use_sigma_clipped_stats=True
            )
            red_contours = [radio_image_sigma * i for i in contourmults]
            red_data = radio_data.squeeze()
            red_norm = normalize_data(red_data, 0, 1)
            red_color = twilightcmap(0.75)
            red_alphas = np.where(red_data >= radio_image_sigma, red_norm, 0.0)
            ax.contour(
                red_data,
                transform=ax.get_transform(wcs),
                colors=[red_color],
                levels=red_contours,
                linewidths=0.2,
            )
            ax4 = plt.imshow(
                red_data,
                origin="lower",
                transform=ax.get_transform(wcs),
                vmax=np.nanmax(red_data) / 1000,
                cmap=redcmap,
                alpha=red_alphas,
            )
            ax.plot(0, 0, "-", c=red_color, label="JVLA", linewidth=3)

        # if self.z is not None or self.dist is not None:
        if xlabel is not None:
            ax.coords[0].set_axislabel(xlabel)
        if ylabel is not None:
            ax.coords[1].set_axislabel(ylabel)
        ax.coords[0].set_format_unit(un.deg)
        ax.coords[1].set_format_unit(un.deg)
        fp = FontProperties(size="xx-large", weight="extra bold")
        x_ray_center = SkyCoord(ra=173.714 * un.deg, dec=49.091 * un.deg, frame="fk5")
        # print(x_ray_center.to_string('hmsdms'))
        x_center, y_center = x_ray_center.to_pixel(wcs, origin=0)
        ax.scatter(x=x_center, y=y_center, marker="x", color="gold", s=35)

        add_number(
            ax=ax,
            ra=173.453 * un.deg,
            dec=48.985 * un.deg,
            frame="fk5",
            text="1",
            fp=fp,
            wcs=wcs,
        )
        add_number(
            ax=ax,
            ra=173.496 * un.deg,
            dec=49.045 * un.deg,
            frame="fk5",
            text="2",
            fp=fp,
            wcs=wcs,
        )
        add_number(
            ax=ax,
            ra=173.496 * un.deg,
            dec=49.062 * un.deg,
            frame="fk5",
            text="3",
            fp=fp,
            wcs=wcs,
        )
        add_number(
            ax=ax,
            ra=173.498 * un.deg,
            dec=48.941 * un.deg,
            frame="fk5",
            text="4",
            fp=fp,
            wcs=wcs,
        )
        add_number(
            ax=ax,
            ra=173.521 * un.deg,
            dec=49.106 * un.deg,
            frame="fk5",
            text="5",
            fp=fp,
            wcs=wcs,
        )
        add_number(
            ax=ax,
            ra=173.571 * un.deg,
            dec=49.152 * un.deg,
            frame="fk5",
            text="6",
            fp=fp,
            wcs=wcs,
        )
        add_number(
            ax=ax,
            ra=173.621 * un.deg,
            dec=48.951 * un.deg,
            frame="fk5",
            text="7",
            fp=fp,
            wcs=wcs,
        )
        add_number(
            ax=ax,
            ra=173.632 * un.deg,
            dec=49.049 * un.deg,
            frame="fk5",
            text="8",
            fp=fp,
            wcs=wcs,
        )
        add_number(
            ax=ax,
            ra=173.693 * un.deg,
            dec=48.956 * un.deg,
            frame="fk5",
            text="9",
            fp=fp,
            wcs=wcs,
        )
        add_number(
            ax=ax,
            ra=173.705 * un.deg,
            dec=49.077 * un.deg,
            frame="fk5",
            text="10",
            fp=fp,
            wcs=wcs,
        )
        add_number(
            ax=ax,
            ra=173.712 * un.deg,
            dec=49.203 * un.deg,
            frame="fk5",
            text="11",
            fp=fp,
            wcs=wcs,
        )
        add_number(
            ax=ax,
            ra=173.762 * un.deg,
            dec=49.193 * un.deg,
            frame="fk5",
            text="12",
            fp=fp,
            wcs=wcs,
        )
        add_number(
            ax=ax,
            ra=173.803 * un.deg,
            dec=48.966 * un.deg,
            frame="fk5",
            text="13",
            fp=fp,
            wcs=wcs,
        )
        add_number(
            ax=ax,
            ra=173.933 * un.deg,
            dec=49.038 * un.deg,
            frame="fk5",
            text="14",
            fp=fp,
            wcs=wcs,
        )
        add_number(
            ax=ax,
            ra=173.943 * un.deg,
            dec=48.921 * un.deg,
            frame="fk5",
            text="15",
            fp=fp,
            wcs=wcs,
        )

        if self.radio_image is not None:
            dx = -header["CDELT1"] * un.deg
            bmaj = 0.003435974915822347 * un.deg
            bmin = 0.003251787026723226 * un.deg
            bpa = 49.36214065551758 * un.deg
            bmaj_pix = (bmaj / dx).value
            bmin_pix = (bmaj / dx).value

            ellipse = Ellipse(
                xy=(20, 20),
                width=bmaj_pix,
                height=bmin_pix,
                angle=(bpa + 90 * un.deg).value,
                edgecolor="crimson",
                fc="None",
                lw=1,
            )
            ax.add_patch(ellipse)

        if self.radio_image is not None and self.sb_length_arcsec is not None:
            sb_length_pix = self.sb_length_arcsec / dx.to(un.arcsec)
            scalebar = Rectangle(
                xy=(header["NAXIS1"] - 1 - sb_length_pix - 10, 10),
                width=sb_length_pix,
                height=4,
                edgecolor="none",
                fc="black",
                alpha=1,
            )
            ax.add_patch(scalebar)
            scaletext = f"{self.scale_length_kpc:0.0f}"
            plt.annotate(
                xy=(header["NAXIS1"] - 1 - (sb_length_pix / 2.0) - 15, 25),
                text=scaletext,
                c="black",
                ha="center",
            )

        plt.gca().set_aspect("equal", adjustable="box")
        plt.legend(loc=2)
        fig.canvas.draw()
        fig.tight_layout()
        plt.show()

        if savefig:
            fig.savefig(save_path, bbox_inches="tight", format=file_format, dpi=dpi)
