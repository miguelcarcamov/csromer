from typing import List, Union

import astropy.units as un
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.units import Quantity
from astropy.wcs import WCS
from mpl_toolkits import axes_grid1
from regions import CirclePixelRegion, CircleSkyRegion, PixCoord

from ..utils.utilities import calculate_noise

SMALL_SIZE = 14
MEDIUM_SIZE = 15
BIGGER_SIZE = 16


def create_circular_skyregion(ra, dec, radius, radius_unit="arcsec", unit="deg"):
    center = SkyCoord(ra, dec, unit=unit)
    radius = Angle(radius, radius_unit)
    region = CircleSkyRegion(center, radius)
    return region


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad, axes_class=maxes.Axes)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


class RMPlotter:

    def __init__(
        self,
        rm_image: Union[fits.PrimaryHDU, fits.HDUList, str] = None,
        rm_image_error: Union[fits.PrimaryHDU, fits.HDUList, str] = None,
        pol_fraction_image: Union[fits.PrimaryHDU, fits.HDUList, str] = None,
        total_intensity_image: Union[fits.PrimaryHDU, fits.HDUList, str] = None,
        center_coord: Union[List[float], Quantity, SkyCoord] = None,
        radius: Union[List[float], Quantity] = 0.0,
        total_intensity_nsigma: float = None,
        use_latex: bool = False,
    ):
        self.rm_image = rm_image
        self.total_intensity_image = total_intensity_image
        self.pol_fraction_image = pol_fraction_image
        self.rm_image_error = rm_image_error
        self.center_coord = center_coord
        self.radius = radius
        self.total_intensity_nsigma = total_intensity_nsigma
        self.use_latex = use_latex
        if self.total_intensity_nsigma is None:
            self.total_intensity_nsigma = 3.0

        if isinstance(self.rm_image, str):
            self.rm_image = fits.open(self.rm_image)

        if isinstance(self.total_intensity_image, str):
            self.total_intensity_image = fits.open(self.total_intensity_image)

        if self.rm_image_error is not None:
            if isinstance(self.rm_image_error, str):
                self.rm_image_error = fits.open(self.rm_image_error)

        if self.pol_fraction_image is not None:
            if isinstance(self.pol_fraction_image, str):
                self.pol_fraction_image = fits.open(self.pol_fraction_image)

        if isinstance(self.center_coord, Quantity):
            if self.center_coord.isscalar:
                raise ValueError("Coordinate cannot be an scalar")
            else:
                self.center_coord = SkyCoord(ra=self.center_coord[0], dec=self.center_coord[1])

        if isinstance(self.radius, Quantity):
            if not self.radius.isscalar:
                raise ValueError("Radius need to be an scalar")

    def get_lims(self, wcs: WCS = None):
        if isinstance(self.radius, Quantity):
            if self.radius.isscalar:
                radius_x = radius_y = self.radius
            else:
                radius_x = self.radius[0]
                radius_y = self.radius[1]

        else:
            if isinstance(self.radius, list):
                radius_x = self.radius[0] * un.deg
                radius_y = self.radius[1] * un.deg
            else:
                radius_x = radius_y = self.radius * un.deg

        left_corner = SkyCoord(
            ra=self.center_coord.ra - radius_x, dec=self.center_coord.dec - radius_y
        )
        right_corner = SkyCoord(
            ra=self.center_coord.ra + radius_x, dec=self.center_coord.dec + radius_y
        )
        left, up = left_corner.to_pixel(wcs, origin=0)
        right, down = right_corner.to_pixel(wcs, origin=0)
        xlim = [int(right), int(left)]
        ylim = [int(up), int(down)]
        return xlim, ylim

    def plot(
        self,
        dpi=600,
        colorbar_label: Union[str, List[str]] = "",
        savefig: bool = False,
        save_path: str = "./plot_rm.pdf",
        file_format: str = "pdf",
    ):
        fig = plt.figure(figsize=(20, 10), dpi=dpi)
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
        plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize

        # Set colormaps
        plasmacmap = plt.cm.get_cmap("plasma").copy()
        # plasmacmap.set_bad('grey', 0.5)
        rmcmap = plt.cm.get_cmap("seismic").copy()
        # rmcmap.set_bad('gray', 0.5)
        magmacmap = plt.cm.get_cmap("magma_r").copy()
        # magmacmap.set_bad('gray', 0.5)
        inferno = plt.cm.get_cmap("inferno_r").copy()
        # inferno.set_bad('gray', 0.5)
        viridis = plt.cm.get_cmap("viridis_r").copy()
        # viridis.set_bad('gray', 0.2)
        twilight = plt.cm.get_cmap("twilight").copy()
        # twilight.set_bad('gray', 0.5)
        cividis = plt.cm.get_cmap("cividis").copy()
        # cividis.set_bad('gray', 0.5)

        # Get headers and data
        if isinstance(self.total_intensity_image, fits.HDUList):
            total_intensity_header = self.total_intensity_image[0].header
            total_intensity_data = self.total_intensity_image[0].data.squeeze()
        else:
            total_intensity_header = self.total_intensity_image.header
            total_intensity_data = self.total_intensity_image.data.squeeze()

        if isinstance(self.rm_image, fits.HDUList):
            rm_image_header = self.rm_image[0].header
            rm_image_data = self.rm_image[0].data
        else:
            rm_image_header = self.rm_image.header
            rm_image_data = self.rm_image.data

        if self.rm_image_error is not None:
            if isinstance(self.rm_image_error, fits.HDUList):
                rm_image_error_header = self.rm_image_error[0].header
                rm_image_error_data = self.rm_image_error[0].data
            else:
                rm_image_error_header = self.rm_image_error.header
                rm_image_error_data = self.rm_image_error.data

        if self.pol_fraction_image is not None:
            if isinstance(self.pol_fraction_image, fits.HDUList):
                pol_fraction_header = self.pol_fraction_image[0].header
                pol_fraction_data = self.pol_fraction_image[0].data
            else:
                pol_fraction_header = self.pol_fraction_image.header
                pol_fraction_data = self.pol_fraction_image.data

        wcs_total_intensity = WCS(total_intensity_header, naxis=2)
        wcs_rm_image: WCS = WCS(rm_image_header, naxis=2)

        if self.rm_image_error is not None:
            wcs_rm_image_error = WCS(rm_image_error_header, naxis=2)

        if self.pol_fraction_image is not None:
            wcs_pol_fraction = WCS(pol_fraction_header, naxis=2)

        # Get noise of total intensity image
        total_intensity_noise = self.total_intensity_nsigma * calculate_noise(
            total_intensity_data, use_sigma_clipped_stats=True
        )
        # contour levels
        # the step parameter is the factor of 2^step each contour goes up by
        # so use step=1 for contours which double each time
        contourexps = np.arange(start=0, stop=32, step=1)
        contourmults = np.power(2, contourexps)

        I_contours = [total_intensity_noise * i for i in contourmults]

        core_region = create_circular_skyregion(173.4962263, 49.0629333,
                                                17.782).to_pixel(wcs_rm_image)
        north_region = create_circular_skyregion(173.4776803, 49.0761733,
                                                 41.295).to_pixel(wcs_rm_image)
        south_region = create_circular_skyregion(173.4719210, 49.0538024,
                                                 42.187).to_pixel(wcs_rm_image)

        if self.rm_image_error is not None and self.pol_fraction_image is not None:
            ax1 = fig.add_subplot(1, 3, 1, projection=wcs_rm_image, box_aspect=1)
            c1 = ax1.imshow(
                rm_image_data,
                origin="lower",
                cmap=inferno,
                vmin=np.nanmin(rm_image_data),
                vmax=np.nanmax(rm_image_data)
            )
            # ax1.text(x=173.4776803, y=49.0761733, s="N")
            # ax1.text(x=173.4719210, y=49.0538024, ha="center", s="S", transform=ax1.get_transform(wcs_rm_image))
            ax1.contour(
                total_intensity_data,
                transform=ax1.get_transform(wcs_total_intensity),
                levels=I_contours,
                colors="black",
                alpha=0.6,
                linewidths=0.1,
            )
            core_region.plot(ax=ax1, color="gold", lw=0.8)
            north_region.plot(ax=ax1, color="royalblue", lw=0.8)
            south_region.plot(ax=ax1, color="darkcyan", lw=0.8)

            ax2 = fig.add_subplot(1, 3, 2, projection=wcs_rm_image_error, box_aspect=1)
            c2 = ax2.imshow(rm_image_error_data, origin="lower", cmap=inferno, vmin=0, vmax=3)
            ax2.contour(
                total_intensity_data,
                transform=ax2.get_transform(wcs_total_intensity),
                levels=I_contours,
                colors="black",
                alpha=0.6,
                linewidths=0.1,
            )

            ax3 = fig.add_subplot(1, 3, 3, projection=wcs_pol_fraction, box_aspect=1)
            c3 = ax3.imshow(pol_fraction_data, origin="lower", cmap=inferno, vmin=0, vmax=0.5)
            ax3.contour(
                total_intensity_data,
                transform=ax3.get_transform(wcs_total_intensity),
                levels=I_contours,
                colors="black",
                alpha=0.6,
                linewidths=0.1,
            )

            # ax2.axes.xaxis.set_visible(False)
            # ax3.axes.xaxis.set_visible(False)
            cbar1 = add_colorbar(c1)
            cbar2 = add_colorbar(c2)
            cbar3 = add_colorbar(c3)

            cbar1.ax.set_ylabel(colorbar_label[0])
            cbar2.ax.set_ylabel(colorbar_label[1])
            cbar3.ax.set_ylabel(colorbar_label[2])

            if self.center_coord is not None and self.radius is not None:
                xlim, ylim = self.get_lims(wcs_total_intensity)
                ax1.set_xlim(xlim)
                ax1.set_ylim(ylim)
                ax2.set_xlim(xlim)
                ax2.set_ylim(ylim)
                ax3.set_xlim(xlim)
                ax3.set_ylim(ylim)
            ax1.coords[0].set_axislabel("Right Ascension [J2000]")
            ax1.coords[1].set_axislabel("Declination [J2000]")
            ax1.coords[0].set_ticks(number=3, exclude_overlapping=True)
            ax1.coords[1].set_ticks(number=4, exclude_overlapping=True)
            ax2.coords[0].set_axislabel("Right Ascension [J2000]")
            ax2.coords[1].set_axislabel("")
            ax2.coords[0].set_ticks(number=3, exclude_overlapping=True)
            ax2.coords[1].set_ticks(number=4, exclude_overlapping=True)
            ax3.coords[0].set_axislabel("Right Ascension [J2000]")
            ax3.coords[1].set_axislabel("")
            ax3.coords[0].set_ticks(number=3, exclude_overlapping=True)
            ax3.coords[1].set_ticks(number=4, exclude_overlapping=True)
            ax2.coords[1].set_ticklabel_visible(False)
            ax3.coords[1].set_ticklabel_visible(False)
            ax1.coords[0].set_format_unit(un.deg)
            ax1.coords[1].set_format_unit(un.deg)
            ax2.coords[0].set_format_unit(un.deg)
            ax2.coords[1].set_format_unit(un.deg)
            ax3.coords[0].set_format_unit(un.deg)
            ax3.coords[1].set_format_unit(un.deg)
            ax1.set_facecolor((0.89, 0.89, 0.89))
            ax2.set_facecolor((0.89, 0.89, 0.89))
            ax3.set_facecolor((0.89, 0.89, 0.89))
            # ax1.set_aspect("equal", adjustable='box')
            # ax2.set_aspect("equal", adjustable='box')

        else:
            ax = fig.add_subplot(projection=wcs_rm_image, box_aspect=1)
            # fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': wcs_rm_image}, dpi=dpi)

            c = ax.imshow(rm_image_data, origin="lower", cmap=viridis, vmin=0, vmax=1)
            ax.contour(
                total_intensity_data,
                transform=ax.get_transform(wcs_total_intensity),
                levels=I_contours,
                colors="black",
                alpha=0.6,
                linewidths=0.1,
            )
            # cbar = create_colorbar(c, ax)
            cbar = add_colorbar(c)
            cbar.ax.set_ylabel(colorbar_label)

            if self.center_coord is not None and self.radius is not None:
                xlim, ylim = self.get_lims(wcs_total_intensity)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            ax.coords[0].set_axislabel("Right Ascension [J2000]")
            ax.coords[1].set_axislabel("Declination [J2000]")
            ax.coords[0].set_format_unit(un.deg)
            ax.coords[1].set_format_unit(un.deg)
            ax.coords[0].set_ticks(number=3)
            ax.coords[1].set_ticks(number=3)
            ax.set_aspect("equal", adjustable="box")
            ax.set_facecolor((0.89, 0.89, 0.89))
        fig.canvas.draw()
        fig.tight_layout()
        plt.show()

        if savefig:
            fig.savefig(save_path, bbox_inches="tight", format=file_format, dpi=dpi)
