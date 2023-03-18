import cmcrameri.cm as cmc
import matplotlib.animation as animation
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import ticker
from mpl_toolkits import axes_grid1
from scipy import asarray as ar


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad, axes_class=maxes.Axes)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def create_animation(
    header,
    cube_axis=np.array([]),
    cube=np.array([]),
    xlabel="",
    ylabel="",
    cblabel="",
    title_pad=0.0,
    vmin=None,
    vmax=None,
    output_video="dynamic_images.mp4",
    fps=30,
    interval=250,
    repeat=False,
):
    num_ims = len(cube)
    if num_ims != 0:
        if vmin is None and vmax is None:
            vmax = np.nanmax(np.nanmax(cube, axis=0))
            vmin = np.nanmin(np.nanmin(cube, axis=0))

        wcs_cube = WCS(header, naxis=2)
        fig = plt.figure()
        ax = plt.subplot(111, projection=wcs_cube)
        cv0 = cube[0]
        im = ax.imshow(
            cv0,
            origin="lower",
            cmap=cmc.davos,
        )
        tx = ax.set_title("", pad=title_pad)
        # time_text = ax.text(.5, .5, '', fontsize=15)
        ax.coords[0].set_axislabel(xlabel)
        ax.coords[1].set_axislabel(ylabel)

        cbar1 = add_colorbar(im)
        cbar1.ax.set_ylabel(cblabel)

        def animate(i):
            arr = cube[i]
            phi_i = cube_axis[i]
            # vmax     = np.max(arr)
            # vmin     = np.min(arr)
            tx.set_text("Faraday Depth Spectrum at {0:.2f} rad/m^2".format(phi_i))
            # time_text.set_text("Phi: {0}".format(phi_i))
            im.set_data(arr)
            im.set_clim(vmin, vmax)

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=num_ims,
            interval=interval,
            repeat=repeat,
            blit=False,
            repeat_delay=1000,
        )
        ani.save(output_video, fps=fps)
        # plt.show()


def create_animation_from_fits(
    fitsfile: str = None,
    phi_bound: float = None,
    xlabel: str = "",
    ylabel: str = "",
    clabel: str = "",
    output_video="dynamic_images.mp4",
    vmin=None,
    vmax=None,
    fps=30,
    interval=250,
    repeat=False
):
    with fits.open(fitsfile) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    if phi_bound is None:
        phi_bound = 500.
    phi_axis = np.arange(0, header["NAXIS3"]) * header["CDELT3"] + header["CRVAL3"]
    phi_axis_mask = np.abs(phi_axis) <= phi_bound

    data_complex = data[0] + 1j * data[1]
    data_complex = data_complex[phi_axis_mask]
    phi_axis = phi_axis[phi_axis_mask]
    create_animation(
        header,
        cube_axis=phi_axis,
        cube=np.abs(data_complex),
        xlabel=xlabel,
        ylabel=ylabel,
        cblabel=clabel,
        vmin=vmin,
        vmax=vmax,
        output_video=output_video,
        fps=fps,
        interval=interval,
        repeat=repeat
    )
