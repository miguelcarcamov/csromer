import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import asarray as ar
from scipy import exp

cmaps = [
    "magma",
    "inferno",
    "inferno_r",
    "plasma",
    "viridis",
    "bone",
    "afmhot",
    "gist_heat",
    "CMRmap",
    "gnuplot",
    "Blues_r",
    "Purples_r",
    "ocean",
    "hot",
    "seismic_r",
    "ocean_r",
]


def config_axes(data, header, units="degrees"):
    u_factor = 1.0
    if units == "arcmin":
        u_factor = 60.0
    elif units == "arcsec":
        u_factor = 3600.0
    elif units == "rad":
        u_factor = np.pi / 180.0
    x = data.shape[0]
    y = data.shape[1]
    dx = header["cdelt1"] * u_factor
    dy = header["cdelt2"] * u_factor
    x = ar(range(x)) * dx
    y = ar(range(y)) * dy
    x1 = (header["crpix1"] - 1.0) * np.abs(dx)
    y1 = (header["crpix2"] - 1.0) * dy
    x = np.arange(x1, -x1 - dx, -dx)
    y = np.arange(-y1, y1 + dy, dy)
    return [x, y, x1, y1]


def colorbar(mappable, title="", location="right"):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, extend="both")
    cbar.set_label(title)
    plt.sca(last_axes)
    return cbar


def create_animation(
    header,
    cube_axis=np.array([]),
    cube=np.array([]),
    xlabel="",
    ylabel="",
    cblabel="",
    title="",
    title_pad=0.0,
    vmin=None,
    vmax=None,
    output_video="dynamic_images.mp4",
    fps=30,
    interval=50,
    repeat=False,
):
    ims = []
    num_ims = len(cube)
    if num_ims != 0:
        if vmin is None and vmax is None:
            vmax = np.amax(np.amax(cube, axis=0))
            vmin = np.amin(np.amin(cube, axis=0))

        fig = plt.figure()
        ax = plt.subplot(111)
        axes = config_axes(cube[0], header)
        cv0 = cube[0]
        im = ax.imshow(
            cv0,
            origin="lower",
            aspect="equal",
            cmap="Spectral",
            extent=[axes[2], -axes[2], -axes[3], axes[3]],
        )
        tx = ax.set_title(title, pad=title_pad)
        # time_text = ax.text(.5, .5, '', fontsize=15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cb = colorbar(im, cblabel)
        cb.locator = tick_locator
        cb.update_ticks()

        def animate(i):
            arr = cube[i]
            phi_i = cube_axis[i]
            # vmax     = np.max(arr)
            # vmin     = np.min(arr)
            tx.set_text("Faraday Depth Spectrum at {0:.4f} rad/m^2".format(phi_i))
            # time_text.set_text("Phi: {0}".format(phi_i))
            im.set_data(arr)
            im.set_clim(vmin, vmax)

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=num_ims,
            interval=250,
            repeat=repeat,
            blit=False,
            repeat_delay=1000,
        )
        ani.save(output_video)
        # plt.show()
