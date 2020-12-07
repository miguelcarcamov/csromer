import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import asarray as ar, exp

cmaps = ['magma', 'inferno', 'inferno_r', 'plasma', 'viridis', 'bone', 'afmhot', 'gist_heat', 'CMRmap', 'gnuplot', 'Blues_r', 'Purples_r', 'ocean', 'hot', 'seismic_r', 'ocean_r']

def config_axes(data, header, units='degrees'):
	u_factor = 1.0
	if units=='arcmin':
                u_factor = 60.0
	elif units=='arcsec':
                u_factor = 3600.0
	elif units=='rad':
		u_factor = np.pi/180.0
	x = data.shape[0]
	y = data.shape[1]
	dx = header['cdelt1'] * u_factor
	dy = header['cdelt2'] * u_factor
	x = ar(range(x)) * dx
	y = ar(range(y)) * dy
	x1 = header['crpix1']*np.abs(dx)
	y1 = header['crpix2']*dy
	x = np.arange(x1, -x1-dx, -dx)
	y = np.arange(-y1, y1+dy, dy)
	return [x, y, x1, y1]

def colorbar(mappable, title="", location="right"):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(title)
    plt.sca(last_axes)
    return cbar

def create_animation(header, cube=np.array([]), xlabel="", ylabel="", cblabel="", title="", title_pad=55.0, output_video="dynamic_images.mp4", fps=30, interval=50, repeat=False):
    ims = []
    num_ims = len(cube)
    if(num_ims != 0):
        fig = plt.figure()
        ax = plt.subplot(111)
        axes = config_axes(cube[0], header)
        cv0 = cube[0]
        im = ax.imshow(cv0, origin='lower', aspect='equal', cmap='ocean_r', extent=[axes[2],-axes[2],-axes[3],axes[3]])
        cb = colorbar(im, cblabel)
        tx = ax.set_title(title, pad=title_pad)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cb.locator = tick_locator
        cb.update_ticks()

        def animate(i):
            arr = cube[i]
            vmax     = np.max(arr)
            vmin     = np.min(arr)
            im.set_data(arr)
            im.set_clim(vmin, vmax)

        ani = animation.FuncAnimation(fig, animate, frames=num_ims, interval=250, repeat=repeat, blit=False, repeat_delay=1000)
        ani.save(output_video)
        #plt.show()
