import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import asarray as ar, exp

cmaps = ['magma', 'inferno', 'inferno_r', 'plasma', 'viridis', 'bone', 'afmhot', 'gist_heat', 'CMRmap', 'gnuplot', 'Blues_r', 'Purples_r', 'ocean', 'hot', 'seismic_r', 'ocean_r']

def config_axes(data, header):
	x = data.shape[0]
	y = data.shape[1]
	dx = header['cdelt1'] * 3600.0
	dy = header['cdelt2'] * 3600.0
	x = ar(range(x)) * dx
	y = ar(range(y)) * dy
	x1 = header['crpix1']
	y1 = header['crpix2']
	x = np.arange(x1, -x1-header['cdelt1'], -header['cdelt1'])
	y = np.arange(-y1, y1+header['cdelt2'], header['cdelt2'])
	return [x, y, x1, y1]

def create_animation(header, cube=np.array([]), xlabel="", ylabel="", cblabel="", title="", title_pad=55.0, output_video="dynamic_images.mp4", fps=30, interval=50, repeat=False):
    ims = []
    num_ims = len(cube)
    if(num_ims != 0):
        fig = plt.figure()
        ax = plt.subplot(111)
        div = make_axes_locatable(ax)
        cax = div.append_axes('top', size='5%', pad=0.1)
        axes = config_axes(cube[0], header)
        cv0 = cube[0]
        im = ax.imshow(cv0, origin='lower', aspect='equal', cmap='ocean_r', extent=[-axes[2], axes[2],-axes[3],axes[3]])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        ax.set_title(title, pad=title_pad)
        cax.set_xlabel(cblabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        tick_locator = ticker.MaxNLocator(nbins=3)
        cb.locator = tick_locator

        def animate(i):
            arr = cube[i]
            vmax     = np.max(arr)
            vmin     = np.min(arr)
            im.set_data(arr)
            im.set_clim(vmin, vmax)

        ani = animation.FuncAnimation(fig, animate, frames=num_ims, interval=50, repeat=repeat, blit=True, repeat_delay=1000)
        ani.save(output_video, fps=fps)
        #plt.show()
