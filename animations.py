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
	x = ar(range(x)) * header['cdelt1']
	y = ar(range(y)) * header['cdelt2']
	x1 = x[len(x)-1]/2
	y1 = y[len(y)-1]/2
	x = np.arange(x1, -x1-header['cdelt1'], -header['cdelt1'])
	y = np.arange(-y1, y1+header['cdelt2'], header['cdelt2'])
	return [x, y, x1, y1]

def animate(header, cube=np.array([]), xlabel="", ylabel="", cblabel="", title="", title_pad=55.0, output_cube="dynamic_images.mp4", fps=30, interval=50):
    ims = []
    num_ims = len(cube)
    if(num_ims != 0):
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        for i in range(0, num_ims):
            data = cube[i]
            axes = config_axes(data, header)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('top', size='5%', pad=0.1)
            im = ax.imshow(data, aspect='equal', origin='lower', cmap='ocean_r', animated=True, extent=[-axes[2], axes[2],-axes[3],axes[3]])
            cax.cla()
            cb = fig.colorbar(im, cax=cax, orientation='horizontal')
            ax.set_title(title, pad=title_pad)
            cax.set_xlabel(cblabel)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            cb.ax.xaxis.set_label_position('top')
            cb.ax.xaxis.set_ticks_position('top')
            tick_locator = ticker.MaxNLocator(nbins=3)
            cb.locator = tick_locator
            cb.update_ticks()
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=1000)
        ani.save(output_cube, fps=fps)
        #plt.show()
