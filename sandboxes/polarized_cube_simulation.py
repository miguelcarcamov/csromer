#!/usr/bin/env python
# coding: utf-8

# In[1]:

import copy
import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.stats as st
from astropy.io import fits
from scipy.constants import c

# In[2]:


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / np.max(kern2d)


# In[3]:


def calculate_l2ref(lambda2, w):
    k = np.sum(w)
    return np.sum(w * lambda2) / k


# In[4]:


def thinsource_simulation(lambda2, l2_ref, nu_0, spectral_idx, phi_gal, s_nu):
    nu = c / np.sqrt(lambda2)
    k = (nu / nu_0)**(-1.0 * spectral_idx)
    mu_q = np.cos(2.0 * phi_gal * (lambda2 - l2_ref))
    mu_u = np.sin(2.0 * phi_gal * (lambda2 - l2_ref))

    # p = np.mean(np.sqrt(mu_q ** 2 + mu_u ** 2))

    return s_nu * k * (mu_q + 1j * mu_u)


# In[5]:


def thicksource_simulation(lambda2, l2_ref, nu_0, spectral_idx, phi_fg, phi_center, s_nu):
    nu = c / np.sqrt(lambda2)
    k = (nu / nu_0)**(-1.0 * spectral_idx)
    phi_fg = phi_fg / 2.0
    j = 2.0 * (lambda2 - l2_ref) * (phi_center + phi_fg)
    k = 2.0 * (lambda2 - l2_ref) * (phi_center - phi_fg)
    mu_q = np.sin(j) - np.sin(k)
    const = s_nu * k / (2.0 * phi_fg * (lambda2 - l2_ref))
    mu_u = np.cos(j) - np.cos(k)

    return const * (mu_q + mu_u / 1j)


# In[6]:


def remove_channels(lambda2=None, remove_frac=None, random_state=None, chunksize=None):

    remove_frac = 1.0 - remove_frac
    m = len(lambda2)
    if chunksize is None:
        chunksize = m // (10 + (10 * (1.0 - remove_frac)))

    _chansremoved = []
    while True:
        if random_state is None:
            pos = np.random.randint(0, m)  # get position
            width = np.random.uniform(0, chunksize)  # get chunk size
        else:
            pos = random_state.randint(0, m)  # get position
            width = random_state.uniform(0, chunksize)  # get chunk size
        low = int(pos - 0.5 * width)
        if low < 0:
            low = 0
        high = int(pos + 0.5 * width)
        if high >= m:
            high = m - 1

        _chansremoved.append(list(np.arange(low, high)))
        merged = list(itertools.chain(*_chansremoved))

        chans_removed = np.unique(np.ravel(np.array(merged)))
        frac = float(len(chans_removed)) / float(m)
        if frac > remove_frac:
            break

    # adjust back towards specified fraction
    # using single channel adjustments:
    while True:
        if random_state is None:
            idx = np.random.randint(0, len(chans_removed))
        else:
            idx = random_state.randint(0, len(chans_removed))
        chans_removed = np.delete(chans_removed, idx)
        frac = float(len(chans_removed)) / float(m)
        if frac <= remove_frac:
            break

    return chans_removed


# In[7]:


def s_nu_at_freq(s_nu_0, nu, nu_0, alpha):
    s_nu = s_nu_0 * (nu / nu_0)**alpha


# In[8]:


def pol_dist(s_nu):
    a = np.random.normal(loc=0.051, scale=0.004, size=s_nu.shape).astype(np.float32)
    b = np.random.normal(loc=0.388, scale=0.007, size=s_nu.shape).astype(np.float32)
    log_pol_fraction = -a * np.log(s_nu * 1000.0) + b
    pol_fraction = np.exp(log_pol_fraction)
    return pol_fraction * s_nu


# In[9]:

np.random.seed(666)

# In[10]:

antenna_diameter = 13.5  # meters
min_baseline = 29  # meters
max_baseline = 7700  # meters

# In[11]:

# Image cube
m = 256
n = 256
nu_size = 1000
flux_max = 0.0668622  # Jy/beam at 1.27962918 GHz

noise_I = 3.23964e-06  # Jy/beam at 1.27962918 GHz
noise_Q = 2.2804e-06  # Jy/beam at 1.27962918 GHz
noise_U = 2.38714e-06  # Jy/beam at 1.27962918 GHz

flux_min = 5.0 * noise_I  # Jy/beam at 1.27962918 GHz

q_noise = np.random.normal(loc=0.0, scale=noise_Q, size=(nu_size, m, n)).astype(np.float32)
u_noise = np.random.normal(loc=0.0, scale=noise_U, size=(nu_size, m, n)).astype(np.float32)
cube = q_noise + 1j * u_noise

# In[12]:

nu_start = 0.9e9
nu_end = 1.67e9

# In[13]:

pix_size = -4.166666666667e-04

# In[14]:

nu = np.linspace(start=nu_start, stop=nu_end, num=nu_size, dtype=np.float32)
nu_0 = np.median(nu)

# In[15]:

lambda_sq = (c / nu)**2
# lambda_sq = lambda_sq[::-1]

# In[16]:

q_noise.shape

# In[17]:

sigma = 0.5 * (np.std(q_noise, axis=(1, 2)) + np.std(u_noise, axis=(1, 2)))
weights = 1.0 / sigma**2

# In[18]:

l2_ref = calculate_l2ref(lambda_sq, weights)

# In[19]:

n_random_pixels = 100
random_pixels = np.random.choice(m * n, size=n_random_pixels, replace=False).astype(np.int32)
random_pixels = np.unravel_index(random_pixels, (m, n))
random_source = np.random.randint(low=0, high=3, size=n_random_pixels, dtype=np.int32)
random_depolarized = np.random.randint(low=0, high=2, size=n_random_pixels, dtype=np.int32)

# In[20]:

idx_thinsources = np.where(random_source == 0)[0]
idx_thicksources = np.where(random_source == 1)[0]
idx_mixsources = np.where(random_source == 2)[0]
idx_depolarized = np.where(random_depolarized == 1)[0]

# In[21]:

random_phi_center_thin = np.random.uniform(low=-200, high=200,
                                           size=len(idx_thinsources)).astype(np.float32)
random_phi_center_thick = np.random.uniform(low=-200, high=200,
                                            size=len(idx_thicksources)).astype(np.float32)
random_phi_center_mix = np.random.uniform(low=-200, high=200,
                                          size=(2, len(idx_mixsources))).astype(np.float32)
random_phi_width_thick = np.random.uniform(low=40, high=90,
                                           size=len(idx_thicksources)).astype(np.float32)
random_phi_width_mix = np.random.uniform(low=40, high=90,
                                         size=len(idx_mixsources)).astype(np.float32)
random_sigma_rm_depol = np.random.uniform(low=0, high=30,
                                          size=len(idx_depolarized)).astype(np.float32)

# In[22]:

I_thin = np.random.uniform(low=flux_min, high=flux_max,
                           size=len(idx_thinsources)).astype(np.float32)
I_thick = np.random.uniform(low=flux_min, high=flux_max,
                            size=len(idx_thicksources)).astype(np.float32)
I_mix = np.random.uniform(low=flux_min, high=flux_max,
                          size=(2, len(idx_mixsources))).astype(np.float32)

# In[23]:

alpha_thin = np.random.normal(loc=0.7, scale=0.1, size=len(idx_thinsources)).astype(np.float32)
alpha_thick = np.random.normal(loc=0.7, scale=0.1, size=len(idx_thicksources)).astype(np.float32)
alpha_mix = np.random.normal(loc=0.7, scale=0.1, size=(2, len(idx_mixsources))).astype(np.float32)

# In[24]:

# s_nu_thin = s_nu_at_freq(I_thin, 1.4e9, 1.27962918e9, alpha_thin)
# s_nu_thick = s_nu_at_freq(I_thick, 1.4e9, 1.27962918e9, alpha_thick)
# s_nu_mix = s_nu_at_freq(I_mix, 1.4e9, 1.27962918e9, alpha_mix)
# s_nu_0, nu, nu_0, alpha

# In[25]:

p_thin = pol_dist(I_thin)
p_thick = pol_dist(I_thick)
p_mix = pol_dist(I_mix)

# In[26]:

x_idx_thin = random_pixels[0][idx_thinsources]
y_idx_thin = random_pixels[1][idx_thinsources]
cube[:, y_idx_thin, x_idx_thin] += thinsource_simulation(
    lambda_sq[:, np.newaxis], l2_ref, nu_0, alpha_thin, random_phi_center_thin, p_thin
)

# In[27]:

x_idx_thick = random_pixels[0][idx_thicksources]
y_idx_thick = random_pixels[1][idx_thicksources]
cube[:, y_idx_thick, x_idx_thick] += thicksource_simulation(
    lambda_sq[:, np.newaxis],
    l2_ref,
    nu_0,
    alpha_thick,
    random_phi_width_thick,
    random_phi_center_thick,
    p_thick,
)

# In[28]:

x_idx_mix = random_pixels[0][idx_mixsources]
y_idx_mix = random_pixels[1][idx_mixsources]
cube[:, y_idx_mix, x_idx_mix] += thinsource_simulation(
    lambda_sq[:, np.newaxis],
    l2_ref,
    nu_0,
    alpha_mix[0],
    random_phi_center_mix[0],
    p_mix[0],
) + thicksource_simulation(
    lambda_sq[:, np.newaxis],
    l2_ref,
    nu_0,
    alpha_mix[1],
    random_phi_width_mix,
    random_phi_center_mix[1],
    p_mix[1],
)

# In[29]:

cube_before_depolarization = copy.deepcopy(cube)

# In[30]:

x_idx_depolarized = random_pixels[0][idx_depolarized]
y_idx_depolarized = random_pixels[1][idx_depolarized]
cube[:, y_idx_depolarized,
     x_idx_depolarized] *= np.exp(-2.0 * random_sigma_rm_depol**2 * lambda_sq[:, np.newaxis]**2)

# In[31]:

lambda_sq = lambda_sq[::-1]
cube_before_depolarization = np.flipud(cube_before_depolarization)
cube = np.flipud(cube)

# In[32]:

print(cube_before_depolarization.shape)
print(cube.shape)

# In[33]:

# Convolution with clean-beam
# Assumming bmaj == bmin for now
bmaj = 5e-03  # degrees
bmaj_pix = bmaj / -pix_size
bmin_pix = bmaj_pix

# In[34]:

bmaj_pix

# In[35]:

gkernel = gkern(11, bmaj_pix)

# In[36]:

gkernel.shape

# In[37]:

plt.imshow(gkernel)
plt.colorbar()

# In[38]:

np.max(gkernel)

# In[39]:

cube = signal.convolve(cube, gkernel[np.newaxis, :, :], mode="same")
cube_before_depolarization = signal.convolve(
    cube_before_depolarization, gkernel[np.newaxis, :, :], mode="same"
)

# In[40]:

remove_frac = 0.3
final_idxs = remove_channels(
    lambda2=lambda_sq, remove_frac=remove_frac, random_state=None, chunksize=None
)

# In[41]:

cube = cube[final_idxs]
cube_before_depolarization = cube_before_depolarization[final_idxs]
lambda_sq = lambda_sq[final_idxs]

# In[42]:

plt.imshow(np.abs(cube[5].real))
plt.colorbar()

# In[43]:

print(y_idx_thin)

# In[44]:

print(x_idx_thin)

# In[45]:

print(y_idx_thick)

# In[46]:

print(x_idx_thick)

# In[47]:

print(x_idx_mix)

# In[48]:

print(y_idx_mix)

# In[49]:

x_idx_depolarized

# In[50]:

y_idx_depolarized

# In[51]:

random_sigma_rm_depol

# In[57]:

x_idx_test = 24
y_idx_test = 142

# In[58]:

plt.plot(lambda_sq, cube_before_depolarization[:, y_idx_test, x_idx_test].real, ".")
plt.plot(lambda_sq, cube_before_depolarization[:, y_idx_test, x_idx_test].imag, ".")
plt.plot(lambda_sq, np.abs(cube_before_depolarization[:, y_idx_test, x_idx_test]), ".")

# In[59]:

plt.plot(lambda_sq, cube[:, y_idx_test, x_idx_test].real, ".")
plt.plot(lambda_sq, cube[:, y_idx_test, x_idx_test].imag, ".")
plt.plot(lambda_sq, np.abs(cube[:, y_idx_test, x_idx_test]), ".")

# In[55]:

output_cube = np.concatenate((cube.real[:, None], cube.imag[:, None]), axis=1)

# In[56]:

fits.writeto("test.fits", output_cube, overwrite=True)

# In[ ]:
