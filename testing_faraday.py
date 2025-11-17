"""
Test script for simulating thin, thick, and mixed Faraday sources
using dask arrays for large frequency arrays.
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import c

from csromer.base import Dataset
from csromer.simulation import FaradayThinSource, FaradayThickSource

# Configure matplotlib for LaTeX and colorblind-friendly colors
# Note: LaTeX rendering can be slow and may require LaTeX installation
# Set to False if LaTeX is not available
USE_LATEX = False
if USE_LATEX:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
else:
    # Use matplotlib's built-in math rendering (slower but works without LaTeX)
    plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts for better math rendering
    plt.rcParams['font.family'] = 'serif'

# Colorblind-friendly color palette
# Using colors that work for all types of colorblindness (protanopia, deuteranopia, tritanopia)
# Avoid red-green combinations - use blue-orange, blue-purple, or other safe combinations
COLORS = {
    'blue': '#0066CC',       # Strong blue (works for all)
    'orange': '#FF6600',     # Strong orange (works for all)
    'purple': '#9933FF',     # Purple (works for all)
    'cyan': '#00CCCC',       # Cyan (works for all)
    'magenta': '#CC0066',    # Magenta (works for all)
    'teal': '#009999',       # Teal (works for all)
    'yellow': '#FFCC00',     # Yellow (works for all)
    'black': '#000000',      # Black
    'gray': '#666666',       # Gray
    # Avoid red and green for colorblind users
}

# Speed of light in m/s
c = c.value

# Define frequency bands
low_freq = da.arange(50e6, 350e6, 3.9e3)      # 50–350 MHz, 3.9 kHz step
b2_freq  = da.arange(950e6, 1760e6, 13.44e3)  # 950–1760 MHz, 13.44 kHz step
b5a_freq = da.arange(4.6e9, 8.5e9, 13.44e3)   # 4.6–8.5 GHz, 13.44 kHz step
b5b_freq = da.arange(8.3e9, 15.4e9, 13.44e3)   # 8.3–15.4 GHz, 13.44 kHz step


def calculate_faraday_depth_parameters(freq):
    """Calculate Faraday depth parameters for a given frequency array."""
    wavelength = c / freq
    wavelength_squared = wavelength**2
    wavelength_squared_min = wavelength_squared.min()
    wavelength_squared_max = wavelength_squared.max()
    Delta_wavelength_squared = wavelength_squared_max - wavelength_squared_min
    delta_wavelength = da.diff(wavelength_squared).mean()
    
    delta_phi_nom = 2 * np.sqrt(3) / Delta_wavelength_squared
    delta_phi_full = 2 / (wavelength_squared_max + wavelength_squared_min)
    max_scale = np.pi / wavelength_squared_min
    phi_max = da.abs(np.sqrt(3) / delta_wavelength)
    
    return delta_phi_nom.compute(), delta_phi_full.compute(), max_scale.compute(), phi_max.compute()


def simulate_thin_source_dask(freq_array, phi_gal=10.0, s_nu=1.0, spectral_idx=0.0, dchi=0.0):
    """
    Simulate a thin Faraday source using dask arrays throughout.
    
    Parameters:
    -----------
    freq_array : dask.array
        Frequency array in Hz
    phi_gal : float
        Galactic rotation measure in rad/m²
    s_nu : float
        Source flux at reference frequency
    spectral_idx : float
        Spectral index
    dchi : float
        Intrinsic polarization angle offset in radians
    """
    # Calculate lambda2 using dask operations
    lambda2 = (c / freq_array)**2
    # Reverse to get ascending order (if needed)
    # Check if descending - compute only the diff check
    lambda2_diff = da.diff(lambda2)
    is_descending = da.all(lambda2_diff < 0).compute()
    if is_descending:
        lambda2 = lambda2[::-1]
    
    # Create source with dask lambda2 array
    # The Dataset lambda2 setter will:
    # - Compute nu from lambda2 (dask operation, stays dask)
    # - Compute nu_0 from nu min/max (triggers computation, but that's okay)
    # - Create w array (will be numpy, but we can fix that)
    source = FaradayThinSource(
        lambda2=lambda2,
        phi_gal=phi_gal,
        s_nu=s_nu,
        spectral_idx=spectral_idx,
        dchi=dchi
    )
    
    # Replace w with dask array to keep everything lazy
    if hasattr(source, 'w') and source.w is not None:
        # Create dask array of ones with same shape
        source.w = da.ones_like(lambda2)
        # Update sigma accordingly
        source.sigma = da.ones_like(lambda2)
    
    # Ensure nu is set (it should be set by lambda2 setter, but verify)
    if source.nu is None:
        source.nu = c / da.sqrt(lambda2)
    
    # Now simulate - this will use dask arrays
    # The simulate method uses numpy operations, but they work with dask arrays
    source.simulate()
    
    return source


def simulate_thick_source_dask(freq_array, phi_fg=5.0, phi_center=0.0, s_nu=1.0, spectral_idx=0.0):
    """
    Simulate a thick Faraday source using dask arrays throughout.
    
    Parameters:
    -----------
    freq_array : dask.array
        Frequency array in Hz
    phi_fg : float
        Faraday depth width in rad/m²
    phi_center : float
        Center Faraday depth in rad/m²
    s_nu : float
        Source flux at reference frequency
    spectral_idx : float
        Spectral index
    """
    # Calculate lambda2 using dask operations
    lambda2 = (c / freq_array)**2
    # Check if descending
    lambda2_diff = da.diff(lambda2)
    is_descending = da.all(lambda2_diff < 0).compute()
    if is_descending:
        lambda2 = lambda2[::-1]
    
    # Create source with dask lambda2 array
    source = FaradayThickSource(
        lambda2=lambda2,
        phi_fg=phi_fg,
        phi_center=phi_center,
        s_nu=s_nu,
        spectral_idx=spectral_idx
    )
    
    # Replace w with dask array to keep everything lazy
    if hasattr(source, '_Dataset__w') and source._Dataset__w is not None:
        source._Dataset__w = da.ones_like(lambda2)
        source._Dataset__sigma = da.ones_like(lambda2)
    
    # Ensure nu is set
    if source.nu is None:
        source._Dataset__nu = c / da.sqrt(lambda2)
    
    # Simulate using dask arrays
    source.simulate()
    
    return source


def simulate_mixed_sources_dask(freq_array, sources_config):
    """
    Simulate multiple sources and combine them using dask arrays.
    
    Parameters:
    -----------
    freq_array : dask.array
        Frequency array in Hz
    sources_config : list of dict
        List of source configurations, each dict should contain:
        - 'type': 'thin' or 'thick'
        - Other parameters specific to the source type
    """
    # Calculate lambda2 using dask operations (once for all sources)
    lambda2 = (c / freq_array)**2
    lambda2_diff = da.diff(lambda2)
    is_descending = da.all(lambda2_diff < 0).compute()
    if is_descending:
        lambda2 = lambda2[::-1]
    
    combined_source = None
    
    for i, config in enumerate(sources_config):
        # Make a copy of config to avoid modifying the original
        config = config.copy()
        source_type = config.pop('type')
        
        if source_type == 'thin':
            source = FaradayThinSource(lambda2=lambda2, **config)
        elif source_type == 'thick':
            source = FaradayThickSource(lambda2=lambda2, **config)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        # Replace w with dask array to keep everything lazy
        if hasattr(source, '_Dataset__w') and source._Dataset__w is not None:
            source._Dataset__w = da.ones_like(lambda2)
            source._Dataset__sigma = da.ones_like(lambda2)
        
        # Ensure nu is set
        if source.nu is None:
            source._Dataset__nu = c / da.sqrt(lambda2)
        
        source.simulate()
        
        if combined_source is None:
            combined_source = source
        else:
            # For addition, we need to handle dask arrays
            # The __add__ method checks (self.nu == other.nu).all()
            # When nu is a dask array, this should work, but if nu is None or same object,
            # it might return a boolean. Let's ensure nu is a dask array for both
            if combined_source.nu is None:
                combined_source._Dataset__nu = c / da.sqrt(lambda2)
            
            # The comparison will work with dask arrays
            combined_source = combined_source + source
    
    return combined_source


# Keep old functions for backward compatibility, but use dask versions
simulate_thin_source = simulate_thin_source_dask
simulate_thick_source = simulate_thick_source_dask
simulate_mixed_sources = simulate_mixed_sources_dask


def dask_forward_ft(dataset, phi_array, l2_ref=None, normalize=True):
    """
    Dask-aware forward Fourier transform to compute Faraday depth spectrum.
    
    Parameters:
    -----------
    dataset : Dataset
        Dataset with lambda2 and data arrays (can be dask arrays)
    phi_array : array-like
        Faraday depth array in rad/m²
    l2_ref : float, optional
        Reference lambda² value. If None, uses dataset.l2_ref
    normalize : bool
        If True, normalize to get proper Jy/RMSF units. If False, returns unnormalized spectrum.
        
    Returns:
    --------
    fd_spectrum : dask.array
        Faraday depth spectrum (complex) in units of Jy/RMSF if normalize=True
    """
    if l2_ref is None:
        if hasattr(dataset, 'l2_ref') and dataset.l2_ref is not None:
            l2_ref = dataset.l2_ref
        else:
            # Compute reference lambda² as weighted mean
            if hasattr(dataset.lambda2, 'compute'):
                l2_ref = da.mean(dataset.lambda2).compute()
            else:
                l2_ref = np.mean(dataset.lambda2)
    
    # Ensure phi_array is numpy (it's typically small)
    if hasattr(phi_array, 'compute'):
        phi_array = phi_array.compute()
    phi_array = np.asarray(phi_array)
    
    # Get lambda2, data, weights, and spectral index from dataset
    lambda2 = dataset.lambda2
    data = dataset.data
    
    # Get weights from dataset
    if hasattr(dataset, 'w') and dataset.w is not None:
        w = dataset.w
    else:
        # Default to ones if weights not set
        if hasattr(lambda2, 'compute'):
            w = da.ones_like(lambda2)
        else:
            w = np.ones_like(lambda2)
    
    # Get spectral index factor s
    if hasattr(dataset, 's') and dataset.s is not None:
        s = dataset.s
    else:
        # Default to ones if spectral index factor not set
        if hasattr(lambda2, 'compute'):
            s = da.ones_like(lambda2)
        else:
            s = np.ones_like(lambda2)
    
    # Compute l2 - l2_ref
    # For dask arrays, we need to handle this carefully
    if hasattr(lambda2, 'compute'):
        l2_diff = lambda2 - l2_ref
    else:
        l2_diff = lambda2 - l2_ref
    
    # Compute exp(2j * phi * (lambda2 - l2_ref))
    # phi_array[:, np.newaxis] * l2_diff[np.newaxis, :] creates a 2D array
    # Shape: (n_phi, n_channels)
    phi_2d = da.asarray(phi_array)[:, np.newaxis] if hasattr(lambda2, 'compute') else phi_array[:, np.newaxis]
    l2_diff_2d = l2_diff[np.newaxis, :] if hasattr(lambda2, 'compute') else l2_diff[np.newaxis, :]
    
    exp_factor = da.exp(2.0j * phi_2d * l2_diff_2d) if hasattr(lambda2, 'compute') else np.exp(2.0j * phi_2d * l2_diff_2d)
    
    # Normalize to get proper Jy/RMSF units
    # Following NDFT.forward_normalized approach:
    # 1. Weight the data: weighted_data = data * (w / s)
    # 2. Transform: F(φ) = Σ[weighted_data * exp(2j*φ*(λ²-λ²_ref))]
    # 3. Normalize: F(φ) = (s_mean / n_phi) * transform
    if normalize:
        n_phi = len(phi_array)
        
        # Compute weighted data: data * (w / s)
        # This accounts for both weights and spectral index
        if hasattr(data, 'compute'):
            # Handle dask arrays
            weighted_data = data * (w / s)
        else:
            # Handle numpy arrays
            if hasattr(w, 'compute'):
                w = w.compute()
            if hasattr(s, 'compute'):
                s = s.compute()
            weighted_data = data * (w / s)
        
        # Matrix multiplication: sum over channels with weighted data
        # weighted_data shape: (n_channels,)
        # exp_factor shape: (n_phi, n_channels)
        # Result shape: (n_phi,)
        if hasattr(weighted_data, 'compute'):
            # Use dask einsum for matrix multiplication: sum over last axis
            # 'ij,j->i' means: exp_factor[i,j] * weighted_data[j] summed over j -> result[i]
            fd_spectrum = da.einsum('ij,j->i', exp_factor, weighted_data)
        else:
            fd_spectrum = np.dot(exp_factor, weighted_data)
        
        # Get mean spectral index factor for final normalization
        if hasattr(s, 'mean'):
            if hasattr(s, 'compute'):
                s_mean = s.mean().compute()
            else:
                s_mean = s.mean()
        else:
            s_mean = np.mean(s) if not hasattr(s, 'compute') else s.compute().mean()
        
        # Final normalization: F(φ) = (s_mean / n_phi) * transform
        # This gives proper Jy/RMSF units
        if hasattr(fd_spectrum, '__truediv__'):
            fd_spectrum = fd_spectrum * s_mean / n_phi
        else:
            fd_spectrum = fd_spectrum * s_mean / n_phi
    else:
        # No normalization - just do the basic transform
        # Matrix multiplication: sum over channels
        if hasattr(data, 'compute'):
            fd_spectrum = da.einsum('ij,j->i', exp_factor, data)
        else:
            fd_spectrum = np.dot(exp_factor, data)
    
    return fd_spectrum


def calculate_faraday_depth_spectrum(source, phi_max=None, n_phi=None, oversampling=8, band_phi_max=None, cellsize=None):
    """
    Calculate Faraday depth spectrum for a source.
    
    Parameters:
    -----------
    source : FaradaySource
        Source with simulated data
    phi_max : float, optional
        Maximum Faraday depth to compute (rad/m²). If None, uses band_phi_max or calculates from data.
    n_phi : int, optional
        Number of Faraday depth points. If None, calculated from cellsize or uses default.
    oversampling : float
        Oversampling factor for cellsize calculation (used if cellsize not provided)
    band_phi_max : float, optional
        Maximum phi for the band (from statistics). If provided, uses this.
    cellsize : float, optional
        Desired cellsize in phi-space (rad/m²). If provided, n_phi is calculated from this.
        
    Returns:
    --------
    phi : array
        Faraday depth array (rad/m²)
    fd_spectrum : array
        Faraday depth spectrum (complex)
    """
    # Calculate appropriate phi range based on lambda2 coverage
    if hasattr(source.lambda2, 'compute'):
        l2_min = da.min(source.lambda2).compute()
        l2_max = da.max(source.lambda2).compute()
        delta_l2 = l2_max - l2_min
    else:
        l2_min = np.min(source.lambda2)
        l2_max = np.max(source.lambda2)
        delta_l2 = l2_max - l2_min
    
    # Calculate theoretical resolution
    delta_phi_fwhm = 2.0 * np.sqrt(3.0) / delta_l2
    delta_phi_theo = np.pi / l2_min
    
    # Determine phi_max
    # If phi_max is explicitly provided, use it (for specific ranges like -1000 to 1000)
    if phi_max is not None:
        phi_max_actual = phi_max
    elif band_phi_max is not None:
        # Use the band-specific phi_max from statistics
        phi_max_actual = band_phi_max
    else:
        # Calculate from data
        theoretical_max = np.sqrt(3) / (delta_l2 / len(source.lambda2)) * 10.0
        phi_max_actual = theoretical_max
    
    # Determine cellsize and n_phi
    if cellsize is not None:
        # Use provided cellsize
        phi_range = 2 * phi_max_actual
        n_phi = int(np.ceil(phi_range / cellsize))
        # Ensure n_phi is even for symmetry
        if n_phi % 2 == 1:
            n_phi += 1
    elif n_phi is None:
        # Calculate n_phi from oversampling
        recommended_cellsize = delta_phi_fwhm / oversampling
        phi_range = 2 * phi_max_actual
        n_phi = int(np.ceil(phi_range / recommended_cellsize))
        # Ensure n_phi is even for symmetry
        if n_phi % 2 == 1:
            n_phi += 1
    
    # Create phi array
    phi = np.linspace(-phi_max_actual, phi_max_actual, n_phi)
    
    # Compute Faraday depth spectrum
    fd_spectrum = dask_forward_ft(source, phi)
    
    # Compute if it's a dask array
    if hasattr(fd_spectrum, 'compute'):
        fd_spectrum = fd_spectrum.compute()
    
    return phi, fd_spectrum


def print_source_info(source, name="Source"):
    """Print information about a simulated source (handles both numpy and dask arrays)."""
    print(f"\n{name}:")
    print(f"  Number of channels: {source.m}")
    
    # Handle dask arrays for min/max
    # Check if nu exists and is not None
    if source.nu is not None:
        if hasattr(source.nu, 'compute'):
            nu_min = da.min(source.nu).compute() / 1e6
            nu_max = da.max(source.nu).compute() / 1e6
        else:
            nu_min = np.min(source.nu) / 1e6
            nu_max = np.max(source.nu) / 1e6
        print(f"  Frequency range: {nu_min:.2f} - {nu_max:.2f} MHz")
    else:
        print(f"  Frequency range: N/A (nu not set)")
    
    # Lambda2 should always be set
    if hasattr(source.lambda2, 'compute'):
        l2_min = da.min(source.lambda2).compute()
        l2_max = da.max(source.lambda2).compute()
    else:
        l2_min = np.min(source.lambda2)
        l2_max = np.max(source.lambda2)
    print(f"  Lambda² range: {l2_min:.6e} - {l2_max:.6e} m²")
    
    # Check if data is dask array
    if source.data is not None:
        if hasattr(source.data, 'compute'):
            print(f"  Data type: dask array (dtype: {source.data.dtype})")
            data_abs = da.abs(source.data)
            data_min = da.min(data_abs).compute()
            data_max = da.max(data_abs).compute()
        else:
            print(f"  Data type: {source.data.dtype}")
            data_min = np.min(np.abs(source.data))
            data_max = np.max(np.abs(source.data))
        print(f"  Polarization amplitude range: {data_min:.6e} - {data_max:.6e}")
    else:
        print(f"  Data: Not simulated yet")
    
    print(f"  Spectral index: {source.spectral_idx}")


def plot_2x2_clean_vs_rfi(clean_source, rfi_source, phi_clean, fd_clean, phi_rfi, fd_rfi, 
                          band_name, source_type, figsize=(18, 12)):
    """
    Create a 2x2 plot comparing clean source vs source with RFI.
    
    Layout:
    - Top left: Clean polarization vs lambda²
    - Top right: Clean Faraday depth spectrum
    - Bottom left: RFI polarization vs lambda²
    - Bottom right: RFI Faraday depth spectrum
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Get data for clean source
    if hasattr(clean_source.lambda2, 'compute'):
        l2_clean = clean_source.lambda2.compute()
        data_clean = clean_source.data.compute()
    else:
        l2_clean = clean_source.lambda2
        data_clean = clean_source.data
    
    # Get data for RFI source
    if hasattr(rfi_source.lambda2, 'compute'):
        l2_rfi = rfi_source.lambda2.compute()
        data_rfi = rfi_source.data.compute()
    else:
        l2_rfi = rfi_source.lambda2
        data_rfi = rfi_source.data
    
    # Top left: Clean polarization - use markers only (no lines) to show potential gaps
    ax = axes[0, 0]
    ax.plot(l2_clean, np.abs(data_clean), '.', color=COLORS['blue'], markersize=0.6, alpha=0.9, label=r'$|P|$')
    ax.plot(l2_clean, data_clean.real, '.', color=COLORS['purple'], markersize=0.5, alpha=0.8, label=r'$\mathrm{Re}(P)$')
    ax.plot(l2_clean, data_clean.imag, '.', color=COLORS['orange'], markersize=0.5, alpha=0.8, label=r'$\mathrm{Im}(P)$')
    ax.set_xlabel(r'$\lambda^2$ [m²]', fontsize=11)
    ax.set_ylabel('Polarization [Jy]', fontsize=11)
    ax.set_title(r'Clean: Polarization vs $\lambda^2$', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Top right: Clean Faraday depth
    ax = axes[0, 1]
    fd_abs_clean = np.abs(fd_clean)
    ax.plot(phi_clean, fd_abs_clean, '-', color=COLORS['black'], linewidth=1.5, alpha=0.9, label=r'$|F(\phi)|$')
    ax.plot(phi_clean, fd_clean.real, '-', color=COLORS['purple'], linewidth=1.2, alpha=0.8, label=r'$\mathrm{Re}(F(\phi))$')
    ax.plot(phi_clean, fd_clean.imag, '-', color=COLORS['orange'], linewidth=1.2, alpha=0.8, label=r'$\mathrm{Im}(F(\phi))$')
    peak_idx = np.argmax(fd_abs_clean)
    peak_phi = phi_clean[peak_idx]
    ax.axvline(peak_phi, color=COLORS['blue'], linestyle='--', linewidth=1.5, alpha=0.6, 
               label=r'$\mathrm{Peak}$ at $\phi=' + f'{peak_phi:.1f}' + r'$ rad/m²')
    ax.set_xlabel(r'$\phi$ [rad/m²]', fontsize=11)
    ax.set_ylabel(r'Faraday Intensity [Jy/RMSF]', fontsize=11)
    ax.set_title('Clean: Faraday Depth Spectrum', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1000, 1000)
    
    # Bottom left: RFI polarization - markers only to show gaps
    ax = axes[1, 0]
    ax.plot(l2_rfi, np.abs(data_rfi), '.', color=COLORS['blue'], markersize=0.6, alpha=0.9, label=r'$|P|$')
    ax.plot(l2_rfi, data_rfi.real, '.', color=COLORS['purple'], markersize=0.5, alpha=0.8, label=r'$\mathrm{Re}(P)$')
    ax.plot(l2_rfi, data_rfi.imag, '.', color=COLORS['orange'], markersize=0.5, alpha=0.8, label=r'$\mathrm{Im}(P)$')
    ax.set_xlabel(r'$\lambda^2$ [m²]', fontsize=11)
    ax.set_ylabel('Polarization [Jy]', fontsize=11)
    ax.set_title(r'With RFI: Polarization vs $\lambda^2$', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Bottom right: RFI Faraday depth
    ax = axes[1, 1]
    fd_abs_rfi = np.abs(fd_rfi)
    ax.plot(phi_rfi, fd_abs_rfi, '-', color=COLORS['black'], linewidth=1.5, alpha=0.9, label=r'$|F(\phi)|$')
    ax.plot(phi_rfi, fd_rfi.real, '-', color=COLORS['purple'], linewidth=1.2, alpha=0.8, label=r'$\mathrm{Re}(F(\phi))$')
    ax.plot(phi_rfi, fd_rfi.imag, '-', color=COLORS['orange'], linewidth=1.2, alpha=0.8, label=r'$\mathrm{Im}(F(\phi))$')
    peak_idx = np.argmax(fd_abs_rfi)
    peak_phi = phi_rfi[peak_idx]
    ax.axvline(peak_phi, color=COLORS['blue'], linestyle='--', linewidth=1.5, alpha=0.6,
               label=r'$\mathrm{Peak}$ at $\phi=' + f'{peak_phi:.1f}' + r'$ rad/m²')
    ax.set_xlabel(r'$\phi$ [rad/m²]', fontsize=11)
    ax.set_ylabel(r'Faraday Intensity [Jy/RMSF]', fontsize=11)
    ax.set_title('With RFI: Faraday Depth Spectrum', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1000, 1000)
    
    plt.suptitle(f'{source_type} Source: Clean vs RFI ({band_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_2x2_depolarization(clean_source, depol_source, phi_clean, fd_clean, phi_depol, fd_depol,
                            band_name, source_type, figsize=(18, 12)):
    """
    Create a 2x2 plot comparing clean source vs source with depolarization.
    
    Layout:
    - Top left: Clean polarization vs lambda²
    - Top right: Clean Faraday depth spectrum
    - Bottom left: Depolarized polarization vs lambda²
    - Bottom right: Depolarized Faraday depth spectrum
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Get data for clean source
    if hasattr(clean_source.lambda2, 'compute'):
        l2_clean = clean_source.lambda2.compute()
        data_clean = clean_source.data.compute()
    else:
        l2_clean = clean_source.lambda2
        data_clean = clean_source.data
    
    # Get data for depolarized source
    if hasattr(depol_source.lambda2, 'compute'):
        l2_depol = depol_source.lambda2.compute()
        data_depol = depol_source.data.compute()
    else:
        l2_depol = depol_source.lambda2
        data_depol = depol_source.data
    
    # Top left: Clean polarization - use markers only (no lines) to show potential gaps
    ax = axes[0, 0]
    ax.plot(l2_clean, np.abs(data_clean), '.', color=COLORS['blue'], markersize=0.6, alpha=0.9, label=r'$|P|$')
    ax.plot(l2_clean, data_clean.real, '.', color=COLORS['blue'], markersize=0.5, alpha=0.8, label=r'$\mathrm{Re}(P)$')
    ax.plot(l2_clean, data_clean.imag, '.', color=COLORS['blue'], markersize=0.5, alpha=0.8, label=r'$\mathrm{Im}(P)$')
    ax.set_xlabel(r'$\lambda^2$ [m²]', fontsize=11)
    ax.set_ylabel('Polarization [Jy]', fontsize=11)
    ax.set_title(r'Clean: Polarization vs $\lambda^2$', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Top right: Clean Faraday depth
    ax = axes[0, 1]
    fd_abs_clean = np.abs(fd_clean)
    ax.plot(phi_clean, fd_abs_clean, '-', color=COLORS['blue'], linewidth=2.0, alpha=0.95, label=r'$|F(\phi)|$')
    ax.plot(phi_clean, fd_clean.real, '--', color=COLORS['cyan'], linewidth=1.8, alpha=0.85, label=r'$\mathrm{Re}(F(\phi))$')
    ax.plot(phi_clean, fd_clean.imag, ':', color=COLORS['teal'], linewidth=1.8, alpha=0.85, label=r'$\mathrm{Im}(F(\phi))$')
    peak_idx = np.argmax(fd_abs_clean)
    peak_phi = phi_clean[peak_idx]
    ax.axvline(peak_phi, color=COLORS['blue'], linestyle='--', linewidth=1.5, alpha=0.6,
               label=r'$\mathrm{Peak}$ at $\phi=' + f'{peak_phi:.1f}' + r'$ rad/m²')
    ax.set_xlabel(r'$\phi$ [rad/m²]', fontsize=11)
    ax.set_ylabel(r'Faraday Intensity [Jy/RMSF]', fontsize=11)
    ax.set_title('Clean: Faraday Depth Spectrum', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1000, 1000)
    
    # Bottom left: Depolarized polarization - markers only to show gaps
    ax = axes[1, 0]
    ax.plot(l2_depol, np.abs(data_depol), '.', color=COLORS['orange'], markersize=0.6, alpha=0.9, label=r'$|P|$')
    ax.plot(l2_depol, data_depol.real, '.', color=COLORS['orange'], markersize=0.5, alpha=0.8, label=r'$\mathrm{Re}(P)$')
    ax.plot(l2_depol, data_depol.imag, '.', color=COLORS['orange'], markersize=0.5, alpha=0.8, label=r'$\mathrm{Im}(P)$')
    ax.set_xlabel(r'$\lambda^2$ [m²]', fontsize=11)
    ax.set_ylabel('Polarization [Jy]', fontsize=11)
    ax.set_title(r'Depolarized: Polarization vs $\lambda^2$', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Bottom right: Depolarized Faraday depth
    ax = axes[1, 1]
    fd_abs_depol = np.abs(fd_depol)
    ax.plot(phi_depol, fd_abs_depol, '-', color=COLORS['orange'], linewidth=2.0, alpha=0.95, label=r'$|F(\phi)|$')
    ax.plot(phi_depol, fd_depol.real, '--', color=COLORS['purple'], linewidth=1.8, alpha=0.85, label=r'$\mathrm{Re}(F(\phi))$')
    ax.plot(phi_depol, fd_depol.imag, ':', color=COLORS['magenta'], linewidth=1.8, alpha=0.85, label=r'$\mathrm{Im}(F(\phi))$')
    peak_idx = np.argmax(fd_abs_depol)
    peak_phi = phi_depol[peak_idx]
    ax.axvline(peak_phi, color=COLORS['orange'], linestyle='--', linewidth=1.5, alpha=0.6,
               label=r'$\mathrm{Peak}$ at $\phi=' + f'{peak_phi:.1f}' + r'$ rad/m²')
    ax.set_xlabel(r'$\phi$ [rad/m²]', fontsize=11)
    ax.set_ylabel(r'Faraday Intensity [Jy/RMSF]', fontsize=11)
    ax.set_title('Depolarized: Faraday Depth Spectrum', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1000, 1000)
    
    plt.suptitle(f'{source_type} Source: Clean vs Depolarized ({band_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_2x1_delta_comparison(band_configs, figsize=(16, 6)):
    """
    Create a 2x1 plot comparing nominal delta vs full resolution delta for all bands.
    
    Parameters:
    -----------
    band_configs : dict
        Dictionary with band names as keys and tuples (delta_nom, delta_full, band_name) as values
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    band_names = list(band_configs.keys())
    delta_nom_values = [band_configs[b][0] for b in band_names]
    delta_full_values = [band_configs[b][1] for b in band_names]
    
    # Left: Bar plot comparison
    ax = axes[0]
    x = np.arange(len(band_names))
    width = 0.35
    ax.bar(x - width/2, delta_nom_values, width, label=r'$\Delta\phi_{\mathrm{nom}}$', 
           color=COLORS['blue'], alpha=0.8)
    ax.bar(x + width/2, delta_full_values, width, label=r'$\Delta\phi_{\mathrm{full}}$', 
           color=COLORS['orange'], alpha=0.8)
    ax.set_xlabel('Band', fontsize=12)
    ax.set_ylabel(r'$\Delta\phi$ [rad/m²]', fontsize=12)
    ax.set_title('Faraday Depth Resolution: Nominal vs Full', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # Right: Ratio plot
    ax = axes[1]
    ratios = [dn / df for dn, df in zip(delta_nom_values, delta_full_values)]
    ax.bar(band_names, ratios, color=COLORS['purple'], alpha=0.8)
    ax.axhline(1.0, color=COLORS['black'], linestyle='--', linewidth=1.5, alpha=0.6, 
               label='Equal resolution')
    ax.set_xlabel('Band', fontsize=12)
    ax.set_ylabel(r'$\Delta\phi_{\mathrm{nom}} / \Delta\phi_{\mathrm{full}}$', fontsize=12)
    ax.set_title('Resolution Ratio', fontsize=13, fontweight='bold')
    ax.set_xticklabels(band_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Faraday Depth Resolution Comparison Across SKA Bands', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("=" * 80)
    print("Faraday Source Simulation Test - Comprehensive Plotting")
    print("=" * 80)
    
    # Define SKA band configurations
    SKA_BANDS = {
        'SKA-LOW': {
            'freq': low_freq,
            'name': 'SKA-LOW',
            'short': 'LOW'
        },
        'SKA-MID B2': {
            'freq': b2_freq,
            'name': 'SKA-MID B2',
            'short': 'B2'
        },
        'SKA-MID B5a': {
            'freq': b5a_freq,
            'name': 'SKA-MID B5a',
            'short': 'B5a'
        },
        'SKA-MID B5b': {
            'freq': b5b_freq,
            'name': 'SKA-MID B5b',
            'short': 'B5b'
        }
    }
    
    # Calculate and print Faraday depth parameters
    print("\nFaraday Depth Parameters:")
    print(f"{'Band':<12} {'Δφ_nom [rad/m²]':>20} {'Δφ_full [rad/m²]':>20} {'Max Scale [rad/m²]':>20} {'|φ|max [rad/m²]':>20}")
    print("-" * 92)
    
    band_params = {}
    band_delta_configs = {}
    
    for band_name, band_info in SKA_BANDS.items():
        delta_nom, delta_full, max_scale, phi_max = calculate_faraday_depth_parameters(band_info['freq'])
        band_params[band_name] = {
            'delta_nom': delta_nom,
            'delta_full': delta_full,
            'max_scale': max_scale,
            'phi_max': phi_max
        }
        band_delta_configs[band_name] = (delta_nom, delta_full, band_info['name'])
        print(f"{band_name:<12} {delta_nom:>20.3e} {delta_full:>20.3e} {max_scale:>20.3e} {phi_max:>20.3e}")
    
    # ========================================================================
    # Simulation Parameters
    # ========================================================================
    # Source parameters (adjustable per band if needed)
    THIN_PARAMS = {
        'phi_gal': 15.0,  # rad/m²
        's_nu': 1.0,
        'spectral_idx': -0.7,
        'dchi': 0.0
    }
    
    THICK_PARAMS = {
        'phi_fg': 10.0,  # rad/m²
        'phi_center': 20.0,  # rad/m²
        's_nu': 1.0,
        'spectral_idx': -0.7
    }
    
    MIXED_CONFIG = [
        {
            'type': 'thin',
            'phi_gal': -400.0,
            's_nu': 0.5,
            'spectral_idx': -0.7,
            'dchi': 0.0
        },
        {
            'type': 'thick',
            'phi_fg': 50.0,
            'phi_center': 400.0,
            's_nu': 0.5,
            'spectral_idx': -0.7
        }
    ]
    
    # Effects parameters
    RFI_REMOVE_FRAC = 0.1  # 10% channels removed
    DEPOL_SIGMA_RM_THIN = 5.0  # rad/m² for thin sources
    DEPOL_SIGMA_RM_THICK = 3.0  # rad/m² for thick sources
    
    # ========================================================================
    # Generate Plots for All Bands
    # ========================================================================
    print("\n" + "=" * 80)
    print("Generating Comprehensive Plots for All SKA Bands...")
    print("=" * 80)
    
    # ========================================================================
    # Generate Plots for Each Band (process and plot immediately)
    # ========================================================================
    phi_max = 1000  # Fixed range for all plots
    cellsize = 0.5
    
    for band_idx, (band_name, band_info) in enumerate(SKA_BANDS.items(), 1):
        print(f"\n{'='*80}")
        print(f"Processing Band {band_idx}/{len(SKA_BANDS)}: {band_name}")
        print(f"{'='*80}")
        freq = band_info['freq']
        phi_max_band = band_params[band_name]['phi_max']
        
        # Simulate sources for this band
        print(f"  Simulating sources...")
        # Thin source (clean)
        print(f"    - Thin source (clean)...")
        thin_clean = simulate_thin_source(freq, **THIN_PARAMS)
        # Thin source with RFI
        print(f"    - Thin source (RFI)...")
        thin_rfi = simulate_thin_source(freq, **THIN_PARAMS)
        thin_rfi.remove_channels(remove_frac=RFI_REMOVE_FRAC, random_state=np.random.RandomState(42))
        # Thin source with depolarization
        print(f"    - Thin source (depolarized)...")
        thin_depol = simulate_thin_source(freq, **THIN_PARAMS)
        thin_depol.add_external_faraday_depolarization(sigma_rm=DEPOL_SIGMA_RM_THIN)
        
        # Thick source (clean) - skip for SKA-LOW
        if band_name != 'SKA-LOW':
            print(f"    - Thick source (clean)...")
            thick_clean = simulate_thick_source(freq, **THICK_PARAMS)
            # Thick source with RFI
            print(f"    - Thick source (RFI)...")
            thick_rfi = simulate_thick_source(freq, **THICK_PARAMS)
            thick_rfi.remove_channels(remove_frac=RFI_REMOVE_FRAC, random_state=np.random.RandomState(43))
            # Thick source with depolarization
            print(f"    - Thick source (depolarized)...")
            thick_depol = simulate_thick_source(freq, **THICK_PARAMS)
            thick_depol.add_external_faraday_depolarization(sigma_rm=DEPOL_SIGMA_RM_THICK)
            
            # Mixed source (clean)
            print(f"    - Mixed source (clean)...")
            mixed_clean = simulate_mixed_sources(freq, MIXED_CONFIG)
            # Mixed source with RFI
            print(f"    - Mixed source (RFI)...")
            mixed_rfi = simulate_mixed_sources(freq, MIXED_CONFIG)
            mixed_rfi.remove_channels(remove_frac=RFI_REMOVE_FRAC, random_state=np.random.RandomState(44))
        else:
            # SKA-LOW: only thin sources
            thick_clean = None
            thick_rfi = None
            thick_depol = None
            mixed_clean = None
            mixed_rfi = None
        
        # Calculate Faraday depth spectra
        print(f"  Calculating Faraday depth spectra...")
        # Thin sources
        print(f"    - Thin clean...")
        phi_thin_clean, fd_thin_clean = calculate_faraday_depth_spectrum(
            thin_clean, phi_max=phi_max, cellsize=cellsize, band_phi_max=phi_max_band)
        print(f"    - Thin RFI...")
        phi_thin_rfi, fd_thin_rfi = calculate_faraday_depth_spectrum(
            thin_rfi, phi_max=phi_max, cellsize=cellsize, band_phi_max=phi_max_band)
        print(f"    - Thin depolarized...")
        phi_thin_depol, fd_thin_depol = calculate_faraday_depth_spectrum(
            thin_depol, phi_max=phi_max, cellsize=cellsize, band_phi_max=phi_max_band)
        
        # Thick sources - skip for SKA-LOW
        if band_name != 'SKA-LOW':
            print(f"    - Thick clean...")
            phi_thick_clean, fd_thick_clean = calculate_faraday_depth_spectrum(
                thick_clean, phi_max=phi_max, cellsize=cellsize, band_phi_max=phi_max_band)
            print(f"    - Thick RFI...")
            phi_thick_rfi, fd_thick_rfi = calculate_faraday_depth_spectrum(
                thick_rfi, phi_max=phi_max, cellsize=cellsize, band_phi_max=phi_max_band)
            print(f"    - Thick depolarized...")
            phi_thick_depol, fd_thick_depol = calculate_faraday_depth_spectrum(
                thick_depol, phi_max=phi_max, cellsize=cellsize, band_phi_max=phi_max_band)
            
            # Mixed sources
            print(f"    - Mixed clean...")
            phi_mixed_clean, fd_mixed_clean = calculate_faraday_depth_spectrum(
                mixed_clean, phi_max=phi_max, cellsize=cellsize, band_phi_max=phi_max_band)
            print(f"    - Mixed RFI...")
            phi_mixed_rfi, fd_mixed_rfi = calculate_faraday_depth_spectrum(
                mixed_rfi, phi_max=phi_max, cellsize=cellsize, band_phi_max=phi_max_band)
        else:
            # SKA-LOW: no thick/mixed sources
            phi_thick_clean = None
            fd_thick_clean = None
            phi_thick_rfi = None
            fd_thick_rfi = None
            phi_thick_depol = None
            fd_thick_depol = None
            phi_mixed_clean = None
            fd_mixed_clean = None
            phi_mixed_rfi = None
            fd_mixed_rfi = None
        
        # Generate plots immediately for this band
        print(f"  Generating plots for {band_name}...")
        
        # Plot 1: 2x2 Thin source clean vs RFI
        print(f"    - Plot 1: Thin clean vs RFI...")
        fig = plot_2x2_clean_vs_rfi(
            thin_clean, thin_rfi,
            phi_thin_clean, fd_thin_clean,
            phi_thin_rfi, fd_thin_rfi,
            band_info['name'], 'Thin'
        )
        filename = f'thin_clean_vs_rfi_{band_info["short"]}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      Saved: {filename}")
        # Free memory - keep thin_clean and its spectra for plot 4a
        del phi_thin_rfi, fd_thin_rfi, thin_rfi
        
        # Plot 2: 2x2 Thick source clean vs RFI - skip for SKA-LOW
        if band_name != 'SKA-LOW':
            print(f"    - Plot 2: Thick clean vs RFI...")
            fig = plot_2x2_clean_vs_rfi(
                thick_clean, thick_rfi,
                phi_thick_clean, fd_thick_clean,
                phi_thick_rfi, fd_thick_rfi,
                band_info['name'], 'Thick'
            )
            filename = f'thick_clean_vs_rfi_{band_info["short"]}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"      Saved: {filename}")
            # Free memory - keep thick_clean and its spectra for plot 4b
            del phi_thick_rfi, fd_thick_rfi, thick_rfi
            
            # Plot 3: 2x2 Mixed source clean vs RFI
            print(f"    - Plot 3: Mixed clean vs RFI...")
            fig = plot_2x2_clean_vs_rfi(
                mixed_clean, mixed_rfi,
                phi_mixed_clean, fd_mixed_clean,
                phi_mixed_rfi, fd_mixed_rfi,
                band_info['name'], 'Mixed'
            )
            filename = f'mixed_clean_vs_rfi_{band_info["short"]}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"      Saved: {filename}")
            # Free memory - mixed sources are done
            del phi_mixed_clean, fd_mixed_clean, phi_mixed_rfi, fd_mixed_rfi
            del mixed_clean, mixed_rfi
        
        # Plot 4a: 2x2 Thin depolarization
        print(f"    - Plot 4a: Thin depolarization...")
        fig = plot_2x2_depolarization(
            thin_clean, thin_depol,
            phi_thin_clean, fd_thin_clean,
            phi_thin_depol, fd_thin_depol,
            band_info['name'], 'Thin'
        )
        filename = f'thin_depolarization_{band_info["short"]}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      Saved: {filename}")
        # Free memory - all thin sources are done
        del phi_thin_clean, fd_thin_clean, phi_thin_depol, fd_thin_depol
        del thin_clean, thin_depol
        
        # Plot 4b: 2x2 Thick depolarization - skip for SKA-LOW
        if band_name != 'SKA-LOW':
            print(f"    - Plot 4b: Thick depolarization...")
            fig = plot_2x2_depolarization(
                thick_clean, thick_depol,
                phi_thick_clean, fd_thick_clean,
                phi_thick_depol, fd_thick_depol,
                band_info['name'], 'Thick'
            )
            filename = f'thick_depolarization_{band_info["short"]}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"      Saved: {filename}")
            # Free memory - all thick sources are done
            del phi_thick_clean, fd_thick_clean, phi_thick_depol, fd_thick_depol
            del thick_clean, thick_depol
        
        print(f"  Completed {band_name}!")
    
    # Plot 5: 2x1 Delta comparison (nominal vs full resolution) - only needs band params
    print(f"\n{'='*80}")
    print("Generating final plot: Delta comparison...")
    print(f"{'='*80}")
    fig = plot_2x1_delta_comparison(band_delta_configs)
    plt.savefig('delta_comparison_all_bands.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: delta_comparison_all_bands.png")
    
    print("\n" + "=" * 80)
    print("All plots generated successfully!")
    print("=" * 80)

