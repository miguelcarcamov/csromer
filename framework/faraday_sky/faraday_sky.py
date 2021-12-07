from astropy.io import fits
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as un
from typing import Union, List
import numpy as np
import h5py
import pathlib


class FaradaySky:
    def __init__(self, filename=None, nside=512, ordering="nested"):
        self.nside = nside
        self.ordering = ordering
        if filename is None:
            self.filename = "./faradaysky/faraday2020v2.hdf5"
        else:
            self.filename = filename
        self.extension = pathlib.Path(self.filename).suffix
        if self.extension == ".hdf5":
            hf = h5py.File(self.filename, 'r')
            self.data = (np.array(hf.get("faraday_sky_mean")), np.array(hf.get("faraday_sky_std")))
            hf.close()
        elif self.extension == ".fits":
            with fits.open(self.filename) as hdul:
                self.data = (hdul[1].data["faraday_sky_mean"], hdul[1].data["faraday_sky_std"])
        else:
            raise ValueError("The extension is not HDF5 or FITS")

        if self.nside is not None and self.ordering is not None:
            self.hp = HEALPix(nside=nside, order=ordering, frame=Galactic())

    def galactic_rm(self, ra: Union[List[Quantity], Quantity, str] = None,
                    dec: Union[List[Quantity], Quantity, str] = None, frame="icrs",
                    use_bilinear_interpolation: bool = False):
        if isinstance(ra, Quantity):
            ra = ra.to(un.rad)
        elif isinstance(ra, str):
            ra = Quantity(ra)
        else:
            raise TypeError("Not valid type")

        if isinstance(dec, Quantity):
            dec = dec.to(un.rad)
        elif isinstance(ra, str):
            dec = Quantity(dec)
        else:
            raise TypeError("Not valid type")

        coord = SkyCoord(ra=ra, dec=dec, frame=frame)

        if use_bilinear_interpolation:
            rm_value_mean = self.hp.interpolate_bilinear_skycoord(coord, self.data[0]) * un.rad / un.m ** 2
            rm_value_std = self.hp.interpolate_bilinear_skycoord(coord, self.data[1]) * un.rad / un.m ** 2
        else:
            healpix_idx = self.hp.skycoord_to_healpix(coord)
            rm_value_mean = self.data[0][healpix_idx] * un.rad / un.m ** 2
            rm_value_std = self.data[1][healpix_idx] * un.rad / un.m ** 2

        return rm_value_mean, rm_value_std

    def galactic_rm_image(self, fitsfile: Union[str, fits.HDUList] = None, use_bilinear_interpolation: bool = False):
        if isinstance(fitsfile, str):
            hdul = fits.open(fitsfile)[0]
        else:
            hdul = fitsfile

        if hdul.header["NAXIS"] > 2:
            w = WCS(hdul.header, naxis=(1, 2))
        else:
            w = WCS(hdul.header)

        m = hdul.header["NAXIS1"]
        n = hdul.header["NAXIS2"]
        frame = hdul.header["RADESYS"].lower()
        x = np.arange(0, m, 1)
        y = np.arange(0, n, 1)
        xx, yy = np.meshgrid(x, y)

        ra, dec = w.all_pix2world(xx, yy, 0) * un.deg

        rm_flattened = self.galactic_rm(ra=ra.ravel(), dec=dec.ravel(), frame=frame,
                                        use_bilinear_interpolation=use_bilinear_interpolation)
        rm_mean = rm_flattened[0].reshape(m, n)
        rm_std = rm_flattened[1].reshape(m, n)
        return rm_mean, rm_std
