import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Union

import astropy.units as un
import h5py
import numpy as np
from astropy.coordinates import Galactic, SkyCoord
from astropy.io import fits
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy_healpix import HEALPix


@dataclass(init=True, repr=True)
class FaradaySky:
    filename: str = None
    nside: int = None
    ordering: str = None
    extension: str = field(init=False, default=None)
    data: Tuple[np.ndarray, np.ndarray] = field(init=False, default=None)
    hp: HEALPix = field(init=False, default=None)

    def __post_init__(self):

        if self.nside is None:
            self.nside = 512

        if self.ordering is None:
            self.ordering = "ring"

        if self.filename is None:
            self.filename = (
                pathlib.Path(__file__).parent.resolve() / "./faraday_sky_files/faraday2020v2.hdf5"
            )

        self.extension = pathlib.Path(self.filename).suffix
        if self.extension == ".hdf5":
            hf = h5py.File(self.filename, "r")
            self.data = (
                np.array(hf.get("faraday_sky_mean")),
                np.array(hf.get("faraday_sky_std")),
            )
            hf.close()
        elif self.extension == ".fits":
            with fits.open(self.filename) as hdul:
                self.data = (
                    hdul[1].data["faraday_sky_mean"],
                    hdul[1].data["faraday_sky_std"],
                )
                if "ORDERING" in hdul[0].header:
                    self.ordering = hdul[0].header["ORDERING"]

                if "NSIDE" in hdul[0].header:
                    self.nside = hdul[0].header["ORDERING"]
        else:
            raise ValueError("The extension is not HDF5 or FITS")

        if self.nside is not None and self.ordering is not None:
            self.hp = HEALPix(nside=self.nside, order=self.ordering, frame=Galactic())

    def galactic_rm(
        self,
        ra: Union[List[Quantity], Quantity, str] = None,
        dec: Union[List[Quantity], Quantity, str] = None,
        frame="icrs",
        use_bilinear_interpolation: bool = False,
    ):
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
            rm_value_mean = (
                self.hp.interpolate_bilinear_skycoord(coord, self.data[0]) * un.rad / un.m**2
            )
            rm_value_std = (
                self.hp.interpolate_bilinear_skycoord(coord, self.data[1]) * un.rad / un.m**2
            )
        else:
            healpix_idx = self.hp.skycoord_to_healpix(coord)
            rm_value_mean = self.data[0][healpix_idx] * un.rad / un.m**2
            rm_value_std = self.data[1][healpix_idx] * un.rad / un.m**2

        return rm_value_mean, rm_value_std

    def galactic_rm_image(
        self,
        fitsfile: Union[str, fits.HDUList, fits.PrimaryHDU, fits.Header] = None,
        use_bilinear_interpolation: bool = False,
    ):
        if isinstance(fitsfile, str):
            hdul = fits.open(fitsfile)[0]
            header = hdul.header
        elif isinstance(fitsfile, fits.HDUList):
            header = fitsfile[0].header
        elif isinstance(fitsfile, fits.PrimaryHDU):
            header = fitsfile.header
        else:
            header = fitsfile

        w = WCS(header, naxis=2)

        m = header["NAXIS1"]
        n = header["NAXIS2"]
        frame = header["RADESYS"].lower()
        x = np.arange(0, m, 1)
        y = np.arange(0, n, 1)
        xx, yy = np.meshgrid(x, y)

        skycoord = w.array_index_to_world(xx, yy)

        rm_flattened = self.galactic_rm(
            ra=skycoord.ra.ravel(),
            dec=skycoord.dec.ravel(),
            frame=frame,
            use_bilinear_interpolation=use_bilinear_interpolation,
        )
        rm_mean = rm_flattened[0].reshape(n, m)
        rm_std = rm_flattened[1].reshape(n, m)

        rm_mean_field = np.mean(rm_mean)
        rm_uncertainty_field = np.mean(rm_std)
        print(
            "The Galactic RM in the field is {0:.2f} \u00B1 {1:.2f}".format(
                rm_mean_field.value, rm_uncertainty_field
            )
        )
        return rm_mean, rm_std
