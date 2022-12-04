# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:16:01 2020

@author: A.Ruiz
"""
import numpy as np
import sixty
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from .enums import Eband


def image(ra, dec, fov, size=100, output_file=None):
    header = _make_wcs_header(ra, dec, fov, size)
    header = _add_simput_keywords(header)
    data = _make_image_data(size)

    hdu = fits.PrimaryHDU(data, header=header)
    hdulist = fits.HDUList([hdu])

    if output_file:
        hdulist.writeto(output_file, overwrite=True)

    return hdulist


def _make_wcs_header(ra, dec, fov, size):
    cenx = size / 2 - 0.5
    ceny = size / 2 - 0.5
    crpix1 = cenx + 1
    crpix2 = ceny + 1
    crval1 = ra.to(u.deg).value
    crval2 = dec.to(u.deg).value
    cdelt = fov.to(u.deg).value / size

    w = WCS(naxis=2)

    w.wcs.crpix = [crpix1, crpix2]
    w.wcs.cdelt = [cdelt, cdelt]
    w.wcs.crval = [crval1, crval2]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.radesys = "FK5"
    w.wcs.equinox = 2000.0

    return w.to_header()


def _add_simput_keywords(header):
    header["EXTNAME"] = "IMAGE"
    header["HDUCLASS"] = "HEASARC"
    header["HDUCLAS1"] = "SIMPUT"
    header["HDUCLAS2"] = "IMAGE"
    header["HDUVERS"] = "1.1.0"
    header["EXTVER"] = 1

    return header


def _make_image_data(size):
    data = np.zeros((size, size))
    data[:, :] = 1

    # rng = np.random.default_rng()
    # mask = rng.random(data.shape)
    # data[mask > 0.2] = 1
    # data = gaussian_filter(data, sigma=15, mode="wrap")
    # data = 1 - (data.max() - data)/(data.max() - data.min())

    return data


def simput(
    ra, dec, mjdref, image_file, spec_file, output_file, eband="SOFT", flux=1e-10
):
    eband = Eband[eband]
    
    sixty.run(
        "simputfile",
        RA=ra.to(u.deg).value,
        Dec=dec.to(u.deg).value,
        MJDREF=mjdref.mjd,
        Emin=eband.emin,
        Emax=eband.emax,
        srcFlux=flux,
        XSPECFile=spec_file,
        ImageFile=image_file,
        Simput=output_file,
        clobber=True,
    )
