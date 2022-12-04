# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:16:01 2020

@author: A.Ruiz
"""
import numpy as np
import xspec as xs
from astropy import units as u
from astropy.coordinates import FK5
from astropy.io import fits
from astropy.table import Table
from astropy_healpix import HEALPix
from mocpy import MOC
from mocpy.mocpy import flatten_pixels

from .enums import Eband
from .source import make_lightcurve

rng = np.random.default_rng()


def simput(
    ra, dec, fov, output_file, eband="SOFT", fluxes=None, exptime=None,
):
    eband = Eband[eband]

    # For the moment we use the same spectrum for all sources
    catalogue = _make_catalogue(ra, dec, fov, eband, fluxes)
    spectrum = _make_spectrum()

    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(catalogue)
    hdul.append(spectrum)

    if exptime is not None:
        timing = _make_timing(exptime)
        for ext in timing:
            hdul.append(ext)

    hdul.writeto(output_file, overwrite=True)


def _make_catalogue(ra, dec, fov, eband, fluxes=None, fmin=1e-15, fmax=1e-10):
    if fluxes is None:
        fluxes = _random_fluxes(fov, fmin, fmax, eband.name, bright_sources=0)

    nsources = len(fluxes)
    coords = _random_coordinates(ra, dec, fov, nsources)

    catalogue = Table()
    catalogue["SRC_ID"] = np.arange(1, nsources + 1)
    catalogue["SRC_NAME"] = [f"C{i:05}" for i in range(1, nsources + 1)]
    catalogue["RA"] = coords.ra
    catalogue["DEC"] = coords.dec
    catalogue["IMGROTA"] = np.float32(0.0) * u.deg
    catalogue["IMGSCAL"] = np.float32(1.0)
    catalogue["E_MIN"] = np.float32(eband.emin) * u.keV
    catalogue["E_MAX"] = np.float32(eband.emax) * u.keV
    catalogue["FLUX"] = fluxes  # * u.erg / u.s / u.cm**2
    catalogue["SPECTRUM"] = "[SPECTRUM,1][NAME=='spec_0000000001']"
    catalogue["IMAGE"] = ""
    catalogue["TIMING"] = ""

    catalogue.meta["EXTNAME"] = "SRC_CAT"
    catalogue.meta["HDUCLASS"] = "HEASARC/SIMPUT"
    catalogue.meta["HDUCLAS1"] = "SRC_CAT"
    catalogue.meta["HDUVERS"] = "1.1.0"
    catalogue.meta["RADESYS"] = "FK5"
    catalogue.meta["EQUINOX"] = 2000.0

    hdu_catalogue = fits.table_to_hdu(catalogue)

    # TODO: is it possible to fix the problem with the units within astropy???
    hdu_catalogue.header["TUNIT9"] = "erg/s/cm**2"

    return hdu_catalogue


def _random_fluxes(fov, fmin, fmax, band="SOFT", bright_sources=0):
    def logn_logs(lgflux, band):
        # logN-logS results from Georgakakis et al. 2008 (I think)
        lgfb14 = {
            "FULL": 0.47,  # units 1e-14
            "HARD": 0.09,  # units 1e-14
            "SOFT": -0.20,  # units 1e-14
            "UHRD": -0.09,  # units 1e-14
            "VHRD": -0.09,
        }  # units 1e-14

        b1, b2, lgknorm = -1.5, -2.5, 1.5  # 0.0
        lgf14 = lgflux + 14
        lgknorm_prime = lgknorm + (b1 - b2) * lgfb14[band]

        mask = lgf14 > lgfb14[band]

        dnds = np.zeros_like(lgflux)
        dnds[mask] = lgknorm_prime + b2 * lgf14[mask]
        dnds[~mask] = lgknorm + b1 * lgf14[~mask]

        return dnds

    rng = np.random.default_rng()

    logs = np.linspace(np.log10(fmin), np.log10(fmax), num=101)
    logn = logn_logs(logs, band)

    area_fov = np.pi * (fov / 2) ** 2  # This should be in sq.deg.
    nsources = 10 ** logn[0] * area_fov.value  # This is not an integer

    # try:
    #     # We do a poisson realization to add more randomness
    nsources = rng.poisson(lam=nsources)
    # except ValueError:
    #     # This happens if nsources is too large,
    #     # so we use a gaussian approximation
    #     nsources = int(rng.normal(loc=nsources, scale=np.sqrt(nsources)))
    # nsources = round(nsources)

    # For sampling a flux distribution following the logN-logS,
    # I need the cumulative of the differential
    # cdf = 1 - logNlogS (if normalize to one in the interval)
    cdf = 1 - 10 ** (logn - logn[0])
    flux = 10 ** np.interp(rng.random(nsources), cdf, logs)

    if bright_sources:
        bright_lgflux = (-13 + 12.0) * rng.random(bright_sources) - 11.0
        flux = np.concatenate((flux, 10 ** bright_lgflux))

    return flux.astype(np.float32)


def _random_coordinates(ra, dec, fov, nsources, max_depth=12):
    # # We build a circular MOC covering the FoV and randomly
    # select n HEALPix cells from it. This way sources are uniformly
    # distributed in the FoV and we avoid overlaping sources
    hp = HEALPix(nside=2 ** max_depth, order="nested", frame=FK5())

    moc_fov = MOC.from_cone(ra, dec, fov / 2, max_depth=max_depth)
    hpcells_fov = flatten_pixels(moc_fov._interval_set._intervals, moc_fov.max_order)
    hpcells_src = rng.choice(hpcells_fov, nsources, replace=False)

    return hp.healpix_to_skycoord(hpcells_src.astype(int))


def _make_spectrum(model="powerlaw", params={1: 1.4, 2: 1.0}, name="spec_0000000001"):
    # model="phabs*powerlaw", params={1: 21.5, 2: 1.5, 3: 10.0}, name="spec_0000000001"
    xs.Xset.chatter = 0
    xs.Xset.xsect = "vern"
    xs.Xset.abund = "angr"
    xs.AllModels.setEnergies("0.1, 100.0, 1000")

    model = xs.Model(model, setPars=params)
    elimits = np.array(model.energies(0), dtype=np.float32)
    energy = elimits[:-1] + np.diff(elimits) / 2
    flux = np.array(model.values(0), dtype=np.float32)

    spectrum = Table()
    spectrum["ENERGY"] = [energy] * u.keV
    spectrum["FLUXDENSITY"] = [flux] * u.photon / u.s / u.cm ** 2 / u.keV
    spectrum["NAME"] = name

    spectrum.meta["EXTNAME"] = "SPECTRUM"
    spectrum.meta["HDUCLASS"] = "HEASARC/SIMPUT"
    spectrum.meta["HDUCLAS1"] = "SPECTRUM"
    spectrum.meta["HDUVERS"] = "1.1.0"
    spectrum.meta["EXTVER"] = 1

    return fits.table_to_hdu(spectrum)


def _make_timing(exptime):
    extensions = []
    durations = [1 / 2, 1 / 5, 1 / 10, 1 / 20]

    for i, d in enumerate(durations, start=1):
        lc = make_lightcurve(exptime, burst_duration=exptime * d)
        #lc = _lightcurve_burst_constant(exptime, exptime * d)
        name = f"TIM_{i:03}"
        extensions.append(_timing_table(lc, name))

    return extensions


def _timing_table(lc, name):
    timing = Table()
    timing["TIME"] = lc["time"] * u.s
    timing["FLUX"] = lc["counts"]

    timing.meta["EXTNAME"] = name
    timing.meta["HDUCLASS"] = "HEASARC/SIMPUT"
    timing.meta["HDUCLAS1"] = "LIGHTCURVE"
    timing.meta["HDUVERS"] = "1.1.0"
    timing.meta["EXTVER"] = 1
    timing.meta["MJDREF"] = 50814.0
    timing.meta["TIMEZERO"] = 0.0
    timing.meta["PERIODIC"] = 0

    return fits.table_to_hdu(timing)
