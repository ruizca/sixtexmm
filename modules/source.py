# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:16:01 2020

@author: A.Ruiz
"""
import numpy as np
import sixty
from astropy import units as u
from astropy.table import Table

from .enums import Eband


def simput(
    ra,
    dec,
    mjdref,
    spec_file,
    output_file,
    lightcurve_file=None,
    eband="SOFT",
    flux=1e-10,
    name=None,
):
    eband = Eband[eband]

    kwargs = {}
    if lightcurve_file:
        kwargs["LCFile"] = lightcurve_file

    if name:
        kwargs["Src_Name"] = name

    sixty.run(
        "simputfile",
        RA=ra.to(u.deg).value,
        Dec=dec.to(u.deg).value,
        XSPECFile=spec_file,
        MJDREF=mjdref.mjd,
        Emin=eband.emin,
        Emax=eband.emax,
        srcFlux=flux,
        Simput=output_file,
        clobber=True,
        **kwargs,
    )


def make_lightcurve(exposure_time, output_file=None, mode="burst_constant", **kwargs):
    if mode == "burst_constant":
        lc = _lightcurve_burst_constant(exposure_time, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if output_file is not None:
        lc.write(output_file, format="ascii.no_header", overwrite=True)

    return lc


def _lightcurve_burst_constant(exposure_time, burst_duration=0):
    if burst_duration > exposure_time:
        raise ValueError("The duration of the burst is larger than the exposure time!")

    lc = Table()
    lc["time"] = np.linspace(0, exposure_time, num=1000)
    lc["counts"] = 0.0

    burst_start = (exposure_time - burst_duration) / 2
    burst_end = (exposure_time + burst_duration) / 2

    mask = np.logical_and(lc["time"] >= burst_start, lc["time"] <= burst_end)
    lc["counts"][mask] = 1.0

    return lc
