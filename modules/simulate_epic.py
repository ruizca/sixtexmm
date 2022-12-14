# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:16:01 2020

@author: A.Ruiz
"""
from tempfile import NamedTemporaryFile

import numpy as np
import pxsas
from astropy.io import fits
from astropy.table import Table

from . import simulate_ccd
from .badpixels import BADPN, BADMOS

rng = np.random.default_rng()


def run_xmm_simulation(
    xmmexp, simput_file, particle_bkg=None, make_attitude=False, split_bkg=False, badpixels=False, suffix="objevlifilt.FIT"
):
    output_file_raw = f"{xmmexp.prefix}_raw.fits"
    output_file_evt = f"{xmmexp.prefix}_evt.fits"
    output_file_evt_xmm = f"{xmmexp.prefix}-{suffix}"

    if make_attitude:
        attitude_file = f"{xmmexp.prefix}_att.fits"
        xmmexp.attitude(output_file=attitude_file)

        attitude_file_xmm = f"{xmmexp.prefix}_xmm_att.fits"
        xmmexp.attitude_xmm(output_file=attitude_file_xmm)
    else:
        attitude_file = None

    for ccd in xmmexp.detector.ccds:
        run_xmm_simulation_ccd(
            xmmexp,
            ccd,
            simput_file,
            output_file_raw,
            output_file_evt,
            output_file_evt_xmm,
            attitude_file=attitude_file,
            include_gti=True,
            particle_bkg=particle_bkg,
            split_bkg=split_bkg,
        )

    _merge_and_fix_evt_files(xmmexp, output_file_evt_xmm)

    if badpixels:
        _badpixels(xmmexp, output_file_evt_xmm)

    if split_bkg:
        output_file_evt_src_xmm = f"src_{xmmexp.prefix}-{suffix}"
        _merge_and_fix_evt_files(xmmexp, output_file_evt_src_xmm)

        output_file_evt_bkg_xmm = f"bkg_{xmmexp.prefix}-{suffix}"
        _merge_and_fix_evt_files(xmmexp, output_file_evt_bkg_xmm)

    return output_file_evt_xmm


def run_xmm_simulation_ccd(
    xmmexp,
    ccd,
    simput_file,
    output_file_raw,
    output_file_evt,
    output_file_evt_xmm,
    attitude_file=None,
    include_gti=False,
    particle_bkg=None,
    split_bkg=False,
):
    sim_class = getattr(simulate_ccd, f"Sim{xmmexp.detector.long}")
    sim = sim_class(
        xmmexp,
        ccd,
        simput_file,
        output_file_raw,
        output_file_evt,
        output_file_evt_xmm,
        split_bkg,
    )
    sim.run(particle_bkg, attitude_file, include_gti)


def _merge_and_fix_evt_files(exposure, evt_file):
    _merge_evt_files(exposure.detector, evt_file)
    _fix_detcoords(exposure, evt_file)
    _fix_physcoords(exposure, evt_file)


def _merge_evt_files(detector, evt_file):
    pxsas.run(
        "evlistcomb",
        eventsets=" ".join(f"ccd{ccd['tag']}_{evt_file}" for ccd in detector.ccds),
        imagingset=evt_file,
        instrument=detector.type,
        othertables="STDGTI",
    )
    with fits.open(evt_file, "update") as hdu:
        hdu["EVENTS"].header["DSTYP1"] = "CCDNR"
        hdu["EVENTS"].header["DSTYP2"] = "TIME"
        hdu["EVENTS"].header["DSUNI2"] = "s"
        hdu["EVENTS"].header["DSVAL1"] = "1"
        hdu["EVENTS"].header["DSVAL2"] = "TABLE"
        hdu["EVENTS"].header["DSREF2"] = ":STDGTI01"

        for i, ccd in enumerate(detector.ccds[1:], 2):
            hdu["EVENTS"].header[f"{i}DSVAL1"] = f"{i}"
            hdu["EVENTS"].header[f"{i}DSREF2"] = f":STDGTI{ccd['tag']}"


def _fix_detcoords(xmmexp, evt_file):
    evt = Table.read(evt_file, hdu="EVENTS")
    evt.keep_columns(["RAWX", "RAWY", "CCDNR"])
    evt.meta = {}

    ra, dec, pa = xmmexp.startracker_pointing
    
    with NamedTemporaryFile("w", suffix=".fits") as tf:
        evt.write(tf.name, format="fits", overwrite=True)
        pxsas.run(
            "edet2sky",
            intab=tf.name,
            calinfostyle="user",
            instrument=xmmexp.detector.long,
            datetime="2021-09-29T00:00:00",
            scattra=ra.value,
            scattdec=dec.value,
            scattapos=pa,         
            inputunit="raw",
        )
        detcoords = Table.read(tf.name)

    with fits.open(evt_file, "update") as hdu:
        hdu["EVENTS"].data["DETX"] = np.round(detcoords["DETX"])
        hdu["EVENTS"].data["DETY"] = np.round(detcoords["DETY"])

        colnames = hdu["EVENTS"].columns.names

        for col in ["DETX", "DETY"]:
            ncol = colnames.index(col) + 1
            # hdu["EVENTS"].header[f"TLMIN{ncol}"] = ""
            # hdu["EVENTS"].header[f"TLMAX{ncol}"] = ""
            hdu["EVENTS"].header[f"TCRPX{ncol}"] = 0
            hdu["EVENTS"].header[f"TCDLT{ncol}"] = 0.05 / 3600


def _fix_physcoords(xmmexp, evt_file):
    ra, dec, pa = xmmexp.startracker_pointing

    pxsas.run(
        "attcalc",
        eventset=evt_file,
        refpointlabel="nom",
        attitudelabel="fixed",
        fixedra=ra.value,
        fixeddec=dec.value,
        fixedposangle=pa,
    )

def _badpixels(xmmexp, evt_file):
    if xmmexp.detector.type == "epn":
        bad = BADPN
    else:
        bad = BADMOS

    badpixels = Table()
    badpixels["RAWX"] = np.array(bad["rawxlist"].split(), dtype=np.int16)
    badpixels["RAWY"] = np.array(bad["rawylist"].split(), dtype=np.int16)
    badpixels["TYPE"] = np.array(bad["typelist"].split(), dtype=np.int16)
    badpixels["YEXTENT"] = np.array(bad["yextentlist"].split(), dtype=np.int16)

    badpixels["RAWX"].unit = "pixels"
    badpixels["RAWY"].unit = "pixels"
    badpixels["YEXTENT"].unit = "pixels"

    badpixels.meta["EXTNAME"] = "BADPIX"
    badpixels.meta["TELESCOP"] = "XMM"
    badpixels.meta["INSTRUME"] = xmmexp.detector.long
    badpixels.meta["EXP_ID"] = xmmexp.expid
    badpixels.meta["OBS_ID"] = xmmexp.obsid

    badpixtables = []
    for ccd in xmmexp.detector.ccds:
        badpixels.meta["CCDID"] = ccd["ccdid"]
        
        if xmmexp.detector.type == "epn":
            badpixels.meta["QUADRANT"] = ccd["quadrant"]
        else:
            badpixels.meta["CCDNODE"] = 0

        badpixels_hdu = fits.BinTableHDU(badpixels)
        primary_hdu = fits.PrimaryHDU()
        hdul = fits.HDUList([primary_hdu, badpixels_hdu])
        
        badpixset = f"ccd{ccd['tag']}_{xmmexp.prefix}-bad.fits"
        hdul.writeto(badpixset, overwrite=True)

        badpixtables.append(badpixset)

    pxsas.run(
        "ebadpixupdate",
        eventset=evt_file,
        badpixtables=" ".join(badpixtables),
        ccds=0,
    )
