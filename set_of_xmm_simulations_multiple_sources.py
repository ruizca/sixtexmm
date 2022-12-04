#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:57:30 2021

@author: ruizca
"""
import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pxsas
import sixty
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from joblib import Parallel, delayed

from modules import background as bkg
from modules import catalogue
from modules import simulate_epic as sp
from modules import source as src
from modules.exposures import ExposureXMM

rng = np.random.default_rng()


def main():
    logging.basicConfig(level=logging.INFO)
    
    n_sims = 1000
    n_jobs = 100
    eband = "SOFT"
    particle_bkg = "low"
    data_path = Path("./sims/ruiz2023")

    ccf_path = make_ccf(data_path)
    os.environ["SAS_CCF"] = str(ccf_path.resolve())

    ids = list(range(n_sims))

    if len(ids) > 1:
        Parallel(n_jobs=n_jobs)(
            delayed(simulate_xmm_exposure)(
                id, data_path, eband, particle_bkg, image=False, split_bkg=True
            ) for id in ids
        )
    else:
        simulate_xmm_exposure(ids[0], data_path, eband, particle_bkg, image=False, split_bkg=True)


        
def make_ccf(path, date=None):
    if date is None:
        date = datetime.now()
        date = date.isoformat()

    if not path.exists():
        path.mkdir(parents=True)

    cif_path = path.joinpath("nowccf.cif")
    
    pxsas.run(
        "cifbuild",
        calindexset=str(cif_path),
        withobservationdate=True,
        observationdate=date,
    )
    return cif_path


def simulate_xmm_exposure(
    id, data_path, eband="SOFT", particle_bkg="med", transient_flux=None, image=True, split_bkg=False
):
    # Exposure with random pointing and PA,
    # and exp. time randomly selected bewtween 10, 25, 50 and 100 ks
    pointing = SkyCoord(10, 0, unit="deg")
    rollangle = None
    exposure_time = None
    xmmexp = ExposureXMM(
        pointing,
        rollangle,
        exposure_time,
        detector="PN",
        filter="Thin1",
        obsid=f"{id:010}",
    )

    bkg_simput_file = make_bkg(xmmexp, eband)
    src_simput_file = make_catalogue(xmmexp, eband)

    if transient_flux is not False:
        transient_simput_file = make_transient(
            xmmexp, eband, flux=transient_flux, burst_duration=5000.0
        )
    else:
        transient_simput_file = None

    final_simput_file = merge_simput_files(
        src_simput_file, transient_simput_file, bkg_simput_file, xmmexp.prefix
    )

    xmm_evt_file = sp.run_xmm_simulation(
        xmmexp,
        final_simput_file,
        make_attitude=True,
        particle_bkg=particle_bkg,
        split_bkg=split_bkg,
    )

    if image:
        make_images(xmmexp, xmm_evt_file, split_bkg)        

    move_files(xmmexp, data_path)


def make_bkg(xmmexp, eband):
    # Create background image,
    bkgimg_file = f"{xmmexp.prefix}_background_constant_img.fits"
    bkgimg_angular_size = 2*xmmexp.fov
    bkg.image(
        xmmexp.pointing.ra,
        xmmexp.pointing.dec,
        bkgimg_angular_size,
        size=600,
        output_file=bkgimg_file,
    )

    bkgimg_arcmin = bkgimg_angular_size.to(u.arcmin).value
    bkgimg_area = bkgimg_arcmin**2
    flux = 3e-15 * bkgimg_area  # The 3e-15 value is the 0.5-2 keV Xspec flux using the athenabkg.xcm model

    # Create background simput file,
    bkg_simput_file = f"{xmmexp.prefix}_background_athena.simput"
    bkg.simput(
        xmmexp.pointing.ra,
        xmmexp.pointing.dec,
        xmmexp.mjdref,
        eband=eband,
        flux=flux,
        image_file=bkgimg_file,
        spec_file="xspec/athenabkg.xcm",
        output_file=bkg_simput_file,
    )
    return bkg_simput_file


def make_catalogue(xmmexp, eband):
    # Create simput file for catalogue of sources
    src_simput_file = f"{xmmexp.prefix}_catalogue.simput"
    catalogue.simput(
        xmmexp.pointing.ra,
        xmmexp.pointing.dec,
        xmmexp.fov,
        src_simput_file,
        eband
    )
    return src_simput_file


def make_transient(xmmexp, eband, flux, burst_duration):
    # Create lightcurve of transient source for this exposure
    lightcurve_file = f"{xmmexp.prefix}_lightcurve_tophat_burst5ks.dat"
    src.make_lightcurve(
        xmmexp.exposure_time, output_file=lightcurve_file, burst_duration=burst_duration
    )

    # Generate random position within FoV
    position_angle = 360 * rng.random() * u.deg
    separation = xmmexp.fov * rng.random() / 3
    transient_coord = xmmexp.pointing.directional_offset_by(position_angle, separation)
    # transient_coord = xmmexp.pointing

    if not flux:
        flux = random.choice([1e-15, 1e-14, 1e-13])

    # Create simput file for the transient
    transient_simput_file = f"{xmmexp.prefix}_transient_source.simput"
    src.simput(
        transient_coord.ra,
        transient_coord.dec,
        xmmexp.mjdref,
        eband=eband,
        flux=flux,
        name="T00001",
        spec_file="xspec/age_lognlogs_spectrum.xcm",
        lightcurve_file=lightcurve_file,
        output_file=transient_simput_file,
    )

    return transient_simput_file


def merge_simput_files(src_file, transient_file, bkg_file, prefix):
    # Merge catalogue and background simput files
    if transient_file is not None:
        sources_simput_file = f"{prefix}_sources.simput"

        sixty.run(
            "simputmerge",
            Infile1=src_file,
            Infile2=transient_file,
            Outfile=sources_simput_file,
            FetchExtensions="yes",
        )
    else:
        sources_simput_file = src_file

    final_simput_file = f"{prefix}_final.simput"
    sixty.run(
        "simputmerge",
        Infile1=sources_simput_file,
        Infile2=bkg_file,
        Outfile=final_simput_file,
        FetchExtensions="yes",
    )
    return final_simput_file


def make_images(xmmexp, xmm_evt_file, split_bkg=False):
    # Create fits image
    image_file = f"{xmmexp.prefix}_xmm_img.fits"
    _make_image(xmm_evt_file, image_file)
    fits.setval(image_file, "EXPOSURE", value=xmmexp.exposure_time, ext=0)

    if split_bkg:
        xmm_src_evt_file = f"src_{xmm_evt_file}"
        image_file = f"{xmmexp.prefix}_src_xmm_img.fits"
        _make_image(xmm_src_evt_file, image_file)
        fits.setval(image_file, "EXPOSURE", value=xmmexp.exposure_time, ext=0)

        xmm_bkg_evt_file = f"bkg_{xmm_evt_file}"
        image_file = f"{xmmexp.prefix}_bkg_xmm_img.fits"
        _make_image(xmm_bkg_evt_file, image_file)
        fits.setval(image_file, "EXPOSURE", value=xmmexp.exposure_time, ext=0)


def _make_image(xmm_evt_file, image_file):
    pxsas.run(
        "evselect",
        table=xmm_evt_file,
        imageset=image_file,
        xcolumn="X",
        ycolumn="Y",
        ximagesize=600,
        yimagesize=600,
        withimagedatatype="true",
        imagedatatype="Real32",
        squarepixels="true",
        imagebinning="imageSize",
        withimageset="Y",
        writedss="true",
        keepfilteroutput="false",
        updateexposure="true",
    )


def move_files(xmmexp, data_path):
    # Move generated files to the data folder
    cwd = Path.cwd()
    exposure_path = data_path.joinpath(xmmexp.obsid)

    if not exposure_path.exists():
        exposure_path.mkdir(parents=True)

    for simfile in cwd.glob(f"*{xmmexp.prefix}*"):
        shutil.move(simfile, exposure_path.joinpath(simfile.name))


if __name__ == "__main__":
    main()
