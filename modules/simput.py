import random

import numpy as np
import sixty
from astropy import units as u

from . import background as bkg
from . import catalogue
from . import source as src

rng = np.random.default_rng()


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


def make_transient(xmmexp, eband, flux, burst_duration, coords=None):
    # Create lightcurve of transient source for this exposure
    lightcurve_file = f"{xmmexp.prefix}_lightcurve_tophat_burst5ks.dat"
    src.make_lightcurve(
        xmmexp.exposure_time, output_file=lightcurve_file, burst_duration=burst_duration
    )

    if not coords:
        # Generate random position within FoV
        position_angle = 360 * rng.random() * u.deg
        separation = xmmexp.fov * rng.random() / 3
        transient_coord = xmmexp.pointing.directional_offset_by(position_angle, separation)
    else:
        transient_coord = coords

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