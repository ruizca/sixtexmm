import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pxsas
from astropy.io import fits
from packaging.version import Version


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


def get_sixte_version():
    v = _get_version("sixte")

    return Version(v)


def _get_version(task):
    if task not in ["sixte", "simput"]:
        raise ValueError(f"Unknown task: {task}")

    output = subprocess.check_output(f"{task}version")
    v = output.split(b"\n")
    v = v[0].split()

    return v[-1].decode()
