import logging
import os
import random
import shutil
from pathlib import Path

from astropy.coordinates import SkyCoord

from modules import simulate_epic as sp
from modules import source as src
from modules.exposures import ExposureXMM

from set_of_xmm_simulations_multiple_sources import make_bkg, make_ccf, make_images, make_catalogue, merge_simput_files, move_files


logging.basicConfig(level=logging.INFO)

data_path = Path("./sims/testmos")
eband = "SOFT" # FULL, SOFT, HARD (see enums module)
pointing = SkyCoord(0, 0, unit="deg")
rollangle = 0.
exposure_time = 10000.

ccf_path = make_ccf(data_path)
os.environ["SAS_CCF"] = str(ccf_path.resolve())

xmmexp_mos1 = ExposureXMM(
    pointing,
    rollangle,
    exposure_time,
    detector="M1",
    expid="S001",
)
xmmexp_mos2 = ExposureXMM(
    xmmexp_mos1.pointing,
    xmmexp_mos1.rollangle,
    exposure_time,
    detector="M2",
    expid="S002",
)
xmmexp_pn = ExposureXMM(
    xmmexp_mos1.pointing,
    xmmexp_mos1.rollangle,
    exposure_time,
    detector="PN",
    expid="S003",
)

bkg_simput_file = make_bkg(xmmexp_mos1, eband)

# src_simput_file = f"{xmmexp_mos1.prefix}_src_source.simput"
# src.simput(
#     pointing.ra,
#     pointing.dec,
#     xmmexp_mos1.mjdref,
#     eband=eband,
#     flux=5e-14,
#     spec_file="xspec/age_lognlogs_spectrum.xcm",
#     output_file=src_simput_file,
# )

src_simput_file = make_catalogue(xmmexp_mos1, eband)

final_simput_file = merge_simput_files(
    src_simput_file, None, bkg_simput_file, xmmexp_mos1.prefix
)

xmm_mos1_evt_file = sp.run_xmm_simulation(
    xmmexp_mos1,
    final_simput_file,
    make_attitude=True,
    particle_bkg="low",
    split_bkg=False,
    badpixels=False,
)
xmm_mos2_evt_file = sp.run_xmm_simulation(
    xmmexp_mos2,
    final_simput_file,
    make_attitude=True,
    particle_bkg="low",
    split_bkg=False,
    badpixels=False,
)
xmm_pn_evt_file = sp.run_xmm_simulation(
    xmmexp_pn,
    final_simput_file,
    make_attitude=True,
    particle_bkg="low",
    split_bkg=False,
    badpixels=False,
)
make_images(xmmexp_mos1, xmm_mos1_evt_file)
make_images(xmmexp_mos2, xmm_mos2_evt_file)
make_images(xmmexp_pn, xmm_pn_evt_file)

move_files(xmmexp_mos1, data_path)
move_files(xmmexp_mos2, data_path)
move_files(xmmexp_pn, data_path)
