import logging
import os
from pathlib import Path

from astropy.coordinates import SkyCoord

from modules import simulate_epic as sp
from modules import simput, utils
from modules.exposures import ExposureXMM


logging.basicConfig(level=logging.INFO)

data_path = Path("./sims/fullobs")
eband = "SOFT" # FULL, SOFT, HARD (see enums module)
pointing = SkyCoord(0, 0, unit="deg")
rollangle = None  # random roll angle
exposure_time = 10000.

ccf_path = utils.make_ccf(data_path)
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

bkg_simput_file = simput.make_bkg(xmmexp_mos1, eband)
src_simput_file = simput.make_catalogue(xmmexp_mos1, eband)

final_simput_file = simput.merge_simput_files(
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
utils.make_images(xmmexp_mos1, xmm_mos1_evt_file)
utils.make_images(xmmexp_mos2, xmm_mos2_evt_file)
utils.make_images(xmmexp_pn, xmm_pn_evt_file)

utils.move_files(xmmexp_mos1, data_path)
utils.move_files(xmmexp_mos2, data_path)
utils.move_files(xmmexp_pn, data_path)
