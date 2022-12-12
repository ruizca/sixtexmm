import logging
import os
from pathlib import Path

from astropy.coordinates import SkyCoord

from modules import simulate_epic as sp
from modules import simput, utils
from modules.exposures import ExposureXMM


logging.basicConfig(level=logging.INFO)


data_path = Path("./sims/transient")
eband = "SOFT" # FULL, SOFT, HARD (see enums module)
pointing = SkyCoord(0, 0, unit="deg")
rollangle = 90
exposure_time = 100000.

ccf_path = utils.make_ccf(data_path)
os.environ["SAS_CCF"] = str(ccf_path.resolve())

xmmexp_pn = ExposureXMM(
    pointing,
    rollangle,
    exposure_time,
    detector="PN",
    expid="S003",
)

bkg_simput_file = simput.make_bkg(xmmexp_pn, eband)
transient_simput_file = simput.make_transient(
    xmmexp_pn, eband, flux=1.2e-14, burst_duration=5000.0, coords=xmmexp_pn.pointing
)
final_simput_file = simput.merge_simput_files(
    transient_simput_file, None, bkg_simput_file, xmmexp_pn.prefix
)

xmm_pn_evt_file = sp.run_xmm_simulation(
    xmmexp_pn,
    final_simput_file,
    make_attitude=True,
    particle_bkg="low",
    split_bkg=False,
    badpixels=True,
)

utils.make_images(xmmexp_pn, xmm_pn_evt_file)
utils.move_files(xmmexp_pn, data_path)
