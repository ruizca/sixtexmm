import logging
import os
from pathlib import Path

from astropy.coordinates import SkyCoord

from modules import simulate_epic as sp
from modules import simput, utils
from modules import source as src
from modules.exposures import ExposureXMM


logging.basicConfig(level=logging.INFO)


data_path = Path("./sims/source")
eband = "FULL" # FULL, SOFT, HARD (see enums module)
pointing = SkyCoord(0, 0, unit="deg")
rollangle = 0
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

bkg_simput_file = simput.make_bkg(xmmexp_mos1, eband)

src_simput_file = f"{xmmexp_mos1.prefix}_src_source.simput"
src.simput(
    pointing.ra,
    pointing.dec,
    xmmexp_mos1.mjdref,
    eband=eband,
    flux=1e-12,
    spec_file="xspec/age_lognlogs_spectrum.xcm",
    output_file=src_simput_file,
)
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
    instrument_dir="/storage/sixte_instrument_files/xmm/",
    n_threads=7,  # Set to 1 for no parallelization
)

utils.make_images(xmmexp_mos1, xmm_mos1_evt_file)
utils.move_files(xmmexp_mos1, data_path)
