#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:57:30 2021

@author: ruizca
"""
import logging
import os
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from joblib import Parallel, delayed

from modules import simulate_epic as sp
from modules import simput
from modules import utils
from modules.exposures import ExposureXMM

rng = np.random.default_rng()


def main():
    logging.basicConfig(level=logging.INFO)
    
    n_sims = 1000
    n_jobs = 100
    eband = "SOFT"
    particle_bkg = "low"
    data_path = Path("./sims/statix")

    ccf_path = utils.make_ccf(data_path)
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


def simulate_xmm_exposure(
    id, data_path, eband="SOFT", particle_bkg="med", transient_flux=None, image=True, split_bkg=False
):
    # EPIC-pn exposure with random PA,
    # and exp. time randomly selected between 10, 25, 50 or 100 ks
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

    bkg_simput_file = simput.make_bkg(xmmexp, eband)

    # We use a higher value of the logN-logS normalization 
    # to obtain more detectable sources per observation
    src_simput_file = simput.make_catalogue(xmmexp, eband, max_depth=12, lgnlgs_norm=1.5)

    if transient_flux is not False:
        transient_simput_file = simput.make_transient(
            xmmexp, eband, flux=transient_flux, burst_duration=5000.0
        )
    else:
        transient_simput_file = None

    final_simput_file = simput.merge_simput_files(
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
        utils.make_images(xmmexp, xmm_evt_file, split_bkg)        

    utils.move_files(xmmexp, data_path)


if __name__ == "__main__":
    main()
