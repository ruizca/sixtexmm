# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:16:01 2020

@author: A.Ruiz
"""
from functools import cached_property

import numpy as np
import pxsas
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.modeling.rotations import Rotation2D

from .enums import Detector, Filter

rng = np.random.default_rng()


class ExposureXMM:
    def __init__(
        self,
        pointing,
        rollangle,
        exposure_time,
        tstart=0,
        detector="PN",
        filter="Thin1",
        obsid="0000000000",
        expid="S001",
    ):
        self.pointing = self._set_pointing(pointing)
        self.rollangle = self._set_rollangle(rollangle)
        self.exposure_time = self._set_exposure_time(exposure_time)
        self.tstart = tstart

        self.detector = self._set_detector(detector)
        self.filter = self._set_filter(filter)

        self.fov = 0.5 * u.deg
        self.mjdref = Time(50814.0, format="mjd")
        self.obsid = obsid
        self.expid = obsid + expid[1:]
        self.expidstr = expid
        self.datamode = "IMAGING"
        self.submode = "PrimeFullWindow"

    def __repr__(self) -> str:
        return (
            "XMM-Newton Exposure:\n"
            f"Obs.ID. {self.obsid}, Exp.ID. {self.expidstr}\n"
            f"Detector: {self.detector.long}; Filter: {self.filter.name}\n"
            f"RA: {self.pointing.ra.deg:.04f} deg, "
            f"Dec: {self.pointing.dec.deg:.04f} deg, "
            f"PA: {self.rollangle:.02f} deg\n"
            f"Exposure time: {self.exposure_time/1000} ks"
        )

    @property
    def prefix(self):
        return f"P{self.obsid}{self.detector.name}{self.expidstr}"

    @property
    def shift_pointing(self):
        rot = Rotation2D(-self.rollangle)
        dra, ddec = rot(-75.6, -50.4)
        shift_ra = (self.pointing.ra + dra * u.arcsec).wrap_at(360 * u.deg)
        shift_dec = self.pointing.dec + ddec * u.arcsec

        return SkyCoord(shift_ra, shift_dec)

    @cached_property
    def startracker_pointing(self):
        rot = Rotation2D(-self.rollangle)
        dra, ddec = rot(*self.detector.boresight)
        shifted_ra = (
            self.pointing.ra + dra * u.arcsec / np.cos(self.pointing.dec)
        ).wrap_at(360 * u.deg)
        shifted_dec = self.pointing.dec + ddec * u.arcsec
        # shifted_ra = self.pointing.ra
        # shifted_dec = self.pointing.dec

        output = pxsas.run(
            "strbs",
            instrument=self.detector.long,
            ra=shifted_ra.value,
            dec=shifted_dec.value,
            apos=self.rollangle,
            bstoolsout="yes",
        )

        for s in output.split():
            if s.startswith("ra"):
                ra = float(s.split("=")[-1]) * u.deg

            if s.startswith("dec"):
                dec = float(s.split("=")[-1]) * u.deg

            if s.startswith("apos"):
                pa = s.split("=")[-1]
                pa = float(pa.replace("strbs:-", "")) * u.deg
                pa = Angle(pa).wrap_at(360 * u.deg).value

        return ra, dec, pa

    @staticmethod
    def _set_pointing(pointing):
        if pointing is None:
            ra = 360 * rng.random()
            dec = 180 * rng.random() - 90
            pointing = SkyCoord(ra, dec, unit="deg")

        return pointing

    @staticmethod
    def _set_rollangle(rollangle):
        if rollangle is None:
            rollangle = 360 * rng.random()

        return rollangle

    @staticmethod
    def _set_exposure_time(exposure_time):
        if exposure_time is None:
            exposure_time_set = [10000.0, 25000.0, 50000.0, 100000.0]
            # weights = [0.25, 0.5, 0.2, 0.05]  # [1.0, 0.0, 0.0, 0.0]
            exposure_time = rng.choice(exposure_time_set)#, p=weights)

        return exposure_time

    @staticmethod
    def _set_detector(detector):
        try:
            return Detector[detector]

        except KeyError:
            raise ValueError(f"Unknown detector: {detector}")

    @staticmethod
    def _set_filter(filter):
        try:
            return Filter[filter]

        except KeyError:
            raise ValueError(f"Unknown filter: {filter}")

    def attitude(self, dt=1, output_file=None):
        att = Table()
        att["Time"] = np.arange(
            self.tstart, self.tstart + self.exposure_time + dt, step=dt
        )
        att["RA"] = self.pointing.ra
        att["Dec"] = self.pointing.dec
        att["ROLLANG"] = 360 - self.rollangle

        att.meta["MJDREF"] = self.mjdref.mjd
        att.meta["TSTART"] = self.tstart
        att.meta["TSTOP"] = self.tstart + self.exposure_time
        att.meta["DETNAM"] = self.detector.long

        if output_file:
            att.write(output_file, format="fits", overwrite=True)

        return att

    def attitude_xmm(self, dt=1, output_file=None):
        ra, dec, pa = self.startracker_pointing

        att = Table()
        att["TIME"] = np.arange(0, self.exposure_time + dt, step=dt, dtype=np.float32)
        att["AHFRA"] = ra
        att["AHFDEC"] = dec
        att["AHFPA"] = pa  # self.rollangle
        # The OM values are wrong, but for the
        # moment we don't need the correct ones
        att["OMRA"] = ra
        att["OMDEC"] = dec
        att["OMPA"] = pa  # self.rollangle
        att["DAHFPNT"] = 0.0
        att["DOMPNT"] = 0.0
        att["DAHFOM"] = 0.0

        att.meta["EXTNAME"] = "ATTHK"
        att.meta["CREATOR"] = "SIXTE"

        if output_file:
            att.write(output_file, format="fits", overwrite=True)

        return att
