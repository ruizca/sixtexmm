
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:16:01 2020

@author: A.Ruiz
"""
from enum import Enum


class Detector(Enum):
    M1 = ("EMOS1", "emos")
    M2 = ("EMOS2", "emos")
    PN = ("EPN", "epn")

    def __init__(self, long, type):
        self.long = long
        self.type = type

    @property
    def ccds(self):
        if self.type == "epn":
            ccds = (
                {"tag": "01", "quadrant": 0, "ccdid": 0},
                {"tag": "02", "quadrant": 0, "ccdid": 1},
                {"tag": "03", "quadrant": 0, "ccdid": 2},
                {"tag": "04", "quadrant": 1, "ccdid": 0},
                {"tag": "05", "quadrant": 1, "ccdid": 1},
                {"tag": "06", "quadrant": 1, "ccdid": 2},
                {"tag": "07", "quadrant": 2, "ccdid": 0},
                {"tag": "08", "quadrant": 2, "ccdid": 1},
                {"tag": "09", "quadrant": 2, "ccdid": 2},
                {"tag": "10", "quadrant": 3, "ccdid": 0},
                {"tag": "11", "quadrant": 3, "ccdid": 1},
                {"tag": "12", "quadrant": 3, "ccdid": 2},
            )
        else:
            ccds = (
                {"tag": "01", "ccdid": 1},
                {"tag": "02", "ccdid": 2},
                {"tag": "03", "ccdid": 3},
                {"tag": "04", "ccdid": 4},
                {"tag": "05", "ccdid": 5},
                {"tag": "06", "ccdid": 6},
                {"tag": "07", "ccdid": 7},
            )

        return ccds

    @property
    def boresight(self):
        if self.long == "EPN":
            boresight = (-21.0, 67.0)

        elif self.long == "EMOS1":
            boresight = (5.85, -10.2)

        elif self.long == "EMOS2":
            boresight = (-62.8, -24.2)

        return boresight

    @property
    def npixels(self):
        if self.type == "epn":
            npixels = len(self.ccds) * 64 * 200
        else:
            npixels = len(self.ccds) * 600 * 600

        return npixels


class Filter(Enum):
    Thin1 = ("Thin1", "thin")
    Thin2 = ("Thin2", "thin")
    Medium = ("Medium", "med")
    Thick = ("Thick", "thick")

    def __init__(self, long, sixte):
        self.long = long
        self.sixte = sixte


class Eband(Enum):
    FULL = (0.5, 10.0)
    SOFT = (0.5, 2.0)
    HARD = (2.0, 10.0)

    def __init__(self, emin, emax):
        self.emin = emin
        self.emax = emax
