# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:16:01 2020

@author: A. Ruiz
"""
from pathlib import Path

import numpy as np
import sixty
from astropy.io import fits
from astropy.table import Table


class SimCCD:
    instrument_dir = sixty._sixte_dir.joinpath("share", "sixte", "instruments", "xmm")

    def __init__(
        self,
        xmmexp,
        ccd,
        simput_file,
        output_file_raw,
        output_file_evt,
        output_file_evt_xmm,
        split_bkg=False,
    ):
        self.xmmexp = xmmexp
        self.ccd = ccd
        self.simput_file = simput_file
        self.output_files = self._set_output_files(
            output_file_raw, output_file_evt, output_file_evt_xmm, split_bkg
        )

    def _set_output_files(self, raw, evt, evt_xmm, split_bkg):
        files = {
            "raw": f"ccd{self.ccd['tag']}_{raw}",
            "evt": f"ccd{self.ccd['tag']}_{evt}",
            "evt_xmm": f"ccd{self.ccd['tag']}_{evt_xmm}",
        }

        if split_bkg:
            files["src"] = {
                "raw": f"ccd{self.ccd['tag']}_src_{raw}",
                "evt": f"ccd{self.ccd['tag']}_src_{evt}",
                "evt_xmm": f"ccd{self.ccd['tag']}_src_{evt_xmm}",
            }
            files["bkg"] = {
                "raw": f"ccd{self.ccd['tag']}_bkg_{raw}",
                "evt": f"ccd{self.ccd['tag']}_bkg_{evt}",
                "evt_xmm": f"ccd{self.ccd['tag']}_bkg_{evt_xmm}",
            }

        return files

    def run(self, particle_bkg=None, attitude_file=None, include_gti=False):
        xml_file = self._set_xml_file(particle_bkg)
        self._runsixt(xml_file, attitude_file)
        self._add_pointing()
        self._epic_events(
            self.output_files["evt"], self.output_files["evt_xmm"], include_gti
        )

        if "bkg" in self.output_files:
            self._split_bkg(include_gti)

    def _runsixt(self, xml_file, attitude_file):
        sixte_parameters = self._set_sixte_parameters(xml_file, attitude_file)
        sixty.run("runsixt", **sixte_parameters)

    def _set_sixte_parameters(self, xml_file, attitude_file):
        parameters = {
            "Exposure": self.xmmexp.exposure_time,
            "MJDREF": self.xmmexp.mjdref.mjd,
            "RawData": self.output_files["raw"],
            "EvtFile": self.output_files["evt"],
            "XMLFile": xml_file,
            "Simput": self.simput_file,
            "clobber": True,
        }
        if attitude_file is None:
            attitude_parameters = {
                "RA": self.xmmexp.pointing.ra.value,
                "Dec": self.xmmexp.pointing.dec.value,
                "rollangle": self.xmmexp.rollangle,
            }
        else:
            attitude_parameters = {
                "Attitude": attitude_file,
            }

        parameters.update(attitude_parameters)

        return parameters

    def _add_pointing(self):
        with fits.open(self.output_files["evt"], "update") as hdu:
            hdu["EVENTS"].header["RA_PNT"] = self.xmmexp.pointing.ra.value
            hdu["EVENTS"].header["DEC_PNT"] = self.xmmexp.pointing.dec.value

    def _get_bkg_id(self):
        sources = Table.read(self.simput_file, hdu=1)
        return sources["SRC_ID"][-1]

    def _src_evt_file(self, bkg_id):
        evt_path = Path(self.output_files["evt"])
        evt_src_path = Path(self.output_files["src"]["evt"])
        evt_src_path.write_bytes(evt_path.read_bytes())

        with fits.open(evt_src_path, "update") as hdu:
            evtlist = hdu["EVENTS"].data
            src_id = evtlist.field("SRC_ID")[:, 0]

            src_mask = np.logical_and(src_id >= 0, src_id != bkg_id)
            hdu["EVENTS"].data = evtlist[src_mask]

    def _bkg_evt_file(self, bkg_id):
        evt_path = Path(self.output_files["evt"])
        evt_bkg_path = Path(self.output_files["bkg"]["evt"])
        evt_bkg_path.write_bytes(evt_path.read_bytes())

        with fits.open(evt_bkg_path, "update") as hdu:
            evtlist = hdu["EVENTS"].data
            src_id = evtlist.field("SRC_ID")[:, 0]

            bkg_mask = np.logical_or(src_id < 0, src_id == bkg_id)
            hdu["EVENTS"].data = evtlist[bkg_mask]

    @staticmethod
    def _get_particle_bkg_str(particle_bkg, mode="cf"):
        # mode: "diehl", "flat", "cf"
        if particle_bkg:
            if particle_bkg not in ["low", "med", "high"]:
                raise ValueError(f"Particle background unknown: {particle_bkg}")

            particle_bkg_str = f"_{particle_bkg}{mode}bkg"
        else:
            particle_bkg_str = ""

        return particle_bkg_str


class SimEPN(SimCCD):
    def _set_xml_file(self, particle_bkg):
        particle_bkg_str = self._get_particle_bkg_str(particle_bkg)
        instrument_dir = self.instrument_dir.joinpath("epicpn")

        xml_file = instrument_dir.joinpath(
            f"fullframe_ccd{self.ccd['tag']}_{self.xmmexp.filter.sixte}filter{particle_bkg_str}.xml"
        )

        return xml_file

    def _epic_events(self, evt, evt_xmm, include_gti):
        sixty.run(
            "epicpn_events", EvtFile=evt, EPICpnEventList=evt_xmm, clobber=True,
        )
        self._fix_evt_header(evt_xmm)

        if include_gti:
            self._include_gti(evt, evt_xmm)

    def _fix_evt_header(self, evt_xmm):
        with fits.open(evt_xmm, "update") as hdu:
            hdu[0].header["CREATOR"] = "SIXTE"
            hdu[0].header["OBSERVER"] = "SIXTE"
            hdu[0].header["DATAMODE"] = self.xmmexp.datamode
            hdu[0].header["SUBMODE"] = self.xmmexp.submode
            hdu[0].header["FILTER"] = self.xmmexp.filter.long
            hdu[0].header["OBS_ID"] = self.xmmexp.obsid
            hdu[0].header["EXP_ID"] = self.xmmexp.expid
            hdu[0].header["EXPIDSTR"] = self.xmmexp.expidstr
            hdu[0].header["RA_PNT"] = self.xmmexp.pointing.ra.deg
            hdu[0].header["DEC_PNT"] = self.xmmexp.pointing.dec.deg
            hdu[0].header["PA_PNT"] = self.xmmexp.rollangle
            hdu[0].header["RA_NOM"] = self.xmmexp.pointing.ra.deg
            hdu[0].header["DEC_NOM"] = self.xmmexp.pointing.dec.deg
            hdu["EVENTS"].header["CCDID"] = self.ccd["ccdid"]
            hdu["EVENTS"].header["QUADRANT"] = self.ccd["quadrant"]

    def _include_gti(self, evt, evt_xmm):
        stdgti = Table.read(evt, hdu=2)

        with fits.open(evt_xmm, "update") as hdu:
            hdu.append(fits.table_to_hdu(stdgti))
            hdu["STDGTI"].header["CCDID"] = self.ccd["ccdid"]
            hdu["STDGTI"].header["QUADRANT"] = self.ccd["quadrant"]
            hdu["STDGTI"].header["INSTRUME"] = self.xmmexp.detector.long

    def _split_bkg(self, include_gti):
        bkg_id = self._get_bkg_id()
        self._src_evt_file(bkg_id)
        self._bkg_evt_file(bkg_id)

        self.output_files["src"]["evt"]
        self._epic_events(
            self.output_files["src"]["evt"],
            self.output_files["src"]["evt_xmm"],
            include_gti,
        )
        self._epic_events(
            self.output_files["bkg"]["evt"],
            self.output_files["bkg"]["evt_xmm"],
            include_gti,
        )


class SimEMOS(SimCCD):
    def _fix_evt_header(self, evt_xmm):
        with fits.open(evt_xmm, "update") as hdu:
            hdu[0].header["CREATOR"] = "SIXTE"
            hdu[0].header["OBSERVER"] = "SIXTE"
            hdu[0].header["INSTRUME"] = self.xmmexp.detector.long
            hdu[0].header["DATAMODE"] = self.xmmexp.datamode
            hdu[0].header["SUBMODE"] = self.xmmexp.submode
            hdu[0].header["FILTER"] = self.xmmexp.filter.long
            hdu[0].header["OBS_ID"] = self.xmmexp.obsid
            hdu[0].header["EXP_ID"] = self.xmmexp.expid
            hdu[0].header["EXPIDSTR"] = self.xmmexp.expidstr
            hdu[0].header["RA_PNT"] = self.xmmexp.pointing.ra.deg
            hdu[0].header["DEC_PNT"] = self.xmmexp.pointing.dec.deg
            hdu[0].header["PA_PNT"] = self.xmmexp.rollangle
            hdu[0].header["RA_NOM"] = self.xmmexp.pointing.ra.deg
            hdu[0].header["DEC_NOM"] = self.xmmexp.pointing.dec.deg
            hdu["EVENTS"].header["INSTRUME"] = self.xmmexp.detector.long
            hdu["EVENTS"].header["CCDID"] = self.ccd["ccdid"]
            hdu["EVENTS"].header["CCDNODE"] = 0

    def _include_gti(self, evt, evt_xmm):
        stdgti = Table.read(evt, hdu=2)

        with fits.open(evt_xmm, "update") as hdu:
            hdu.append(fits.table_to_hdu(stdgti))
            hdu["STDGTI"].header["INSTRUME"] = self.xmmexp.detector.long
            hdu["STDGTI"].header["CCDID"] = self.ccd["ccdid"]
            hdu["STDGTI"].header["CCDNODE"] = 0


class SimEMOS1(SimEMOS):
    def _set_xml_file(self, particle_bkg):
        # TODO: change default background to close filter
        particle_bkg_str = self._get_particle_bkg_str(particle_bkg, mode="flat")
        instrument_dir = self.instrument_dir.joinpath("epicmos")
        
        xml_file = instrument_dir.joinpath(
            f"mos1_fullframe_ccd{self.ccd['tag']}_{self.xmmexp.filter.sixte}filter{particle_bkg_str}.xml"
        )

        return xml_file

    def _epic_events(self, evt, evt_xmm, include_gti):
        sixty.run(
            "epicmos1_events", EvtFile=evt, EPICmos1EventList=evt_xmm, clobber=True,
        )
        self._fix_evt_header(evt_xmm)

        if include_gti:
            self._include_gti(evt, evt_xmm)


class SimEMOS2(SimEMOS):
    def _set_xml_file(self, particle_bkg):
        # TODO: change default background to close filter
        particle_bkg_str = self._get_particle_bkg_str(particle_bkg, mode="flat")
        instrument_dir = self.instrument_dir.joinpath("epicmos")
        
        xml_file = instrument_dir.joinpath(
            f"mos2_fullframe_ccd{self.ccd['tag']}_{self.xmmexp.filter.sixte}filter{particle_bkg_str}.xml"
        )

        return xml_file

    def _epic_events(self, evt, evt_xmm, include_gti):
        sixty.run(
            "epicmos2_events", EvtFile=evt, EPICmos2EventList=evt_xmm, clobber=True,
        )
        self._fix_evt_header(evt_xmm)

        if include_gti:
            self._include_gti(evt, evt_xmm)
