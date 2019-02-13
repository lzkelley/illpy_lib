"""This module handles the processing of Illustris BH files.
"""

import os

# import numpy as np

import pycore


class Settings(pycore.Settings):

    # NAME = "fire-mergers"
    VERBOSITY = 30
    LOG_FILENAME = "illpy-lib_bh.log"

    INPUT = "/projects/b1026/anglesd/FIRE/h2_HR_sn1dy300ro100ss/"
    OUTPUT = "/projects/p30669/h2_HR_sn1dy300ro100ss_bh/"

    RECREATE = False
    BREAK_ON_FAIL = False

    def add_arguments(argself, parser):
        '''
        parser.add_argument(
            '-s', '--sim', type=str,
            help='type of simulation being processed')
        '''
        return


class Paths(pycore.Paths):
    FNAME_MERGERS = "blackhole_details/bhMRG.hdf5"
    FNAME_DETAILS_REGEX = "blackhole_details/blackhole_details_*.txt"

    FNAME_DETAILS_CLEAN = "bh_details.hdf5"
    FNAME_MERGERS_CLEAN = "bh_mergers.hdf5"

    def __init__(self, core, **kwargs):
        super().__init__(core)
        self.OUTPUT = core.sets.OUTPUT
        self.INPUT = core.sets.INPUT
        return

    @property
    def mergers(self):
        return os.path.join(self.INPUT, self.FNAME_MERGERS)

    @property
    def details_clean(self):
        return os.path.join(self.OUTPUT, self.FNAME_DETAILS_CLEAN)

    @property
    def mergers_clean(self):
        return os.path.join(self.OUTPUT, self.FNAME_MERGERS_CLEAN)

    @property
    def output_plots(self):
        path = os.path.join('.', self._DNAME_PLOTS, "")
        # path = os.path.realpath(path)
        path = self.check_path(path)
        return path


class Core(pycore.Core):
    _CLASS_SETTINGS = Settings
    _CLASS_PATHS = Paths

    def setup_for_ipython(self):
        import matplotlib as mpl
        mpl.use('Agg')
        return


from . import bh_constants  # noqa
