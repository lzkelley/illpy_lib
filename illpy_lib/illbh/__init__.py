"""This module handles the processing of Illustris BH files.
"""
import os
import glob

# import numpy as np

import pycore


class Settings(pycore.Settings):

    # NAME = "fire-mergers"
    # VERBOSITY = 30
    VERBOSITY = 20
    LOG_FILENAME = "log_illpy-lib_bh.log"
    RUN_NUM = 1

    INPUT = "/n/ghernquist/Illustris/Runs/L75n1820FP/"
    OUTPUT = "/n/regal/hernquist_lab/lkelley/illustris-processed/"

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
    _MERGERS_FILENAME_REGEX = "blackhole_mergers_*.txt"
    _DETAILS_FILENAME_REGEX = "blackhole_details_*.txt"

    FNAME_DETAILS_CLEAN = "bh_details.hdf5"
    FNAME_MERGERS_CLEAN = "bh_mergers.hdf5"

    # "ill-%d_blackhole_details_temp_snap-%d.txt"
    FNAME_DETAILS_TEMP_SNAP = "ill-{run_num:d}_blackhole_details_temp_snap-{snap_num:03d}.txt"

    # "ill-%d_blackhole_details_save_snap-%d_v%.2f.npz"
    FNAME_DETAILS_SNAP = "ill-{run_num:d}_blackhole_details_snap-{snap_num:03d}.hdf5"

    # The substituted string should be either 'mergers' or 'details'
    _ILL_1_TXT_DIRS = [
        "txt-files-curie/blackhole_{}/",
        "txt-files-supermuc/blackhole_{}/",
        "txt-files-partial/Aug8/blackhole_{}/",
        "txt-files-partial/Aug14/blackhole_{}/",
        "txt-files-partial/Sep25/blackhole_{}/",
        "txt-files-partial/Oct10/blackhole_{}/"
    ]

    def __init__(self, core, **kwargs):
        super().__init__(core)
        self.OUTPUT = os.path.realpath(core.sets.OUTPUT)
        self.INPUT = os.path.realpath(core.sets.INPUT)
        return

    @property
    def mergers_input(self):
        return self._find_input_files('mergers', self._MERGERS_FILENAME_REGEX)

    @property
    def fnames_details_input(self):
        return self._find_input_files('details', self._DETAILS_FILENAME_REGEX)

    def _find_input_files(self, name, regex):
        log = self._core.log

        if ('illustris-1' in self.INPUT.lower()) or ('L75n1820FP' in self.INPUT):
            log.debug("Input looks like `illustris-1` ('{}')".format(self.INPUT))
            _path = os.path.join(self.INPUT, 'txt-files/txtfiles_new/')
            paths = [os.path.join(_path, td.format(name), '') for td in self._ILL_1_TXT_DIRS]
        elif ('illustris-2' in self.INPUT.lower()) or ('L75n910FP' in self.INTPUT):
            log.debug("Input looks like `illustris-2` ('{}')".format(self.INPUT))
            # subdir = "/combined_output/blackhole_mergers/"
            subdir = "/combined_output/blackhole_{}/".format(name)
            paths = [os.path.join(self.INPUT, subdir)]
        else:
            log.debug("Input looks like `illustris-3` or default ('{}')".format(
                self.INPUT))
            # subdir = "/output/blackhole_mergers/"
            subdir = "/output/blackhole_{}/".format(name)
            paths = [os.path.join(self.INPUT, subdir)]

        files = []
        log.debug("Checking {} directories for {} files".format(len(paths), name))
        for pp in paths:
            if not os.path.exists(pp):
                raise RuntimeError("Expected path '{}' does not exist!".format(pp))
            pattern = os.path.join(pp, regex)
            log.debug("  Getting {} files from '{}'".format(name, pp))
            _fils = sorted(glob.glob(pattern))
            num_fils = len(_fils)
            if num_fils == 0:
                raise RuntimeError("No {} files found matching '{}'".format(name, pattern))
            log.debug("    Found '{}' files, e.g. '{}'".format(num_fils, os.path.basename(_fils[0])))
            files += _fils

        log.debug("Found {} {} files".format(len(files), name))
        return files

    @property
    def details_clean(self):
        return os.path.join(self.OUTPUT, self.FNAME_DETAILS_CLEAN)

    @property
    def output_details(self):
        return os.path.join(self.OUTPUT, "details", "")

    @property
    def mergers_clean(self):
        return os.path.join(self.OUTPUT, self.FNAME_MERGERS_CLEAN)

    @property
    def output_plots(self):
        path = os.path.join('.', self._DNAME_PLOTS, "")
        # path = os.path.realpath(path)
        path = self.check_path(path)
        return path

    def fname_details_temp_snap(self, snap, run_num=None):
        if run_num is None:
            run_num = self._core.sets.RUN_NUM

        fname = self.FNAME_DETAILS_TEMP_SNAP.format(snap_num=snap, run_num=run_num)
        fname = os.path.join(self.output_details, fname)
        return fname

    def fname_details_snap(self, snap, run_num=None):
        if run_num is None:
            run_num = self._core.sets.RUN_NUM

        fname = self.FNAME_DETAILS_SNAP.format(snap_num=snap, run_num=run_num)
        fname = os.path.join(self.output_details, fname)
        return fname


class Core(pycore.Core):
    _CLASS_SETTINGS = Settings
    _CLASS_PATHS = Paths

    def setup_for_ipython(self):
        import matplotlib as mpl
        mpl.use('Agg')
        return


from . import bh_constants  # noqa
