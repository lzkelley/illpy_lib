"""This module handles the processing of Illustris BH files.
"""
import os
import glob

import numpy as np
import h5py
np.seterr(divide='ignore', invalid='ignore')

import pycore

from illpy_lib import NUM_SNAPS


class MERGERS:
    # Meta Data
    RUN       = 'run'
    CREATED   = 'created'
    NUM       = 'num'
    VERSION   = 'version'
    FILE      = 'filename'

    # Physical Parameters
    IDS       = 'ids'
    SCALES    = 'scales'
    MASSES    = 'masses'

    # Maps
    # MAP_STOM  = 's2m'
    # MAP_MTOS  = 'm2s'
    # MAP_ONTOP = 'ontop'
    SNAP_NUMS = "snap_nums"
    ONTOP_NEXT = "ontop_next"
    ONTOP_PREV = "ontop_prev"


MERGERS_PHYSICAL_KEYS = [MERGERS.IDS, MERGERS.SCALES, MERGERS.MASSES]


class DETAILS:
    RUN     = 'run'
    CREATED = 'created'
    VERSION = 'version'
    NUM     = 'num'
    SNAP    = 'snap'
    FILE    = 'filename'

    IDS     = 'id'
    SCALES  = 'scales'
    MASSES  = 'masses'
    MDOTS   = 'mdots'
    DMDTS   = 'dmdts'     # differences in masses
    RHOS    = 'rhos'
    CS      = 'cs'

    UNIQUE_IDS = 'unique_ids'
    UNIQUE_INDICES = 'unique_indices'
    UNIQUE_COUNTS = 'unique_counts'


DETAILS_PHYSICAL_KEYS = [DETAILS.IDS, DETAILS.SCALES, DETAILS.MASSES,
                         DETAILS.MDOTS, DETAILS.DMDTS, DETAILS.RHOS, DETAILS.CS]


class BH_TYPE:
    IN  = 0
    OUT = 1


class Settings(pycore.Settings):

    # NAME = "fire-mergers"
    # VERBOSITY = 30
    VERBOSITY = 20
    LOG_FILENAME = "log_illpy-lib_bh.log"
    RUN_NUM = 1

    INPUT = "/n/ghernquist/Illustris/Runs/L75n1820FP/"
    # OUTPUT = "/n/regal/hernquist_lab/lkelley/illustris-processed/"
    OUTPUT = "/n/scratchlfs/hernquist_lab/lzkelley/illustris-processed/"

    RECREATE = False
    BREAK_ON_FAIL = False

    MAX_DETAILS_PER_SNAP = 10

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

    FNAME_BH_PARTICLES = "bh_particles.hdf5"

    # "ill-%d_blackhole_details_temp_snap-%d.txt"
    FNAME_DETAILS_TEMP_SNAP = "ill-{run_num:d}_blackhole_details_temp_snap-{snap_num:03d}.txt"

    # "ill-%d_blackhole_details_save_snap-%d_v%.2f.npz"
    FNAME_DETAILS_SNAP = "ill-{run_num:d}_blackhole_details_snap-{snap_num:03d}.hdf5"

    # _MERGERS_RAW_COMBINED_FILENAME  = "ill-%d_blackhole_mergers_combined.txt"
    # _MERGERS_RAW_MAPPED_FILENAME    = "ill-%d_blackhole_mergers_mapped_v%.2f.npz"
    FNAME_MERGERS_TEMP = "ill-{run_num:d}_blackhole_mergers_temp.hdf5"

    # _MERGERS_FIXED_FILENAME         = "ill-%d_blackhole_mergers_fixed_v%.2f.npz"
    FNAME_MERGERS_FIXED = "ill-{run_num:d}_blackhole_mergers_fixed.hdf5"

    # _MERGER_DETAILS_FILENAME        = 'ill-%d_blackhole_merger-details_persnap-%03d_v%s.npz'
    FNAME_MERGER_DETAILS = "ill-{run_num:d}_blackhole_merger-details_per-snap-{per_snap:03d}.hdf5"

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
    def output(self):
        return self.OUTPUT

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

    def fname_mergers_temp(self, run_num=None):
        if run_num is None:
            run_num = self._core.sets.RUN_NUM

        fname = self.FNAME_MERGERS_TEMP.format(run_num=run_num)
        fname = os.path.join(self.output, fname)
        return fname

    def fname_mergers_fixed(self, run_num=None):
        if run_num is None:
            run_num = self._core.sets.RUN_NUM

        fname = self.FNAME_MERGERS_FIXED.format(run_num=run_num)
        fname = os.path.join(self.output, fname)
        return fname

    def fname_merger_details(self, run_num=None, max_per_snap=None):
        if run_num is None:
            run_num = self._core.sets.RUN_NUM

        if max_per_snap is None:
            max_per_snap = self._core.sets.MAX_DETAILS_PER_SNAP

        fname = self.FNAME_MERGER_DETAILS.format(run_num=run_num, per_snap=max_per_snap)
        fname = os.path.join(self.output, fname)
        return fname

    @property
    def fname_bh_particles(self):
        return os.path.join(self.OUTPUT, self.FNAME_BH_PARTICLES)


class Core(pycore.Core):
    _CLASS_SETTINGS = Settings
    _CLASS_PATHS = Paths

    def setup_for_ipython(self):
        import matplotlib as mpl
        mpl.use('Agg')
        return

    def _load_cosmology(self):
        import illpy_lib.illcosmo
        cosmo = illpy_lib.illcosmo.cosmology.Illustris_Cosmology()
        return cosmo


def load_hdf5_to_mem(fname):
    with h5py.File(fname, 'r') as data:
        out = {kk: data[kk][:] for kk in data.keys()}
    return out


def _distribute_snapshots(comm):
    """Evenly distribute snapshot numbers across multiple processors.
    """
    size = comm.size
    rank = comm.rank
    mySnaps = np.arange(NUM_SNAPS)
    if size > 1:
        # Randomize which snapshots go to which processor for load-balancing
        mySnaps = np.random.permutation(mySnaps)
        # Make sure all ranks are synchronized on initial (randomized) list before splitting
        mySnaps = comm.bcast(mySnaps, root=0)
        mySnaps = np.array_split(mySnaps, size)[rank]

    return mySnaps


from . import bh_constants  # noqa
