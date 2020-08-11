"""This module handles the processing of Illustris BH files.
"""

import os
import logging
import enum

import numpy as np
# import h5py
# np.seterr(divide='ignore', invalid='ignore')

# from illpy_lib.constants import NUM_SNAPS  # noqa

# from . deep_core import Core  # noqa

PATH_PROCESSED = ["output", "processed"]


'''
@enum.unique
class _ENUM(enum.Enum):
    def __str__(self):
        return str(self.value)
'''


class _ENUM(type):

    def __iter__(self):
        for kk in self.names():
            yield getattr(self, kk)

    def keys(self):
        return list(iter(self))

    def names(self):
        return sorted([kk for kk in dir(self) if not kk.startswith('_')])


class ENUM(metaclass=_ENUM):
    pass


@enum.unique
class _INT_ENUM(enum.IntEnum):
    def __str__(self):
        return str(self.value)


class BH_TYPE(_INT_ENUM):
    OUT = 0
    IN = 1


class Processed:

    _PROCESSED_FILENAME = None

    class KEYS(ENUM):
        pass

    def __init__(self, sim_path=None, filename=None, verbose=True, recreate=False):
        # -- Initialize
        if (sim_path is None) and (filename is None):
            err = "ERROR: Either `sim_path` or `filename` must be provided!"
            logging.error(err)
            raise ValueError(err)
        elif (sim_path is not None) and (not os.path.isdir(sim_path)):
            err = "ERROR: `sim_path` '{}' does not exist!".format(sim_path)
            logging.error(err)
            raise ValueError(err)
        elif (sim_path is None) and (not os.path.isfile(filename)):
            err = "ERROR: `filename` '{}' does not exist!".format(filename)
            logging.error(err)
            raise ValueError(err)

        if filename is not None:
            filename = os.path.abspath(filename)

        self._verbose = verbose
        self._recreate = recreate
        self._sim_path = sim_path
        self._filename = filename
        self._size = None

        # -- Load data
        self._load(recreate)

        return

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return [str(kk) for kk in self.KEYS]

    @property
    def size(self):
        return self._size

    @property
    def filename(self):
        if self._filename is None:
            # `sim_path` has already been checked to exist in initializer
            sim_path = self._sim_path
            self._filename = os.path.join(sim_path, *PATH_PROCESSED, self._PROCESSED_FILENAME)

        return self._filename

    def _load(self, recreate):
        if not os.path.exists(self.filename) or recreate:
            self._process()

        self._load_from_save()
        return

    def _load_from_save(self):
        raise NotImplementedError()

    def _process(self):
        raise NotImplementedError()


'''
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


class _LenMeta(type):

    def __len__(self):
        return self.__len__()


class BH_TYPE(metaclass=_LenMeta):
    IN  = 0
    OUT = 1

    @classmethod
    def __len__(cls):
        return 2


class BH_TREE:
    PREV         = 'prev'
    NEXT         = 'next'
    SCALE_PREV   = 'scale_prev'
    SCALE_NEXT   = 'scale_next'
    TIME_PREV    = 'time_prev'
    TIME_NEXT    = 'time_next'

    NUM_BEF      = 'num_bef'
    NUM_AFT      = 'num_aft'
    TIME_BETWEEN = 'time_between'

    CREATED      = 'created'
    RUN          = 'run'
    VERSION      = 'version'
    NUM          = 'num'


from . utils import load_hdf5_to_mem, _distribute_snapshots  # noqa

# from . import bh_constants  # noqa
'''
