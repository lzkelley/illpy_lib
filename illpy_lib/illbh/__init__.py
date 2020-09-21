"""This module handles the processing of Illustris BH files.
"""

__version__ = "0.5.2"

import os
import logging
# import enum

# import numpy as np
import h5py
# np.seterr(divide='ignore', invalid='ignore')

# from illpy_lib.constants import NUM_SNAPS  # noqa

# from . deep_core import Core  # noqa
from . import utils

# PATH_PROCESSED = ["output", "processed"]    # relative to simulation directory (i.e. where 'output' directory lives)
PATH_PROCESSED = ["processed"]                # relative to simulation-output directory (i.e. where snapshots and groups live)

VERBOSE = True

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

    def __len__(self):
        return len(iter(self))


'''
@enum.unique
class _INT_ENUM(enum.IntEnum):
    def __str__(self):
        return str(self.value)


class BH_TYPE(_INT_ENUM):
    OUT = 0
    IN = 1
    REMNANT = 2
'''


class BH_TYPE(ENUM):
    OUT = 0
    IN = 1
    REMNANT = 2

    @classmethod
    def from_value(cls, value):
        for nn in cls.names():
            if getattr(cls, nn) == value:
                return nn
        else:
            raise KeyError("Unrecognized value '{}'!  values: {}".format(value, cls.keys()))


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

    def _save_to_hdf5(self, fname, keys, vals, script, **header):
        try:
            with h5py.File(fname, 'w') as save:
                utils._save_meta_to_hdf5(save, self._sim_path, __version__, script)
                for kk in keys:
                    save.create_dataset(kk, data=vals[kk])

                for kk, vv in header.items():
                    save.attrs[kk] = vv

        except:
            if os.path.exists(fname):
                logging.error("ERROR: Failure while writing to '{}', deleting file!".format(fname))
                os.remove(fname)
            raise

        return

    def _load_from_save(self):
        fname = self.filename
        with h5py.File(fname, 'r') as load:
            vers = load.attrs['version']
            if vers != __version__:
                msg = "WARNING: loaded version '{}' does not match current '{}'!".format(
                    vers, __version__)
                logging.warning(msg)
            spath = load.attrs['sim_path']
            if self._sim_path is not None:
                if os.path.abspath(self._sim_path).lower() != os.path.abspath(spath).lower():
                    msg = "WARNING: loaded sim_path '{}' does not match current '{}'!".format(
                        spath, self._sim_path)
                    logging.warning(msg)
            else:
                self._sim_path = spath

            keys = self.keys()
            try:
                size = load[self.KEYS.SCALE].size
            except:
                wrn = "WARNING: Could not set `size` based on key: `{}`".format(self.KEYS.SCALE)
                logging.warning(wrn)
                size = None

            for kk in keys:
                try:
                    vals = load[kk][:]
                    setattr(self, kk, vals)
                except Exception as err:
                    msg = "ERROR: failed to load '{}' from '{}'!".format(kk, fname)
                    logging.error(msg)
                    logging.error(str(err))
                    raise

            for kk in load.keys():
                if kk not in keys:
                    err = "WARNING: '{}' has unexpected data '{}'".format(fname, kk)
                    logging.warning(err)

            if self._verbose:
                dt = load.attrs['created']
                print("Loaded {:10d} entries from '{}', created '{}'".format(size, fname, dt))

        self._size = size
        return

    def _process(self):
        raise NotImplementedError()
