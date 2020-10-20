"""This module handles the processing of Illustris BH files.
"""

__version__ = "0.5.3"

import os
import logging
# import enum

import numpy as np
import h5py
# np.seterr(divide='ignore', invalid='ignore')

import zcode.inout as zio

# from illpy_lib.constants import NUM_SNAPS  # noqa

# from . deep_core import Core  # noqa
from . import utils

# PATH_PROCESSED = ["output", "processed"]    # relative to simulation directory (i.e. where 'output' directory lives)
# relative to simulation-output directory (i.e. where snapshots and groups live)
# _PROCESSED_DIR = ["postprocessing", "blackholes"]

VERBOSE = True


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


class KEYS(ENUM):

    # keys from Illustris
    ParticleIDs             = 'ParticleIDs'
    Masses                  = 'Masses'

    BH_Mass                 = 'BH_Mass'
    BH_Mdot                 = 'BH_Mdot'
    BH_Density              = 'BH_Density'
    BH_SoundSpeed           = 'BH_SoundSpeed'   # unique to tng-details files

    # Added keys for basic parameters
    SCALE = 'scale'
    SNAP = 'snap'
    TASK = 'task'

    # Aliases
    MASS = Masses
    BH_MASS = BH_Mass
    ID = ParticleIDs

    _ALIASES = [MASS, BH_MASS, ID]

    # Added keys for derived parameters
    U_IDS = 'unique_ids'
    U_INDICES = 'unique_indices'
    U_COUNTS = 'unique_counts'
    T_NEXT = 'tree_next'
    T_PREV = 'tree_prev'
    T_FINAL = 'tree_final'
    T_NUM_PREV = 'tree_num_prev'
    T_NUM_NEXT = 'tree_num_next'

    # _DERIVED = [TASK, SNAP, U_IDS, U_INDICES, U_COUNTS]
    # _INTERP_KEYS = [MASS, BH_MASS, MDOT, MDOT_B, POT, DENS, POS, VEL]
    _U_KEYS = [U_IDS, U_INDICES, U_COUNTS]
    _T_KEYS = [T_NEXT, T_PREV, T_FINAL, T_NUM_PREV, T_NUM_NEXT]
    _DERIVED = _U_KEYS + _T_KEYS


class Processed:

    _PROCESSED_FILENAME = "bh-snapshots.hdf5"

    def __init__(self, sim_path, processed_path, verbose=VERBOSE, recreate=False, load=True):
        sim_path = os.path.abspath(sim_path)
        sim_path = sim_path.rstrip('/')
        if os.path.dirname(sim_path) == 'output':
            sim_path = sim_path.rstrip('output')

        temp = os.path.join(sim_path, 'output')
        if not os.path.isdir(temp):
            raise ValueError("Could not find simulation `output` path '{}'!".format(temp))

        self._verbose = verbose
        self._recreate = recreate
        self._sim_path = sim_path
        self._processed_path = processed_path
        # self._filename = None
        self._keys = []
        self._size = None

        # -- Load data
        # if load:
        #     self._load(recreate)

        return

    def keys(self):
        return self._keys

    def __getitem__(self, key):
        return getattr(self, key)

    '''
    def _unique(self):
        keys = self.KEYS
        try:
            vals = zip(self[keys.U_IDS], self[keys.U_INDICES], self[keys.U_COUNTS])
        except KeyError as err:
            raise AttributeError("Missing `unique` keys! Error: '{}'".format(str(err)))

        for idn, idx, num in vals:
            yield idn, idx, num

        return
    '''

    def _finalize_and_save(self, fname, data, **header):
        if data is None:
            data = {}
            logging.warning("Saving with no data '{}' !".format(fname))

        data = self._finalize(data)
        self._save_to_hdf5(fname, __file__, data, **header)

        if self._verbose:
            msg = "Saved data to '{}' size {}".format(fname, zio.get_file_size(fname))
            print(msg)

        return

    def _finalize(self, data):
        return data

    def _save_to_hdf5(self, fname, script_name, data, keys=None, **header):
        try:
            if keys is None:
                keys = data.keys()

            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with h5py.File(fname, 'w') as save:
                utils._save_meta_to_hdf5(save, self._sim_path, __version__, script_name)
                for kk in keys:
                    save.create_dataset(kk, data=data[kk])

                for kk, vv in header.items():
                    save.attrs[kk] = vv

        except:
            if os.path.exists(fname):
                logging.error("ERROR: Failure while writing to '{}', deleting file!".format(fname))
                os.remove(fname)
            raise

        return

    def _load(self, fname, recreate):
        # fname = self.filename
        exists = os.path.exists(fname)
        if not exists or recreate:
            if self._verbose:
                print("Running `_process()`; recreate: {}, exists: {} ({})".format(
                    recreate, exists, fname))
            self._process()

        self._load_from_save(fname)
        return

    def _load_from_save(self, fname):
        # fname = self.filename
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

            keys = list(load.keys())
            size = 0
            for kk in keys:
                try:
                    vals = load[kk][:]
                    setattr(self, kk, vals)
                except Exception as err:
                    msg = "ERROR: failed to load '{}' from '{}'!".format(kk, fname)
                    logging.error(msg)
                    logging.error(str(err))
                    raise

                if not np.isscalar(vals):
                    size = np.max([size, np.shape(vals)[0]])

            self._keys = keys

            if self._verbose:
                dt = load.attrs['created']
                print("Loaded {:10d} entries from '{}', created '{}'".format(size, fname, dt))
            if size == 0:
                logging.warning("No values loaded; keys = '{}' from '{}'!".format(keys, fname))

        return

    def _process(self):
        raise NotImplementedError()

    def filename(self, *args, **kwargs):
        # temp = self._PROCESSED_FILENAME.format(snap=self._snap)
        temp = self._PROCESSED_FILENAME
        fname = os.path.join(self._processed_path, temp)
        return fname

    @property
    def size(self):
        return self._size
