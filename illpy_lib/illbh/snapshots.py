"""
"""

import os
import sys
import glob
import logging
from datetime import datetime

import numpy as np

import illpy.snapshot
import illpy_lib  # noqa
from illpy_lib.illbh import Processed, VERBOSE, KEYS, utils
from illpy_lib.illbh.groupcats import GCAT_KEYS

_BAD_SNAPS_TOS = [53, 55]   # Missing snapshots in original Illustris


class Snapshots(Processed):

    _PROCESSED_FILENAME = "bh-snapshots.hdf5"
    _SNAP_CLASS = None
    # Skip keys that are not in all snapshots
    #    Note: these are still stored in individual files, but not in the combined version here
    _SKIP_KEYS = []

    def __init__(self, *args, load=True, **kwargs):
        super().__init__(*args, load=load, **kwargs)
        return

    def _process(self):
        # sim_path = self._sim_path
        SNAP_CLASS = self._SNAP_CLASS
        if SNAP_CLASS is None:
            SNAP_CLASS = Snapshots_Snap

        # Check output filename
        fname = self.filename()
        utils._check_output_path(fname)
        verbose = self._verbose

        bhs = {}
        snap = 0
        count = 0
        warn_first_skip_key = verbose   # warn on first skipped key from _SKIP_KEYS, if verbose
        while (snap < 10000):
            if verbose:
                print("snap = {}".format(snap))

            # WARNING: `bh_snap` may be an empty dictionary, when there are no BHs in snapshot!
            try:
                bh_snap = SNAP_CLASS(
                    snap, self._sim_path, self._processed_path,
                    must_exist=True, verbose=verbose
                )
            except OSError as err:
                if verbose:
                    print("Failed to load snap {} : {}".format(snap, str(err)))
                break

            if len(bh_snap.keys()) > 0:
                count = bh_snap[KEYS.ID].size

            for kk in bh_snap.keys():
                if kk in self._SKIP_KEYS:
                    if warn_first_skip_key:
                        msg = "Skipping key '{}' and all of '{}'".format(kk, self._SKIP_KEYS)
                        logging.warning(msg)
                    warn_first_skip_key = False
                    continue

                temp = bh_snap[kk]
                prev = bhs.get(kk, None)
                if (prev is None) or (len(prev) == 0):
                    bhs[kk] = temp
                else:
                    bhs[kk] = np.concatenate([prev, temp])

            snap += 1

        if verbose:
            print("Loaded {} entries from {} snaps".format(count, snap))

        if len(bhs) == 0:
            err = "After {} snaps, `bhs` is empty!".format(snap)
            logging.error(err)
            raise RuntimeError(err)

        fname = self.filename()
        self._finalize_and_save(fname, bhs)
        return

    def _add_derived(self, data):
        u_ids, u_inds, u_counts = np.unique(data[KEYS.ID], return_index=True, return_counts=True)
        data[KEYS.U_IDS] = u_ids
        data[KEYS.U_INDICES] = u_inds
        data[KEYS.U_COUNTS] = u_counts
        return data

    def _add_gcat_vals(self, snaps, gcats=None):
        verbose = self._verbose
        if gcats is None:
            try:
                gcats = illpy_lib.illbh.groupcats.Groupcats(
                    self._sim_path, self._processed_path,
                    must_exist=True, verbose=self._verbose, load=True
                )
            except:
                logging.error("FAILED to load `Groupcats`!")
                raise

        sn_u_ids = snaps[KEYS.U_IDS]
        sn_u_inds = snaps[KEYS.U_INDICES]
        sn_u_counts = snaps[KEYS.U_COUNTS]

        gc_u_ids = gcats[GCAT_KEYS.U_IDS]
        gc_u_inds = gcats[GCAT_KEYS.U_INDICES]
        gc_u_counts = gcats[GCAT_KEYS.U_COUNTS]

        if verbose:
            print(f"Unique IDs: snapshots {sn_u_ids.size}, groupcats {gc_u_ids.size}")
        if np.any(sn_u_ids != gc_u_ids) or np.any(sn_u_inds != gc_u_inds) or np.any(sn_u_counts != gc_u_counts):
            err = f"Mismatch in unique IDs or indices!"
            raise ValueError(err)

        if np.any(snaps[KEYS.ID] != gcats[GCAT_KEYS.ID]) or np.any(snaps[KEYS.SNAP] != gcats[GCAT_KEYS.SNAP]):
            raise ValueError(f"Mismatch between ID numbers!")

        keys = [
            GCAT_KEYS.SUBHALO, GCAT_KEYS.HALO,
            GCAT_KEYS.SubhaloMassType, GCAT_KEYS.SubhaloMassInHalfRadType, GCAT_KEYS.SubhaloHalfmassRadType,
            GCAT_KEYS.SubhaloSFRinHalfRad, GCAT_KEYS.SubhaloVelDisp, GCAT_KEYS.SubhaloVmax,
        ]

        if verbose:
            print("Copying keys: {keys}")

        for kk in keys:
            snaps[kk] = np.copy(gcats[kk])

        return snaps

    def _finalize(self, data, **header):
        # ---- Sort by ID number and then scale-factor
        idx = np.lexsort((data[KEYS.SCALE], data[KEYS.ID]))
        for kk in data.keys():
            temp = data[kk]
            temp = temp[idx, ...]
            data[kk] = temp

        data = self._add_derived(data)
        data = self._add_gcat_vals(data)
        return data


class Snapshots_Snap(Snapshots):

    _PROCESSED_FILENAME = "bh-snapshots_{snap:03d}.hdf5"
    _SKIP_KEYS = []   # NOTE: this is not used in `Snapshots_Snap`
    _PROCESSED_DIR_NAME = "bh-snapshots"

    def __init__(self, snap, *args, **kwargs):
        self._snap = snap
        # This should have `load=True` default from `Snapshots` class
        super().__init__(*args, **kwargs)
        return

    def _process(self):
        snap = self._snap

        # Check output filename
        fname = self.filename()
        fname = utils._check_output_path(fname)

        pt_bh = illpy.PARTICLE.BH
        sim_path = os.path.join(self._sim_path, 'output')
        header = illpy.snapshot.get_header(sim_path, snap)
        if self._verbose:
            time = header['Time']
            num_bhs = header['NumPart_Total'][pt_bh]
            print("Loaded header time={:.8f}, nbh={:d}".format(time, num_bhs))
        bh_snap = illpy.snapshot.loadSubset(sim_path, snap, pt_bh, fields=None, sq=False)
        num_bhs = bh_snap.pop('count')
        if self._verbose:
            print("Loaded {} BHs from snapshot {}".format(num_bhs, snap))

        if num_bhs == 0:
            print("Snapshot {} contains no BHs".format(snap))
            bh_snap = None
        else:
            bh_snap[KEYS.SCALE] = header['Time'] * np.ones_like(bh_snap[KEYS.MASS])
            bh_snap[KEYS.SNAP] = snap * np.ones_like(bh_snap[KEYS.MASS], dtype=int)

        try:
            self._finalize_and_save(fname, bh_snap, **header)
        except Exception as err:
            logging.error("FAILED on snap {}!".format(snap))
            logging.error(str(err))
            raise

        return

    def _finalize(self, data, **header):
        if len(data) == 0:
            return data

        idx = np.argsort(data[KEYS.ID])
        for kk in data.keys():
            temp = data[kk]
            temp = temp[idx, ...]
            data[kk] = temp

        return data

    def filename(self):
        temp = self._PROCESSED_FILENAME.format(snap=self._snap)
        fname = os.path.join(self._processed_path, self._PROCESSED_DIR_NAME, temp)
        return fname


class Snapshots_TNG(Snapshots):
    # These values are only in the 'full' snapshots (not in 'mini' ones), skip them for now
    _SKIP_KEYS = ['SubfindDMDensity', 'SubfindDensity', 'SubfindHsml', 'SubfindVelDisp']

    '''
    _NUM_SNAPS = 100
    
    _PROCESSED_FILENAME = "bh-snapshots.hdf5"
    # _SKIP_KEYS = []   # NOTE: this is not used in `Snapshots_Snap`
    # _PROCESSED_DIR_NAME = "bh-snapshots"

    def __init__(self, *args, load=True, **kwargs):
        super().__init__(*args, load=load, **kwargs)
        return

    def _process(self):
        snap = self._snap

        # Check output filename
        fname_out = self.filename()
        fname_out = utils._check_output_path(fname)
        
        for snap in range(self._NUM_SNAPS):
            

        return

    def filename(self):
        fname = os.path.join(self._processed_path, self._PROCESSED_FILENAME)
        return fname
    '''

class Snapshots_TOS(Snapshots):
    pass


class Snapshots_New(Snapshots):
    pass


def process_snapshot_snaps(mode, sim_path, proc_path, recreate=False, verbose=VERBOSE):
    try:
        comm = MPI.COMM_WORLD
    except NameError:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    beg = datetime.now()
    if comm.rank == 0:
        temp = os.path.join(sim_path, 'output', 'snapdir_*')
        snap_dirs = sorted(glob.glob(temp))
        num_snaps = len(snap_dirs)
        if verbose:
            print("Found {} snapshot directories".format(num_snaps))
        if num_snaps == 0:
            err = "ERROR: no snapshot directories found (pattern: '{}')".format(temp)
            logging.error(err)
            raise FileNotFoundError(err)

        snap_list = np.arange(num_snaps).tolist()
        bad_snaps = None
        if mode == 'tos':
            bad_snaps = _BAD_SNAPS_TOS
        elif mode == 'tng':
            pass
        elif mode == 'new':
            pass
        else:
            raise ValueError("Unrecognized `mode` '{}'!".format(mode))

        if bad_snaps is not None:
            logging.warning("Removing TOS bad snaps '{}' from targets".format(bad_snaps))
            for bsn in bad_snaps:
                snap_list.remove(bsn)
                assert (bsn not in snap_list), "FAILED"

        np.random.seed(1234)
        np.random.shuffle(snap_list)
        snap_list = np.array_split(snap_list, comm.size)
    else:
        snap_list = None

    snap_list = comm.scatter(snap_list, root=0)

    comm.barrier()
    num_lines = 0
    for snap_num in snap_list:
        try:
            snap = Snapshots_Snap(
                snap_num, sim_path=sim_path, processed_path=proc_path,
                recreate=recreate, verbose=verbose)
        except Exception as err:
            logging.error("FAILED to load snap '{}'!".format(snap_num))
            logging.error(str(err))
            raise err

        if len(snap.keys()) > 0:
            num_lines += snap[KEYS.SCALE].size

    num_lines = comm.gather(num_lines, root=0)

    end = datetime.now()
    if verbose:
        print("Rank: {}, done at {}, after {}".format(comm.rank, end, (end-beg)))
    if comm.rank == 0:
        tot_num_lines = np.sum(num_lines)
        ave = np.mean(num_lines)
        med = np.median(num_lines)
        std = np.std(num_lines)
        if verbose:
            print("Tot lines={:.2e}, med={:.2e}, ave={:.2e}±{:.2e}".format(
                tot_num_lines, med, ave, std))

        tail = f"Done at {str(end)} after {str(end-beg)} ({(end-beg).total_seconds()})"
        print("\n" + "=" * len(tail) + "\n" + tail + "\n")
        logging.error(tail)

    comm.barrier()
    return


if __name__ == "__main__":
    # logging.warning("Loading MPI...")
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    beg = datetime.now()
    recreate = ('-r' in sys.argv) or ('--recreate' in sys.argv)
    verbose = ('-v' in sys.argv) or ('--verbose' in sys.argv) or VERBOSE

    if comm.rank == 0:
        print("`recreate` = {}".format(recreate))
        print("`verbose`  = {}".format(verbose))
        this_fname = os.path.abspath(__file__)
        head = f"{this_fname} : {str(beg)} - rank: {comm.rank}/{comm.size}"
        print("\n" + head + "\n" + "=" * len(head) + "\n")

        if (len(sys.argv) < 4) or ('-h' in sys.argv) or ('--help' in sys.argv):
            logging.warning("USAGE: `python {} <MODE> <PATH_IN> <PATH_OUT>`\n\n".format(__file__))
            sys.exit(0)

        cc = 1
        mode = sys.argv[cc]
        cc += 1
        sim_path = os.path.abspath(sys.argv[cc]).rstrip('/')
        cc += 1
        proc_path = os.path.abspath(sys.argv[cc]).rstrip('/')
        cc += 1

        mode = mode.strip().lower()
        print("mode   '{}'".format(mode))
        print("input  `sim_path`  : {}".format(sim_path))
        print("output `proc_path` : {}".format(proc_path))
        valid_modes = ['tos', 'tng', 'new']
        if mode not in valid_modes:
            err = "Unrecognized `mode` '{}'!".format(mode)
            logging.error(err)
            raise ValueError(err)

    else:
        mode = None
        sim_path = None
        proc_path = None

    mode = comm.bcast(mode, root=0)
    sim_path = comm.bcast(sim_path, root=0)
    proc_path = comm.bcast(proc_path, root=0)
    comm.barrier()

    process_snapshot_snaps(mode, sim_path, proc_path, recreate=recreate, verbose=verbose)




'''
class SNAP_KEYS(Processed.KEYS):

    BH_Hsml                 = 'BH_Hsml'
    BH_Pressure             = 'BH_Pressure'
    BH_Progs                = 'BH_Progs'
    BH_U                    = 'BH_U'
    SubfindDensity          = 'SubfindDensity'
    SubfindHsml             = 'SubfindHsml'
    SubfindVelDisp          = 'SubfindVelDisp'

    BH_Density              = 'BH_Density'
    BH_Mass                 = 'BH_Mass'
    BH_Mdot                 = 'BH_Mdot'
    Coordinates             = 'Coordinates'
    Masses                  = 'Masses'
    ParticleIDs             = 'ParticleIDs'
    Potential               = 'Potential'
    Velocities              = 'Velocities'

    SCALE                   = 'scale'
    SNAP                    = 'snap'

    ID                      = ParticleIDs
    MASS                    = Masses
    BH_MASS                 = BH_Mass
    MDOT                    = BH_Mdot
    POT                     = Potential
    DENS                    = BH_Density
    POS                     = Coordinates
    VEL                     = Velocities

    # NOTE: these are 'derived' in the sense for each snapshot: these parameters must be added
    #       when combining all snapshots together, these are not derived, they come from the data
    _DERIVED = [SCALE, SNAP]
    _ALIASES = [ID, MASS, BH_MASS, MDOT, POT, DENS, POS, VEL]

    @classmethod
    def names(cls):
        nn = [kk for kk in dir(cls) if (not kk.startswith('_')) and (kk not in cls._ALIASES)]
        nn = [kk for kk in nn if (not callable(getattr(cls, kk)))]
        return sorted(nn)
'''

'''
class KEYS(SNAP_KEYS):

    U_IDS                   = 'unique_ids'
    U_INDICES               = 'unique_indices'
    U_COUNTS                = 'unique_counts'

    # These are the 'derived' keys for *all* snapshots together
    #    the 'SCALE' and 'SNAP' are included in the individual snapshot files
    _DERIVED = [U_IDS, U_INDICES, U_COUNTS]
'''
