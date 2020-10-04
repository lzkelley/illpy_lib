"""
"""

import numpy as np


import illpy.snapshot
import illpy_lib  # noqa
from illpy_lib.illbh import Processed, PATH_PROCESSED, VERBOSE


class GCAT_KEYS(Processed.KEYS):

    'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloHalfmassRadType', 'SubhaloLenType',
    'SubhaloMassType', 'SubhaloMassInHalfRadType', 'SubhaloSFRinHalfRad', 'SubhaloSFR',
    'SubhaloStellarPhotometrics', 'SubhaloVelDisp', 'SubhaloVmax'


    BH_BPressure            = 'BH_BPressure'
    BH_CumEgyInjection_QM   = 'BH_CumEgyInjection_QM'
    BH_CumEgyInjection_RM   = 'BH_CumEgyInjection_RM'
    BH_CumMassGrowth_QM     = 'BH_CumMassGrowth_QM'
    BH_CumMassGrowth_RM     = 'BH_CumMassGrowth_RM'
    BH_HostHaloMass         = 'BH_HostHaloMass'
    BH_Hsml                 = 'BH_Hsml'
    BH_MPB_CumEgyHigh       = 'BH_MPB_CumEgyHigh'
    BH_MPB_CumEgyLow        = 'BH_MPB_CumEgyLow'
    BH_Pressure             = 'BH_Pressure'
    BH_Progs                = 'BH_Progs'
    BH_U                    = 'BH_U'
    SubfindDMDensity        = 'SubfindDMDensity'
    SubfindDensity          = 'SubfindDensity'
    SubfindHsml             = 'SubfindHsml'
    SubfindVelDisp          = 'SubfindVelDisp'

    BH_Density              = 'BH_Density'
    BH_Mass                 = 'BH_Mass'
    BH_Mdot                 = 'BH_Mdot'
    BH_MdotBondi            = 'BH_MdotBondi'
    BH_MdotEddington        = 'BH_MdotEddington'
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
    MDOT_B                  = BH_MdotBondi
    MDOT                    = BH_Mdot
    POT                     = Potential
    DENS                    = BH_Density
    POS                     = Coordinates
    VEL                     = Velocities

    # NOTE: these are 'derived' in the sense for each snapshot: these parameters must be added
    #       when combining all snapshots together, these are not derived, they come from the data
    _DERIVED = [SCALE, SNAP]
    _ALIASES = [ID, MASS, BH_MASS, MDOT_B, MDOT, POT, DENS, POS, VEL]

    @classmethod
    def names(cls):
        nn = [kk for kk in dir(cls) if (not kk.startswith('_')) and (kk not in cls._ALIASES)]
        nn = [kk for kk in nn if (not callable(getattr(cls, kk)))]
        return sorted(nn)


class Snapshots(Processed):

    _PROCESSED_FILENAME = "bh-snapshots.hdf5"

    class KEYS(SNAP_KEYS):

        U_IDS                   = 'unique_ids'
        U_INDICES               = 'unique_indices'
        U_COUNTS                = 'unique_counts'

        # These are the 'derived' keys for *all* snapshots together
        #    the 'SCALE' and 'SNAP' are included in the individual snapshot files
        _DERIVED = [U_IDS, U_INDICES, U_COUNTS]

    def _process(self):
        sim_path = self._sim_path
        if sim_path is None:
            err = "ERROR: cannot process {} without `sim_path` set!".format(self.__class__)
            logging.error(err)
            raise ValueError(err)

        # Check output filename
        fname = self.filename
        path = os.path.dirname(fname)
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.isdir(path):
            err = "ERROR: filename '{}' path '{}' is not a directory!".format(fname, path)
            logging.error(err)
            raise FileNotFoundError(err)

        cosmo = illpy_lib.illcosmo.Simulation_Cosmology(sim_path, verbose=False)
        scales = cosmo.scale
        num_snaps = scales.size

        snap = 0
        # KEYS = self.KEYS
        verb = self._verbose

        bhs = {}
        for snap in range(num_snaps):
            if verb:
                print("snap = {}".format(snap))

            bh_snap = Snapshots_Snap(snap, sim_path=sim_path)
            for kk in bh_snap.KEYS:
                temp = bh_snap[kk]
                prev = bhs.get(kk, None)
                if (prev is None) or (len(prev) == 0):
                    bhs[kk] = temp
                else:
                    bhs[kk] = np.concatenate([prev, temp])

        self._finalize_and_save(bhs)

        return

    def _add_derived(self, data):
        KEYS = self.KEYS
        u_ids, u_inds, u_counts = np.unique(data[KEYS.ID], return_index=True, return_counts=True)
        # num_unique = u_ids.size
        data[KEYS.U_IDS] = u_ids
        data[KEYS.U_INDICES] = u_inds
        data[KEYS.U_COUNTS] = u_counts
        return data

    def _finalize_and_save(self, data, **header):
        KEYS = self.KEYS
        idx = np.lexsort((data[KEYS.SCALE], data[KEYS.ID]))
        count = idx.size
        for kk in KEYS:
            if kk in KEYS._DERIVED:
                continue

            temp = data[kk]
            temp = temp[idx, ...]
            data[kk] = temp

        data = self._add_derived(data)

        # -- Save values to file
        # Get output filename for this snapshot
        #   path should have already been created in `_process` by rank=0
        fname = self.filename
        self._save_to_hdf5(fname, KEYS, data, __file__, **header)
        if self._verbose:
            msg = "Saved data for {} BHs to '{}' size {}".format(
                count, fname, zio.get_file_size(fname))
            print(msg)

        return


class Snapshots_Snap(Snapshots):

    _PROCESSED_FILENAME = "bh-snapshots_{snap:03d}.hdf5"

    class KEYS(SNAP_KEYS):
        pass

    def __init__(self, snap, *args, **kwargs):
        self._snap = snap
        super().__init__(*args, **kwargs)
        return

    def _add_derived(self, data):
        return data

    def _process(self):
        KEYS = self.KEYS
        snap = self._snap

        # Check output filename
        fname = self.filename
        path = os.path.dirname(fname)
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.isdir(path):
            err = "ERROR: filename '{}' path '{}' is not a directory!".format(fname, path)
            logging.error(err)
            raise FileNotFoundError(err)

        pt_bh = illpy.PARTICLE.BH
        fields = [kk for kk in KEYS.keys() if kk not in KEYS._DERIVED]
        header = illpy.snapshot.get_header(self._sim_path, snap)
        bh_snap = illpy.snapshot.loadSubset(self._sim_path, snap, pt_bh, fields=fields, sq=False)
        num_bhs = bh_snap['count']
        if self._verbose:
            print("Loaded {} particles from snapshot {}".format(num_bhs, snap))

        if num_bhs == 0:
            print("Snapshot {} contains no BHs".format(snap))
            bh_snap = {}
            for kk in KEYS:
                bh_snap[kk] = np.array([])
        else:
            bh_snap[KEYS.SCALE] = header['Time'] * np.ones_like(bh_snap[KEYS.MASS])
            bh_snap[KEYS.SNAP] = snap * np.ones_like(bh_snap[KEYS.MASS], dtype=int)

        self._finalize_and_save(bh_snap, **header)
        return

    @property
    def filename(self):
        if self._filename is None:
            # `sim_path` has already been checked to exist in initializer
            sim_path = self._sim_path
            temp = self._PROCESSED_FILENAME.format(snap=self._snap)
            self._filename = os.path.join(sim_path, *PATH_PROCESSED, temp)

        return self._filename


def process_snapshot_snaps(sim_path, recreate=False, verbose=VERBOSE):
    try:
        comm = MPI.COMM_WORLD
    except NameError:
        logging.warning("Loading MPI...")
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    beg = datetime.now()
    if comm.rank == 0:
        temp = 'snapdir_' + '[0-9]' * 3
        temp = os.path.join(sim_path, temp)
        snap_dirs = sorted(glob.glob(temp))
        if verbose:
            print("Found {} snapshot directories".format(len(snap_dirs)))
        if len(snap_dirs) == 0:
            err = "ERROR: no snapshot directories found (pattern: '{}')".format(temp)
            logging.error(err)
            raise FileNotFoundError(err)

        snap_list = []
        prev = None
        for dd in snap_dirs:
            sn = int(os.path.basename(dd).split('_')[-1])
            snap_list.append(sn)
            if prev is None:
                if sn != 0:
                    err = "WARNING: first snapshot ({}) is not zero!".format(sn)
                    logging.warning(err)
            elif prev + 1 != sn:
                err = "WARNING: snapshot {} does not follow previous {}!".format(sn, prev)
                logging.warning(err)

            prev = sn

        np.random.seed(1234)
        np.random.shuffle(snap_list)
        snap_list = np.array_split(snap_list, comm.size)

    else:
        snap_list = None

    snap_list = comm.scatter(snap_list, root=0)

    comm.barrier()
    num_lines = 0
    for snap_num in snap_list:
        snap = Snapshots_Snap(snap_num, sim_path=sim_path, recreate=recreate, verbose=verbose)
        num_lines += snap[snap.KEYS.SCALE].size

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

    comm.barrier()
    return

