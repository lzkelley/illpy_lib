"""
"""
import sys
import glob
import os
from datetime import datetime
import logging

import numpy as np

import zcode.inout as zio

import illpy.snapshot
from illpy import PARTICLE
import illpy_lib  # noqa
from illpy_lib import DTYPE
from illpy_lib.illbh import Processed, ENUM, utils, KEYS, VERBOSE


class GCAT_KEYS(ENUM):

    SubhaloBHMass                       = 'SubhaloBHMass'
    SubhaloBHMdot                       = 'SubhaloBHMdot'
    SubhaloBfldDisk                     = 'SubhaloBfldDisk'
    SubhaloBfldHalo                     = 'SubhaloBfldHalo'
    SubhaloCM                           = 'SubhaloCM'
    SubhaloGasMetalFractions            = 'SubhaloGasMetalFractions'
    SubhaloGasMetalFractionsHalfRad     = 'SubhaloGasMetalFractionsHalfRad'
    SubhaloGasMetalFractionsMaxRad      = 'SubhaloGasMetalFractionsMaxRad'
    SubhaloGasMetalFractionsSfr         = 'SubhaloGasMetalFractionsSfr'
    SubhaloGasMetalFractionsSfrWeighted = 'SubhaloGasMetalFractionsSfrWeighted'
    SubhaloGasMetallicity               = 'SubhaloGasMetallicity'
    SubhaloGasMetallicityHalfRad        = 'SubhaloGasMetallicityHalfRad'
    SubhaloGasMetallicityMaxRad         = 'SubhaloGasMetallicityMaxRad'
    SubhaloGasMetallicitySfr            = 'SubhaloGasMetallicitySfr'
    SubhaloGasMetallicitySfrWeighted    = 'SubhaloGasMetallicitySfrWeighted'
    SubhaloGrNr                         = 'SubhaloGrNr'
    SubhaloHalfmassRad                  = 'SubhaloHalfmassRad'
    SubhaloHalfmassRadType              = 'SubhaloHalfmassRadType'
    SubhaloIDMostbound                  = 'SubhaloIDMostbound'
    SubhaloLen                          = 'SubhaloLen'
    SubhaloLenType                      = 'SubhaloLenType'
    SubhaloMass                         = 'SubhaloMass'
    SubhaloMassInHalfRad                = 'SubhaloMassInHalfRad'
    SubhaloMassInHalfRadType            = 'SubhaloMassInHalfRadType'
    SubhaloMassInMaxRad                 = 'SubhaloMassInMaxRad'
    SubhaloMassInMaxRadType             = 'SubhaloMassInMaxRadType'
    SubhaloMassInRad                    = 'SubhaloMassInRad'
    SubhaloMassInRadType                = 'SubhaloMassInRadType'
    SubhaloMassType                     = 'SubhaloMassType'
    SubhaloParent                       = 'SubhaloParent'
    SubhaloPos                          = 'SubhaloPos'
    SubhaloSFR                          = 'SubhaloSFR'
    SubhaloSFRinHalfRad                 = 'SubhaloSFRinHalfRad'
    SubhaloSFRinMaxRad                  = 'SubhaloSFRinMaxRad'
    SubhaloSFRinRad                     = 'SubhaloSFRinRad'
    SubhaloSpin                         = 'SubhaloSpin'
    SubhaloStarMetalFractions           = 'SubhaloStarMetalFractions'
    SubhaloStarMetalFractionsHalfRad    = 'SubhaloStarMetalFractionsHalfRad'
    SubhaloStarMetalFractionsMaxRad     = 'SubhaloStarMetalFractionsMaxRad'
    SubhaloStarMetallicity              = 'SubhaloStarMetallicity'
    SubhaloStarMetallicityHalfRad       = 'SubhaloStarMetallicityHalfRad'
    SubhaloStarMetallicityMaxRad        = 'SubhaloStarMetallicityMaxRad'
    SubhaloStellarPhotometrics          = 'SubhaloStellarPhotometrics'
    SubhaloStellarPhotometricsMassInRad = 'SubhaloStellarPhotometricsMassInRad'
    SubhaloStellarPhotometricsRad       = 'SubhaloStellarPhotometricsRad'
    SubhaloVel                          = 'SubhaloVel'
    SubhaloVelDisp                      = 'SubhaloVelDisp'
    SubhaloVmax                         = 'SubhaloVmax'
    SubhaloVmaxRad                      = 'SubhaloVmaxRad'
    SubhaloWindMass                     = 'SubhaloWindMass'

    SCALE                               = 'scale'
    SNAP                                = 'snap'
    # ID                                  = 'id'        # blackhole ID numbers
    ID                                  = 'ParticleIDs'        # blackhole ID numbers
    SUBHALO                             = 'subhalo'   # subhalo index number for this snapshot
    HALO                                = 'halo'      # halo index number for this snapshot

    # Added keys for derived parameters
    U_IDS = 'unique_ids'
    U_INDICES = 'unique_indices'
    U_COUNTS = 'unique_counts'

    # _INTERP_KEYS = [MASS, BH_MASS, MDOT, MDOT_B, POT, DENS, POS, VEL]
    _U_KEYS = [U_IDS, U_INDICES, U_COUNTS]

    # NOTE: these are 'derived' in the sense for each snapshot: these parameters must be added
    #       when combining all snapshots together, these are not derived, they come from the data
    _DERIVED = [SCALE, SNAP, ID, SUBHALO, HALO] + _U_KEYS
    _ALIASES = []   # ID, MASS, BH_MASS, MDOT_B, MDOT, POT, DENS, POS, VEL]

    # @classmethod
    # def names(cls):
    #     nn = [kk for kk in dir(cls) if (not kk.startswith('_')) and (kk not in cls._ALIASES)]
    #     nn = [kk for kk in nn if (not callable(getattr(cls, kk)))]
    #     return sorted(nn)


class Groupcats(Processed):

    _PROCESSED_FILENAME = "bh-groupcats.hdf5"
    _SKIP_KEYS = []

    def _process(self):
        # Check output filename
        fname = self.filename()
        utils._check_output_path(fname)
        verbose = self._verbose

        gcat = {}
        snap = 0
        count = 0
        warn_first_skip_key = verbose
        while (snap < 10000):
            if verbose:
                print("snap = {}".format(snap))
            # Keep loading snapshots until we fail, assume that's the last snapshot
            try:
                gcat_snap = Groupcats_Snap(
                    snap, self._sim_path, self._processed_path,
                    verbose=verbose, load=True, must_exist=True
                )
            except OSError as err:
                if verbose:
                    print("Failed to load snap {} : {}".format(snap, str(err)))
                print(f"finished after snap {snap-1}")
                break
                
            if len(gcat_snap.keys()) == 0:
                logging.info(f"groupcat from {snap=} is empty")
            else:
                count = gcat_snap[GCAT_KEYS.SCALE].size

            for kk in gcat_snap.keys():
                if kk in self._SKIP_KEYS:
                    if warn_first_skip_key:
                        msg = "Skipping key '{}' and all of '{}'".format(kk, self._SKIP_KEYS)
                        logging.warning(msg)
                    warn_first_skip_key = False
                    continue

                temp = gcat_snap[kk]
                prev = gcat.get(kk, None)
                if (prev is None) or (len(prev) == 0):
                    gcat[kk] = temp
                else:
                    gcat[kk] = np.concatenate([prev, temp])

            snap += 1

        if verbose:
            print("Loaded {} entries from {} snaps".format(count, snap))

        if len(gcat) == 0:
            err = f"After {snap} snaps, `gcat` is empty!"
            logging.error(err)
            raise RuntimeError(err)

        if 'id' in gcat.keys():
            gcat[GCAT_KEYS.ID] = gcat.pop('id')

        self._finalize_and_save(gcat)
        return

    def _add_derived(self, data):
        u_ids, u_inds, u_counts = np.unique(data[GCAT_KEYS.ID], return_index=True, return_counts=True)
        # num_unique = u_ids.size
        data[GCAT_KEYS.U_IDS] = u_ids
        data[GCAT_KEYS.U_INDICES] = u_inds
        data[GCAT_KEYS.U_COUNTS] = u_counts
        return data

    def _finalize_and_save(self, data, **header):
        idx = np.lexsort((data[GCAT_KEYS.SCALE], data[GCAT_KEYS.ID]))
        count = idx.size
        for kk in GCAT_KEYS:
            # if kk in KEYS._DERIVED:
            #     continue
            # Don't want 'unique' keys from each snap; recalculate them for ALL snapshots below
            if kk.startswith('unique'):
                continue

            temp = data[kk]
            temp = temp[idx, ...]
            data[kk] = temp

        # Add unique entries here
        data = self._add_derived(data)

        # -- Save values to file
        fname = self.filename()
        fname = utils._check_output_path(fname)
        self._save_to_hdf5(fname, __file__, data, keys=GCAT_KEYS, **header)
        if self._verbose:
            msg = f"Saved data for {count} BHs to '{fname}' size {zio.get_file_size(fname)}"
            print(msg)

        return


class Groupcats_Snap(Groupcats):

    _PROCESSED_FILENAME = "bh-groupcats_{snap:03d}.hdf5"
    _PROCESSED_DIR_NAME = "bh-groupcats"

    def __init__(self, snap, *args, load=True, **kwargs):
        self._snap = snap
        super().__init__(*args, load=load, **kwargs)
        return

    def _add_derived(self, data):
        return data

    def _process(self):
        snap = self._snap

        # Check output filename
        fname = self.filename()
        fname = utils._check_output_path(fname)

        pt_bh = illpy.PARTICLE.BH
        gcat_fields = [kk for kk in GCAT_KEYS.keys() if kk not in GCAT_KEYS._DERIVED]
        snap_fields = [KEYS.ID, KEYS.POS, KEYS.MASS]
        path = os.path.join(self._sim_path, 'output', '')
        header = illpy.snapshot.get_header(path, snap)
        if self._verbose:
            print(f"Loading {snap=:03d} (a={header['Time']:.8f})")
        bh_snap = illpy.snapshot.loadSubset(path, snap, pt_bh, fields=snap_fields, sq=False)
        num_bhs = bh_snap['count']
        if self._verbose:
            print(f"\tloaded {num_bhs=} from snapshot")

        bh_gcat = {}
        if num_bhs == 0:
            print("Snapshot {} contains no BHs".format(snap))
            for kk in GCAT_KEYS:
                bh_gcat[kk] = np.array([])
        else:
            groups = illpy.groupcat.loadSubhalos(path, snap, fields=gcat_fields)
            num_grs = groups['count']
            if self._verbose:
                print(f"\tloaded {num_grs} subhalos")

            halo_fields = ['GroupLenType', 'GroupNsubs', 'GroupFirstSub']
            halos = illpy.groupcat.loadHalos(path, snap, fields=halo_fields)
            num_halos = halos['count']
            if self._verbose:
                print(f"\tloaded {num_halos} halos")

            # Match BHs to parent (sub)halos
            bh_subhalos, bh_halos = self._match(bh_snap, halos, groups)

            # Store catalog parameters
            idx = (bh_subhalos >= 0)
            bh_subhalos_idx = bh_subhalos[idx]
            nbad = np.count_nonzero(bh_subhalos < 0)
            if nbad > 0:
                print("{} bhs are unmatched to subhalos".format(nbad))
            for kk in GCAT_KEYS:
                if (kk in GCAT_KEYS._DERIVED) or (kk in GCAT_KEYS._ALIASES):
                    continue
                gc_vals = groups[kk]
                shape = list(np.shape(gc_vals))
                shape[0] = num_bhs
                temp = np.zeros(shape, dtype=gc_vals.dtype)
                temp[idx] = gc_vals[bh_subhalos_idx]
                bh_gcat[kk] = temp

            # Store derived parameters
            bh_gcat[GCAT_KEYS.SCALE] = header['Time'] * np.ones(num_bhs)
            bh_gcat[GCAT_KEYS.SNAP] = snap * np.ones(num_bhs, dtype=int)
            bh_gcat[GCAT_KEYS.ID] = bh_snap[KEYS.ID]
            bh_gcat[GCAT_KEYS.SUBHALO] = bh_subhalos
            bh_gcat[GCAT_KEYS.HALO] = bh_halos

        self._finalize_and_save(bh_gcat, **header)
        return

    def filename(self):
        temp = self._PROCESSED_FILENAME.format(snap=self._snap)
        fname = os.path.join(self._processed_path, self._PROCESSED_DIR_NAME, temp)
        return fname

    def _match(self, bh_snap, halos, groups):
        """Use BH ID numbers to find parent subhalos.

        Arguments
        ---------
        snap : dict
            Dictionary of snapshot BH-particle data from `illustris_python.snapshot`
        gcat : dict
            Dictionary of subhalo catalog data from `illustris_python.subhalos`

        Returns
        -------
        bh_subhalos: (M,) int
        bh_halos: (M,) int

        """
        bh_id = bh_snap[KEYS.ID][:]
        bh_mass = bh_snap[KEYS.MASS][:]
        bh_pos = bh_snap[KEYS.POS][:]

        num_bh = len(bh_id)
        # Create 'indices' of BHs
        bh_indices = np.arange(num_bh)

        if np.any(bh_mass <= 0.0):
            raise ValueError("Found non-positive BH masses in snapshot!")

        halo_num, subh_num, offsets = _construct_offset_table(halos, groups)
        # On success, Find BH Subhalos
        bin_inds = np.digitize(bh_indices, offsets[:, PARTICLE.BH]).astype(DTYPE.INDEX) - 1
        if any(bin_inds < 0):
            raise ValueError("Some bh_inds not matched!! '{}'".format(str(bin_inds)))

        bh_halos = halo_num[bin_inds]
        bh_subhalos = subh_num[bin_inds]
        if np.any(bh_subhalos < 0):
            print("\nFound unmatched BHs")
            print("BHs = ", num_bh, "LenType = ", np.sum(groups['SubhaloLenType'], axis=0))
            """
            print("bh_indices = ", bh_indices)
            print("offsets = ", offsets[:, PARTICLE.BH])
            print("bin_inds = ", bin_inds)
            print("bh_subhalos = ", bh_subhalos)
            print()
            """

        for bh, sh in enumerate(bh_subhalos):
            if sh < 0:
                print("BH {} has subhalo {}, i.e. no match".format(bh, sh))
                continue

            sh_nbh = groups['SubhaloLenType'][sh, PARTICLE.BH]

            snap_pos = bh_pos[bh]
            gcat_pos = groups[GCAT_KEYS.SubhaloPos][sh]
            dist = np.linalg.norm(snap_pos - gcat_pos)

            vmax_rad = groups['SubhaloVmaxRad'][sh]
            hm_rad = groups['SubhaloHalfmassRad'][sh]
            max_rad = np.max([vmax_rad, hm_rad])

            gcat_mass = groups[GCAT_KEYS.SubhaloMassType][sh, PARTICLE.BH]
            mdiff = gcat_mass - bh_mass[bh]

            gcat_idmb = groups['SubhaloIDMostbound'][sh]
            # print("\nBH: {} (ID: {}) ==> subh: {} (nbh: {})".format(bh, bh_id[bh], sh, sh_nbh))
            # print("subh idmb: {}".format(gcat_idmb))
            # print("snap_pos = {}".format(snap_pos))
            # print("gcat_pos = {}".format(gcat_pos))
            # print("dist = {:.4e}".format(dist))
            # print("snap_mass = {}".format(bh_mass[bh]))
            # print("gcat_mass = {}".format(gcat_mass))
            # print("diff = {:.4e}".format(mdiff))

            if (dist > 4 * max_rad) or (mdiff < 0.0):
                logging.error("BH: {}, ID: {}, subhalo: {} (Nbh: {}, most bound: {})".format(
                    bh, bh_id[bh], sh, sh_nbh, gcat_idmb))
                err = (
                    f"bh_pos = {snap_pos}, subh_pos = {gcat_pos} ===> dist: {dist:.4e} "
                    f"(hmass,vmax rads = {hm_rad:.4e}, {max_rad:4e})"
                )
                logging.error(err)
                logging.error("bh_mass = {:.4e}, subh_bh_mass = {:.4e} ===> diff: {:.4e}".format(
                    bh_mass[bh], gcat_mass, mdiff))
                err = "Match between BH {} (ID: {}) and subhalo {} looks bad!".format(
                    bh, bh_id[bh], sh
                )
                logging.warning(err)
                # raise ValueError(err)

        return bh_subhalos, bh_halos


def _construct_offset_table(halos, groups):
    """Construct offset table from halo and subhalo catalogs.

    Each 'entry' is the first particle index number for a group of particles.  Particles are
    grouped by the halos and subhalos they belong to.  The first entry is particles in the first
    subhalo of the first halo.  The last entry for the first halo is particles that dont belong to
    any subhalo (but still belong to the first halo).  The very last entry is for particles that
    dont belong to any halo or subhalo.

    """

    # Initialize Storage
    # ------------------

    num_halos = halos['count']
    num_subhs = groups['count']

    table_size = num_halos + num_subhs + 1

    # See object description; recall entries are [HALO, SUBHALO, PART0, ... PART5]
    #    (Sub)halo numbers are smaller, use signed-integers for `-1` to be no (Sub)halo
    halo_num = np.zeros(table_size, dtype=DTYPE.INDEX)
    subh_num = np.zeros(table_size, dtype=DTYPE.INDEX)
    # Offsets approach total number of particles, must be uint64
    offsets = np.zeros([table_size, PARTICLE.NUM()], dtype=DTYPE.ID)

    subh = 0
    offs = 0
    cum_halo_parts = np.zeros(PARTICLE.NUM(), dtype=DTYPE.ID)
    cum_subh_parts = np.zeros(PARTICLE.NUM(), dtype=DTYPE.ID)

    # Iterate Over Each Halo
    # ----------------------
    for ii in range(num_halos):

        # Add the number of particles in this halo
        temp = halos['GroupLenType'][ii, :].astype(DTYPE.ID)
        temp = temp.astype(DTYPE.ID)
        cum_halo_parts[:] += temp

        # Iterate over each Subhalo, in halo ``ii``
        # -----------------------------------------
        for jj in range(halos['GroupNsubs'][ii]):

            # Consistency check: make sure subhalo number is as expected
            if (jj == 0) and (subh != halos['GroupFirstSub'][ii]):
                logging.error("ii = {:d}, jj = {:d}, subh = {:d}".format(ii, jj, subh))
                raise ValueError("Subhalo iterator doesn't match Halo's first subhalo!")

            # Add entry for each subhalo
            halo_num[offs] = ii
            subh_num[offs] = subh
            offsets[offs, :] = cum_subh_parts

            # Add particles in this subhalo to offset counts
            temp = groups['SubhaloLenType'][subh, :]
            cum_subh_parts[:] += temp.astype(DTYPE.ID)

            # Increment subhalo and entry number
            subh += 1
            offs += 1

        # Add Entry for particles with NO subhalo
        halo_num[offs] = ii                        # Still part of halo ``ii``
        subh_num[offs] = -1                        # `-1` means no (sub)halo
        offsets[offs, :] = cum_subh_parts

        # Increment particle numbers to include this halo
        cum_subh_parts = np.copy(cum_halo_parts)

        # Increment entry number
        offs += 1

    # Add entry for particles with NO halo and NO subhalo
    halo_num[offs] = -1
    subh_num[offs] = -1
    offsets[offs, :] = cum_subh_parts

    return halo_num, subh_num, offsets


def process_groupcats_snaps(sim_path, proc_path, recreate=False, verbose=VERBOSE):
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
        # bad_snaps = None
        # if mode == 'tos':
        #     bad_snaps = _BAD_SNAPS_TOS
        # elif mode == 'tng':
        #     pass
        # elif mode == 'new':
        #     pass
        # else:
        #     raise ValueError("Unrecognized `mode` '{}'!".format(mode))

        # if bad_snaps is not None:
        #     logging.warning("Removing TOS bad snaps '{}' from targets".format(bad_snaps))
        #     for bsn in bad_snaps:
        #         snap_list.remove(bsn)
        #         assert (bsn not in snap_list), "FAILED"

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
            snap = Groupcats_Snap(
                snap_num, sim_path=sim_path, processed_path=proc_path,
                recreate=recreate, verbose=verbose, load=True)
        except Exception as err:
            logging.error("FAILED to load snap '{}'!".format(snap_num))
            logging.error(str(err))
            raise err

        if len(snap.keys()) > 0:
            num_lines += snap[GCAT_KEYS.SCALE].size

    num_lines = comm.gather(num_lines, root=0)

    end = datetime.now()
    if verbose:
        print("Rank: {}, done at {}, after {}".format(comm.rank, end, (end - beg)))
    if comm.rank == 0:
        tot = np.sum(num_lines)
        ave = np.mean(num_lines)
        med = np.median(num_lines)
        std = np.std(num_lines)
        if verbose:
            print(f"Tot lines={tot:.2e}, med={med:.2e}, ave={ave:.2e}±{std:.2e}")

        tail = f"Done at {str(end)} after {str(end-beg)} ({(end-beg).total_seconds()})"
        print("\n" + "=" * len(tail) + "\n" + tail + "\n")
        logging.warning(tail)

    comm.barrier()
    return


if __name__ == "__main__":
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
            # logging.warning("USAGE: `python {} <MODE> <PATH_IN> <PATH_OUT>`\n\n".format(__file__))
            logging.warning(f"USAGE: `python {__file__} <PATH_IN> <PATH_OUT>`\n\n")
            sys.exit(0)

        cc = 1
        # mode = sys.argv[cc]
        # cc += 1
        sim_path = os.path.abspath(sys.argv[cc]).rstrip('/')
        cc += 1
        proc_path = os.path.abspath(sys.argv[cc]).rstrip('/')
        cc += 1

        # mode = mode.strip().lower()
        # print("mode   '{}'".format(mode))
        print("input  `sim_path`  : {}".format(sim_path))
        print("output `proc_path` : {}".format(proc_path))
        # valid_modes = ['tos', 'tng', 'new']
        # if mode not in valid_modes:
        #     err = "Unrecognized `mode` '{}'!".format(mode)
        #     logging.error(err)
        #     raise ValueError(err)

    else:
        # mode = None
        sim_path = None
        proc_path = None

    # mode = comm.bcast(mode, root=0)
    sim_path = comm.bcast(sim_path, root=0)
    proc_path = comm.bcast(proc_path, root=0)
    comm.barrier()

    # process_snapshot_snaps(mode, sim_path, proc_path, recreate=recreate, verbose=verbose)
    process_groupcats_snaps(sim_path, proc_path, recreate=recreate, verbose=verbose)
