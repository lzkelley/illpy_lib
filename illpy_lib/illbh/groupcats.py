"""
"""

import os
# from datetime import datetime
import logging

import numpy as np

import zcode.inout as zio

import illpy.snapshot
from illpy import PARTICLE
import illpy_lib  # noqa
from illpy_lib import DTYPE
from illpy_lib.illbh import Processed, PATH_PROCESSED  # , VERBOSE
from illpy_lib.illbh.snapshots import SNAP_KEYS


class GCAT_KEYS(Processed.KEYS):

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
    ID                                  = 'id'     # blackhole ID numbers
    SUBHALO                             = 'subhalo'   # subhalo index number for this snapshot

    # NOTE: these are 'derived' in the sense for each snapshot: these parameters must be added
    #       when combining all snapshots together, these are not derived, they come from the data
    _DERIVED = [SCALE, SNAP, ID, SUBHALO]
    _ALIASES = []   # ID, MASS, BH_MASS, MDOT_B, MDOT, POT, DENS, POS, VEL]

    @classmethod
    def names(cls):
        nn = [kk for kk in dir(cls) if (not kk.startswith('_')) and (kk not in cls._ALIASES)]
        nn = [kk for kk in nn if (not callable(getattr(cls, kk)))]
        return sorted(nn)


class Groupcats(Processed):

    _PROCESSED_FILENAME = "bh-groupcats.hdf5"

    class KEYS(GCAT_KEYS):

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
                print("\nsnap = {}".format(snap))

            gcat_snap = Groupcats_Snap(snap, sim_path=sim_path)
            for kk in gcat_snap.KEYS:
                temp = gcat_snap[kk]
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
            # if kk in KEYS._DERIVED:
            #     continue
            if kk.startswith('unique'):
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


class Groupcats_Snap(Groupcats):

    _PROCESSED_FILENAME = "bh-groupcats_{snap:03d}.hdf5"

    class KEYS(GCAT_KEYS):
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
        gcat_fields = [kk for kk in KEYS.keys() if kk not in KEYS._DERIVED]
        snap_fields = [SNAP_KEYS.ID, SNAP_KEYS.POS, SNAP_KEYS.MASS]
        header = illpy.snapshot.get_header(self._sim_path, snap)
        bh_snap = illpy.snapshot.loadSubset(self._sim_path, snap, pt_bh,
                                            fields=snap_fields, sq=False)
        num_bhs = bh_snap['count']

        bh_gcat = {}
        if num_bhs == 0:
            print("Snapshot {} contains no BHs".format(snap))
            for kk in KEYS:
                bh_gcat[kk] = np.array([])
        else:
            groups = illpy.groupcat.loadSubhalos(self._sim_path, snap, fields=gcat_fields)
            num_grs = groups['count']

            halo_fields = ['GroupLenType', 'GroupNsubs', 'GroupFirstSub']
            halos = illpy.groupcat.loadHalos(self._sim_path, snap, fields=halo_fields)
            num_halos = halos['count']

            if self._verbose:
                print("Loaded {} BHs, {} halos, {} groups from snap {}".format(
                    num_bhs, num_halos, num_grs, snap))

            bh_subhalos = self._match(bh_snap, halos, groups)

            # Store catalog parameters
            idx = (bh_subhalos >= 0)
            bh_subhalos_idx = bh_subhalos[idx]
            nbad = np.count_nonzero(bh_subhalos < 0)
            if nbad > 0:
                print("{} bhs are unmatched to subhalos".format(nbad))
            for kk in KEYS:
                if (kk in KEYS._DERIVED) or (kk in KEYS._ALIASES):
                    continue
                gc_vals = groups[kk]
                shape = list(np.shape(gc_vals))
                shape[0] = num_bhs
                temp = np.zeros(shape, dtype=gc_vals.dtype)
                temp[idx] = gc_vals[bh_subhalos_idx]
                bh_gcat[kk] = temp

            # Store derived parameters
            bh_gcat[KEYS.SCALE] = header['Time'] * np.ones(num_bhs)
            bh_gcat[KEYS.SNAP] = snap * np.ones(num_bhs, dtype=int)
            bh_gcat[KEYS.ID] = bh_snap[SNAP_KEYS.ID]
            bh_gcat[KEYS.SUBHALO] = bh_subhalos

        self._finalize_and_save(bh_gcat, **header)
        return

    @property
    def filename(self):
        if self._filename is None:
            # `sim_path` has already been checked to exist in initializer
            sim_path = self._sim_path
            temp = self._PROCESSED_FILENAME.format(snap=self._snap)
            self._filename = os.path.join(sim_path, *PATH_PROCESSED, temp)

        return self._filename

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
        cat_inds_matched : (M,) int
            Indices for the BH snapshot particles which are valid and matched.
        snap_inds_matched : (M,) int
            Indices for the Subhalo catalog entries which are valid and matched.

        """
        bh_id = bh_snap[SNAP_KEYS.ID][:]
        bh_mass = bh_snap[SNAP_KEYS.MASS][:]
        bh_pos = bh_snap[SNAP_KEYS.POS][:]

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

        # bh_halos = halo_num[bin_inds]
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
                    "bh_pos = {}, subh_pos = {} ===> dist: {:.4e} ".format(
                        snap_pos, gcat_pos, dist) +
                    "(hmass,vmax rads = {:.4e}, {:4e})".format(
                        hm_rad, max_rad)
                )
                logging.error(err)
                logging.error("bh_mass = {:.4e}, subh_bh_mass = {:.4e} ===> diff: {:.4e}".format(
                    bh_mass[bh], gcat_mass, mdiff))
                err = "Match between BH {} (ID: {}) and subhalo {} looks bad!".format(
                    bh, bh_id[bh], sh
                )
                logging.warning(err)
                # raise ValueError(err)

        return bh_subhalos


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


'''
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
'''
