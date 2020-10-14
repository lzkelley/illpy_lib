"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import logging
import tqdm

import numpy as np
# import h5py

import parse

import zcode.inout as zio
import zcode.math as zmath

import illpy_lib  # noqa
from illpy_lib.illbh import BH_TYPE, Processed  # , utils


VERSION = 0.5


class Mergers(Processed):

    _PROCESSED_FILENAME = "bh-mergers.hdf5"

    class KEYS(Processed.KEYS):
        SCALE = 'scale'
        ID   = 'id'
        MASS = 'mass'

        # NEXT = 'next'
        # PREV = 'prev'
        #
        # NUM_NEXT = 'num_next'
        # NUM_PREV = 'num_prev'

        U_IDS = 'unique_ids'
        U_INDICES = 'unique_indices'

        _DERIVED = [U_IDS, U_INDICES]

    def _parse_raw_file_line(self, line):
        KEYS = self.KEYS

        format = (
            "{time:g} {task:d}  " +
            "{id_o:d} {mass_o:g} " +
            "{px_o:g} {py_o:g} {pz_o:g}  " +
            "{id_i:d} {mass_i:g} " +
            "{px_i:g} {py_i:g} {pz_i:g}"
        )

        format_mass = (
            "{time:g} {task:d}  " +
            "{id_o:d} {mass_o:g} {bh_mass_o:g} " +
            "{px_o:g} {py_o:g} {pz_o:g}  " +
            "{id_i:d} {mass_i:g} {bh_mass_i:g} " +
            "{px_i:g} {py_i:g} {pz_i:g}"
        )

        format_mass_vel = (
            "{time:g} {task:d}  " +
            "{id_o:d} {mass_o:g} {bh_mass_o:g} " +
            "{px_o:g} {py_o:g} {pz_o:g} {vx_o:g} {vy_o:g} {vz_o:g}  " +
            "{id_i:d} {mass_i:g} {bh_mass_i:g} " +
            "{px_i:g} {py_i:g} {pz_i:g} {vx_i:g} {vy_i:g} {vz_i:g}"
        )

        for form in [format_mass_vel, format_mass, format]:
            temp = parse.parse(form, line)
            if temp is not None:
                break
        else:
            err = "ERROR: failed to parse line '{}'".format(line)
            logging.error(err)
            raise ValueError(err)

        ids = np.zeros(2, dtype=np.uint64)
        ids[BH_TYPE.OUT] = temp['id_o']
        ids[BH_TYPE.IN] = temp['id_i']

        mass = np.zeros(2)
        mass[BH_TYPE.OUT] = temp['mass_o']
        mass[BH_TYPE.IN] = temp['mass_i']

        pos = np.zeros((3, 2))
        pos[:, BH_TYPE.OUT] = [temp['px_o'], temp['py_o'], temp['pz_o']]
        pos[:, BH_TYPE.IN] = [temp['px_i'], temp['py_i'], temp['pz_i']]

        bh_mass = np.zeros(2)
        if 'bh_mass_o' in temp:
            bh_mass[BH_TYPE.OUT] = temp['bh_mass_o']
            bh_mass[BH_TYPE.IN] = temp['bh_mass_i']

        vel = np.zeros((3, 2))
        if 'vx_o' in temp:
            vel[:, BH_TYPE.OUT] = [temp['vx_o'], temp['vy_o'], temp['vz_o']]
            vel[:, BH_TYPE.IN] = [temp['vx_i'], temp['vy_i'], temp['vz_i']]

        rv = {
            KEYS.SCALE: temp['time'],
            KEYS.TASK: temp['task'],

            KEYS.ID: ids,
            KEYS.MASS: mass,
            KEYS.BH_MASS: bh_mass,
            KEYS.POS: pos,
            KEYS.VEL: vel
        }

        return rv

    def _get_merger_file_list(self, sim_path, cosmo=None):
        raise NotImplementedError()

    def _clean_raw_mergers(self, mrgs):
        return mrgs

    def _load_raw_mergers(self):
        KEYS = self.KEYS
        sim_path = self._sim_path

        if sim_path is None:
            err = "ERROR: `sim_path` is not set, cannot load raw mergers!"
            logging.error(err)
            raise ValueError(err)

        if not os.path.isdir(sim_path):
            err = "ERROR: `sim_path` '{}' does not exist!".format(sim_path)
            logging.error(err)
            raise ValueError(err)

        '''
        cosmo = illpy_lib.illcosmo.load_sim_cosmo(sim_path, verbose=self._verbose)
        if cosmo is None:
            msg = "WARNING: could not load cosmology"
            logging.warning(msg)
        '''
        cosmo = None

        merger_files = self._get_merger_file_list(sim_path, cosmo=cosmo)
        if self._verbose:
            print("Found {} merger files".format(len(merger_files)))

        # --- Go through all files compiling a list of dict for each merger ---
        mergers = []
        first = True
        for ii, mf in enumerate(tqdm.tqdm(merger_files, desc='files')):
            prev = None
            for ll, line in enumerate(open(mf, 'r').readlines()):
                if first:
                    print("Example merger line: '{}'".format(line.strip()))
                    first = False

                try:
                    vals = self._parse_raw_file_line(line)
                except:
                    logging.error("FAILED on line {} of file '{}'".format(ll, mf))
                    raise

                sc = vals[KEYS.SCALE]
                if prev is None:
                    prev = sc
                elif sc < prev:
                    err = "file: {}, {} - line: {} - scale:{:.8f} is before previous:{:.8f}!"
                    err = err.format(ii, mf, ll, sc, prev)
                    logging.error(err)
                    raise ValueError(err)

                mergers.append(vals)

        # --- Convert to dict of arrays for all mergers ---
        num_mrgs = len(mergers)
        mrgs = {}
        for mm, temp in enumerate(mergers):
            if mm == 0:
                for kk, vv in temp.items():
                    if np.isscalar(vv):
                        shp = num_mrgs
                        tt = type(vv)
                    else:
                        shp = (num_mrgs,) + np.shape(vv)
                        tt = vv.dtype

                    mrgs[kk] = np.zeros(shp, dtype=tt)

            for kk, vv in temp.items():
                mrgs[kk][mm, ...] = vv

        idx = np.argsort(mrgs['scale'])
        for kk, vv in mrgs.items():
            mrgs[kk] = vv[idx, ...]

        if self._verbose:
            print("Loaded {} raw mergers".format(idx.size))

        return mrgs

    def _process(self):
        # Check output filename
        fname = self.filename
        path = os.path.dirname(fname)
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.isdir(path):
            err = "ERROR: filename '{}' path '{}' is not a directory!".format(fname, path)
            logging.error(err)
            raise FileNotFoundError(err)

        # Load merger events directly from Illustris blackhole_mergers txt files, unmodified
        mrgs = self._load_raw_mergers()
        if self._verbose:
            print("Loaded ", mrgs['scale'].size, "mergers!")

        # Process mergers
        mrgs = self._clean_raw_mergers(mrgs)

        # Construct merger tree ('next' and 'prev' arrays) and do some consistency checks
        logging.warning("WARNING: skipping merger tree")
        '''
        mrgs = self._build_tree(mrgs)
        if self._verbose:
            print("Merger tree constructed with {} next and {} prev links".format(
                np.count_nonzero(mrgs['next'] >= 0), np.count_nonzero(mrgs['prev'] >= 0)))
        '''

        # -- Identify Unique BHs based on ID numbers
        KEYS = self.KEYS
        mids = mrgs[KEYS.ID]
        # Convert from (M, 2) ==> (2*M,), i.e. [ID0_pri, ID0_sec, ID1_pri, ID1_sec ...]
        mids = mids.flatten()

        # Find unique indices, and first occurence of each index
        u_ids, u_inds = np.unique(mids, return_index=True)
        # Convert from index number in array of length (2*M,) to index number in mergers (M,)
        u_inds = u_inds // 2

        mrgs[KEYS.U_IDS] = u_ids
        mrgs[KEYS.U_INDICES] = u_inds
        if self._verbose:
            print("Identified {} unique BHs in {} mergers".format(u_ids.size, mids.size))

        # Save values to file
        self._save_to_hdf5(fname, self.KEYS, mrgs, __file__)
        if self._verbose:
            msg = "Saved data for {} mergers to '{}' size {}".format(
                len(mrgs[self.KEYS.SCALE]), fname, zio.get_file_size(fname))
            print(msg)

        return

    def _build_tree(self, mrgs):
        """
        The first BH is the "out" (remaining) BH, second BH is "in" (removed) BH
        """
        KEYS = self.KEYS

        def mstr(mm):
            msg = "{:6d} a={:.8f} task={:3d} ".format(mm, mrgs[KEYS.SCALE][mm], mrgs[KEYS.TASK][mm])
            ids = mrgs[KEYS.ID][mm]
            mass = mrgs[KEYS.MASS][mm]
            bh_mass = mrgs[KEYS.BH_MASS][mm]
            for name, tt in zip([' in', 'out'], [BH_TYPE.IN, BH_TYPE.OUT]):
                try:
                    msg += "{} id={:14d} bh_mass={:.8f} mass={:.8f} ".format(
                        name, ids[tt], bh_mass[tt], mass[tt])
                except:
                    print("ids  = ", ids[tt], type(ids[tt]))
                    print("mass = ", mass[tt], type(mass[tt]))
                    raise

            return msg

        # -- Build Merger Tree

        num = mrgs['scale'].size
        next = np.ones(num, np.int32) * -1
        prev = np.ones((num, 2), np.int32) * -1
        use_bh_mass = True
        for mm in tqdm.tqdm(range(num), 'mergers'):
            mids = mrgs[KEYS.ID][mm]
            ma = mrgs[KEYS.SCALE][mm]
            if use_bh_mass:
                mass = mrgs[KEYS.BH_MASS][mm]
                if np.all(mass == 0.0):
                    mass = None
                    logging.warning("WARNING: ")
            else:
                mass = mrgs[KEYS.MASS][mm]

            for nn in range(mm+1, num):
                nids = mrgs[KEYS.ID][nn]
                na = mrgs[KEYS.SCALE][nn]
                # Make sure scales are not decreasing (data is already sorted; should be impossible)
                if na < ma:
                    err = f"ERROR: scale regression: {mm}={ma} ==> {nn}={na}!"
                    logging.error(err)
                    raise ValueError(err)

                # make sure 'in' BH is never in another merger
                if mids[BH_TYPE.IN] in nids:
                    err = "ERROR: 'in' BH from merger {} found in merger {}!".format(mm, nn)
                    logging.error(err)
                    logging.error(mstr(mm))
                    logging.error(mstr(nn))
                    # raise ValueError(err)

                # Check if 'out' BH is in a subsequent merger
                if mids[BH_TYPE.OUT] in nids:
                    next[mm] = nn
                    mass_next = mrgs[KEYS.MASS][nn]
                    for jj in range(2):
                        if mids[BH_TYPE.OUT] == nids[jj]:
                            prev[nn, jj] = mm
                            # Make sure masses increased
                            if mass_next[jj] < mass[BH_TYPE.OUT]:
                                err = "ERROR: BH mass loss between merger {} and {}".format(mm, nn)
                                logging.error(err)
                                logging.error(mstr(mm))
                                logging.error(mstr(nn))
                                if use_bh_mass:
                                    raise ValueError(err)
                                else:
                                    logging.error("Using `Mass` not `BH_Mass`... continuing!")

                            break  # if

                    break  # for jj

        mrgs[KEYS.NEXT] = next
        mrgs[KEYS.PREV] = prev

        num_next = np.ones(num, np.int32) * -1
        num_prev = np.ones((num, 2), np.int32) * -1

        def _count_prev(mm):
            for bh in range(2):
                temp = prev[mm][bh]
                if temp >= 0:
                    num_prev[mm, bh] = _count_prev(temp)
                else:
                    num_prev[mm, bh] = 0

            return num_prev[mm].sum() + 1

        for mm in reversed(range(num)):
            temp = next[mm]
            if temp >= 0:
                num_next[mm] = num_next[temp] + 1
                continue

            num_next[mm] = 0
            _count_prev(mm)

        mrgs[KEYS.NUM_NEXT] = num_next
        mrgs[KEYS.NUM_PREV] = num_prev

        # -- Report statistics

        if self._verbose:
            num_next = (next >= 0)
            print("Mergers with subsequent mergers: {}".format(zmath.frac_str(num_next)))
            print("Mergers that are final remnants: {}".format(zmath.frac_str(~num_next)))

            mult = num_prev[num_prev >= 0]
            print("Remnant merger-multiplicity:")
            print("\tzero = {}".format(zmath.frac_str(mult == 0)))
            print("\tave  = {:.2f}".format(mult.mean()))
            print("\tmed  = {:.2f}".format(np.median(mult)))
            print("\t{}".format(zmath.stats_str(mult)))

        return mrgs


class Mergers_TNG(Mergers):

    class KEYS(Mergers.KEYS):
        TASK = 'task'
        BH_MASS = 'bh_mass'    # NOTE: the 'in' BH has 'BH_Mass' but the 'out' uses reg. 'mass'

    def _get_merger_file_list(self, sim_path, cosmo=None):
        mfil_pattern = 'blackhole_mergers_*.txt'
        mfil_pattern = os.path.join(sim_path, 'output', 'blackhole_mergers', mfil_pattern)
        merger_files = sorted(glob.glob(mfil_pattern))
        return merger_files

    def _parse_raw_file_line(self, line):
        """
        TNG:

        fprintf(FdBlackHolesMergers, "%d %g %llu %g %llu %g\n",
                ThisTask, All.Time, (long long) id, mass, (long long) P[no].ID, BPP(no).BH_Mass);
        """
        KEYS = self.KEYS

        format = (
            "{task:d} {scale:g} " +
            "{id_o:d} {mass_o:g} " +
            "{id_i:d} {bh_mass_i:g}"
        )

        temp = parse.parse(format, line)
        if temp is None:
            err = "ERROR: failed to parse line '{}'".format(line)
            logging.error(err)
            raise ValueError(err)

        ids = np.zeros(2, dtype=np.uint64)
        ids[BH_TYPE.OUT] = temp['id_o']
        ids[BH_TYPE.IN] = temp['id_i']

        mass = np.zeros(2)
        mass[BH_TYPE.OUT] = temp['mass_o']
        # mass[BH_TYPE.IN] = temp['mass_i']

        bh_mass = np.zeros(2)
        # bh_mass[BH_TYPE.OUT] = temp['bh_mass_o']
        bh_mass[BH_TYPE.IN] = temp['bh_mass_i']

        mrg = {
            KEYS.SCALE: temp['scale'],
            KEYS.TASK: temp['task'],

            KEYS.ID: ids,
            KEYS.MASS: mass,
            KEYS.BH_MASS: bh_mass,
        }

        return mrg


class Mergers_New(Mergers):
    """New-Seed Simulations (Blecha, Kelley, Torrey collaboration project).
    """

    class KEYS(Mergers.KEYS):
        SNAP = 'snap'
        TASK = 'task'

        BH_MASS = 'bh_mass'
        POS  = 'pos'
        VEL  = 'vel'

    def _parse_raw_file_line(self, line):
        KEYS = self.KEYS

        format = (
            "{time:g} {task:d}  " +
            "{id_o:d} {mass_o:g} " +
            "{px_o:g} {py_o:g} {pz_o:g}  " +
            "{id_i:d} {mass_i:g} " +
            "{px_i:g} {py_i:g} {pz_i:g}"
        )

        format_mass = (
            "{time:g} {task:d}  " +
            "{id_o:d} {mass_o:g} {bh_mass_o:g} " +
            "{px_o:g} {py_o:g} {pz_o:g}  " +
            "{id_i:d} {mass_i:g} {bh_mass_i:g} " +
            "{px_i:g} {py_i:g} {pz_i:g}"
        )

        format_mass_vel = (
            "{time:g} {task:d}  " +
            "{id_o:d} {mass_o:g} {bh_mass_o:g} " +
            "{px_o:g} {py_o:g} {pz_o:g} {vx_o:g} {vy_o:g} {vz_o:g}  " +
            "{id_i:d} {mass_i:g} {bh_mass_i:g} " +
            "{px_i:g} {py_i:g} {pz_i:g} {vx_i:g} {vy_i:g} {vz_i:g}"
        )

        for form in [format_mass_vel, format_mass, format]:
            temp = parse.parse(form, line)
            if temp is not None:
                break
        else:
            err = "ERROR: failed to parse line '{}'".format(line)
            logging.error(err)
            raise ValueError(err)

        ids = np.zeros(2, dtype=np.uint64)
        ids[BH_TYPE.OUT] = temp['id_o']
        ids[BH_TYPE.IN] = temp['id_i']

        mass = np.zeros(2)
        mass[BH_TYPE.OUT] = temp['mass_o']
        mass[BH_TYPE.IN] = temp['mass_i']

        pos = np.zeros((3, 2))
        pos[:, BH_TYPE.OUT] = [temp['px_o'], temp['py_o'], temp['pz_o']]
        pos[:, BH_TYPE.IN] = [temp['px_i'], temp['py_i'], temp['pz_i']]

        bh_mass = np.zeros(2)
        if 'bh_mass_o' in temp:
            bh_mass[BH_TYPE.OUT] = temp['bh_mass_o']
            bh_mass[BH_TYPE.IN] = temp['bh_mass_i']

        vel = np.zeros((3, 2))
        if 'vx_o' in temp:
            vel[:, BH_TYPE.OUT] = [temp['vx_o'], temp['vy_o'], temp['vz_o']]
            vel[:, BH_TYPE.IN] = [temp['vx_i'], temp['vy_i'], temp['vz_i']]

        rv = {
            KEYS.SCALE: temp['time'],
            KEYS.TASK: temp['task'],

            KEYS.ID: ids,
            KEYS.MASS: mass,
            KEYS.BH_MASS: bh_mass,
            KEYS.POS: pos,
            KEYS.VEL: vel
        }

        return rv

    def _get_merger_file_list(self, sim_path, cosmo=None):
        scales = None if cosmo is None else cosmo.scale
        num_snaps = None if scales is None else len(scales)

        mdir_pattern = 'mergers_' + '[0-9]' * 3
        mdir_pattern = os.path.join(sim_path, 'output', 'blackholes', mdir_pattern)
        merger_dirs = sorted(glob.glob(mdir_pattern))
        num_mdirs = len(merger_dirs)
        if self._verbose:
            print("Found {} merger directories".format(num_mdirs))
        if (num_snaps is not None) and (num_snaps != num_mdirs):
            err = "WARNING: found {} merger dirs but {} snapshots!".format(num_mdirs, num_snaps)
            logging.warning(err)
        if num_mdirs == 0:
            err = "No merger directories found matching {}!".format(mdir_pattern)
            logging.error(err)
            raise FileNotFoundError(err)

        file_list = []
        for kk, mdir in enumerate(merger_dirs):
            snap = int(os.path.basename(mdir).split('_')[-1])
            if snap != kk:
                msg = "WARNING: {}th merger-directory is snapshot {}!".format(kk, snap)
                logging.warning(msg)

            mfils = os.path.join(mdir, "blackhole_mergers_*.txt")
            mfils = sorted(glob.glob(mfils))
            if len(mfils) == 0:
                err = "ERROR: snap={} : no files found in '{}'".format(snap, mdir)
                raise FileNotFoundError(err)

            file_list.extend(mfils)

        if self._verbose:
            print("Found {} merger files".format(len(file_list)))

        return file_list

    '''
    def _load_raw_mergers(self):
        KEYS = self.KEYS
        sim_path = self._sim_path

        if sim_path is None:
            err = "ERROR: `sim_path` is not set, cannot load raw mergers!"
            logging.error(err)
            raise ValueError(err)

        if not os.path.isdir(sim_path):
            err = "ERROR: `sim_path` '{}' does not exist!".format(sim_path)
            logging.error(err)
            raise ValueError(err)

        try:
            cosmo = illpy_lib.illcosmo.Simulation_Cosmology(sim_path, verbose=self._verbose)
            scales = cosmo.scale
            num_snaps = len(scales)
        except FileNotFoundError as err:
            scales = None
            num_snaps = None
            msg = "WARNING: could not load cosmology (snapshots missing?): '{}'".format(str(err))
            logging.warning(msg)
            logging.warning("WARNING: Cannot compare merger times to snapshot times!")

        mergers = []
        msnaps = []

        mdir_pattern = 'mergers_' + '[0-9]' * 3
        # mdir_pattern = os.path.join(self._sim_path, 'output', 'blackholes', mdir_pattern)
        mdir_pattern = os.path.join(self._sim_path, 'blackholes', mdir_pattern)
        merger_dirs = sorted(glob.glob(mdir_pattern))
        num_mdirs = len(merger_dirs)
        if self._verbose:
            print("Found {} merger directories".format(num_mdirs))
        if (num_snaps is not None) and (num_snaps != num_mdirs):
            err = "WARNING: found {} merger dirs but {} snapshots!".format(num_mdirs, num_snaps)
            logging.warning(err)
        if num_mdirs == 0:
            err = "No merger directories found matching {}!".format(mdir_pattern)
            logging.error(err)
            raise FileNotFoundError(err)

        num_mrgs = 0
        # for snap in range(len(scale)):
        for kk, mdir in enumerate(merger_dirs):
            snap = int(os.path.basename(mdir).split('_')[-1])
            if snap != kk:
                msg = "WARNING: {}th merger-directory is snapshot {}!".format(kk, snap)
                logging.warning(msg)

            mfils = os.path.join(mdir, "blackhole_mergers_*.txt")
            mfils = sorted(glob.glob(mfils))
            if len(mfils) == 0:
                err = "ERROR: snap={} : no files found in '{}'".format(snap, mdir)
                raise FileNotFoundError(err)

            if scales is not None:
                lo = scales[snap-1] if snap > 0 else 0.0
                try:
                    hi = scales[snap]
                except IndexError:
                    # There may not be a final snapshot
                    if snap != len(scales):
                        raise
                    hi = 1.0

            for ii, mf in enumerate(mfils):
                prev = None
                for ll, line in enumerate(open(mf, 'r').readlines()):
                    try:
                        vals = self._parse_raw_file_line(line)
                    except:
                        logging.error("FAILED on line {} of file '{}'".format(ll, mf))
                        raise

                    vals[KEYS.SNAP] = snap

                    sc = vals[KEYS.SCALE]
                    if prev is None:
                        prev = sc
                    elif sc < prev:
                        err = f"Snap: {snap}, file: {ii} - scale:{sc:.8f} is before previous:{prev:.8f}!"
                        logging.error(err)
                        raise ValueError(err)

                    if (scales is not None):
                        lo_flag = (sc < lo) & (not np.isclose(sc, lo, rtol=1e-6, atol=0.0))
                        hi_flag = (sc > hi) & (not np.isclose(sc, hi, rtol=1e-6, atol=0.0))
                        if (lo_flag or hi_flag):
                            err = "Snap: {}, file: {}, scale:{:.8f} not in [{:.8f}, {:.8f}]!".format(
                                snap, ii, sc, lo, hi)
                            logging.error(err)
                            raise ValueError(err)

                    mergers.append(vals)
                    msnaps.append(snap)
                    num_mrgs += 1

        mrgs = {}
        for mm, temp in enumerate(mergers):
            if mm == 0:
                for kk, vv in temp.items():
                    if np.isscalar(vv):
                        shp = num_mrgs
                        tt = type(vv)
                    else:
                        shp = (num_mrgs,) + np.shape(vv)
                        tt = vv.dtype

                    mrgs[kk] = np.zeros(shp, dtype=tt)

            for kk, vv in temp.items():
                mrgs[kk][mm, ...] = vv

        idx = np.argsort(mrgs['scale'])
        for kk, vv in mrgs.items():
            mrgs[kk] = vv[idx, ...]

        if self._verbose:
            print("Loaded {} raw mergers".format(idx.size))

        return mrgs
    '''
