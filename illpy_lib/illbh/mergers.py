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

# import zcode.inout as zio
# import zcode.math as zmath

# import illpy_lib  # noqa
from illpy_lib.illbh import BH_TYPE, Processed, utils, KEYS


class Mergers(Processed):

    _PROCESSED_FILENAME = "bh-mergers.hdf5"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fname = self.filename()
        self._load(fname, self._recreate)
        return

    def _parse_raw_file_line(self, line):
        raise NotImplementedError("This must be subclassed!")
        rv = None
        return rv

    def _get_merger_file_list(self, sim_path):
        raise NotImplementedError()

    def _clean_raw_mergers(self, mrgs):
        return mrgs

    def _load_raw_mergers(self):
        sim_path = self._sim_path
        verbose = self._verbose

        # Get list of illustris blackhole-merger text files
        #    This should be subclasses to find the files for particular output structures
        merger_files = self._get_merger_file_list(sim_path)
        if verbose:
            print("Found {} merger files".format(len(merger_files)))

        # --- Go through all files compiling a list of dict for each merger
        mergers = []
        print_first = verbose   # print the first merger line if verbose
        for ii, mf in enumerate(tqdm.tqdm(merger_files, desc='files')):
            file_mergers = _load_raw_mergers_from_file(mf, len(mergers) == 0)
            mergers = mergers + file_mergers

        num_mrgs = len(mergers)
        if verbose:
            print("Loaded {} raw mergers".format(num_mrgs))

        # --- Convert to dict of arrays for all mergers ---
        mrgs = {}
        for mm, temp in enumerate(mergers):
            # Initialize arrays on first value
            if mm == 0:
                # Go through each item to determine necessary shape and data-type
                for kk, vv in temp.items():
                    # Scalars --> (N,)
                    if np.isscalar(vv):
                        shp = num_mrgs
                        tt = type(vv)
                    # Arrays --> (N,) + (shape,)
                    else:
                        shp = (num_mrgs,) + np.shape(vv)
                        tt = vv.dtype

                    mrgs[kk] = np.zeros(shp, dtype=tt)

            for kk, vv in temp.items():
                mrgs[kk][mm, ...] = vv

        # --- Sort by scale-factor ---
        #    Sort here so that ordering can be used for cleaning later
        #    NOTE: these should already be all sorted... but lets be safe
        idx = np.argsort(mrgs[KEYS.SCALE])
        for kk, vv in mrgs.items():
            mrgs[kk] = vv[idx, ...]

        return mrgs

    def _load_raw_mergers_from_file(self, fname, print_first):
        prev = None
        mergers = []
        for ll, line in enumerate(open(fname, 'r').readlines()):
            if print_first:
                print("first merger line: '{}'".format(line.strip()))
                print_first = False

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

        return mergers

    def _process(self):
        # Check output filename
        fname = self.filename()
        fname = utils._check_output_path(fname)

        # Load merger events directly from Illustris blackhole_mergers txt files, unmodified
        mrgs = self._load_raw_mergers()
        if self._verbose:
            print("Loaded ", mrgs[KEYS.SCALE].size, "mergers")

        # Process mergers
        #    This should be subclasses for simulation-specific operations
        mrgs = self._clean_raw_mergers(mrgs)
        self._finalize_and_save(fname, mrgs)
        return

    def _finalize(self, mrgs):
        # -- Identify Unique BHs based on ID numbers
        mids = mrgs[KEYS.ID]
        # Convert from (M, 2) ==> (2*M,), i.e. [ID0_pri, ID0_sec, ID1_pri, ID1_sec ...]
        mids = mids.flatten()

        # Find unique indices, and first occurence of each index
        u_ids, u_inds, u_counts = np.unique(mids, return_index=True, return_counts=True)
        # Convert from index number in array of length (2*M,) to index number in mergers (M,)
        u_inds = u_inds // 2
        mrgs[KEYS.U_IDS] = u_ids
        mrgs[KEYS.U_INDICES] = u_inds
        mrgs[KEYS.U_COUNTS] = u_counts
        if self._verbose:
            print("Identified {} unique BHs in {} mergers".format(u_ids.size, mids.size))

        return mrgs

    '''
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
    '''


class Mergers_TNG(Mergers):

    def _get_merger_file_list(self, sim_path):
        mfil_pattern = 'blackhole_mergers_*.txt'
        mfil_pattern = os.path.join(sim_path, 'output', 'blackhole_mergers', mfil_pattern)
        merger_files = sorted(glob.glob(mfil_pattern))
        return merger_files

    def _parse_raw_file_line(self, line):
        """
        TOS & TNG:

        fprintf(FdBlackHolesMergers, "%d %g %llu %g %llu %g\n",
                ThisTask, All.Time, (long long) id, mass, (long long) P[no].ID, BPP(no).BH_Mass);
        """

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


class Mergers_TOS(Mergers_TNG):

    def _get_merger_file_list(self, sim_path):
        files = []
        path = os.path.join(sim_path, 'txt-files', 'txtfiles_new', '')
        for dirpath, dirnames, filenames in os.walk(path):
            filenames = [os.path.join(dirpath, fn) for fn in filenames if fn.startswith('blackhole_mergers')]
            if (len(filenames) > 0) and self._verbose:
                print("Found {} files in {}".format(len(filenames), dirpath))
            files = files + filenames

        return files


class Mergers_New(Mergers):
    """New-Seed Simulations (Blecha, Kelley, Torrey collaboration project).
    """

    def _parse_raw_file_line(self, line):
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
