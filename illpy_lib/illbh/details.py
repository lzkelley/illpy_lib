"""
"""

# from collections import OrderedDict
import os
import sys
import glob
import logging
from datetime import datetime

import numpy as np
import h5py
from mpi4py import MPI

import zcode.inout as zio
import zcode.math as zmath

import illpy_lib  # noqa
from illpy_lib.illbh import Processed, utils, PATH_PROCESSED, mergers, BH_TYPE

VERBOSE = True
VERSION = 0.4                                    # Version of details
DETS_RESOLUTION_TARGET = 1.0e-3                # in units of scale-factor
DETS_RESOLUTION_TOLERANCE = 0.5              # allowed fraction below `DETS_RESOLUTION_TARGET`
DETS_RESOLUTION_MIN_NUM = 10

comm = MPI.COMM_WORLD

# Start

if DETS_RESOLUTION_MIN_NUM < 10:
    raise ValueError("ERROR: `DETS_RESOLUTION_MIN_NUM` must be >= 10!")


class Details_New(Processed):

    _PROCESSED_FILENAME = "bh-details.hdf5"

    class KEYS(Processed.KEYS):
        SCALE  = 'scale'
        ID     = 'id'
        POT    = 'potential'
        MASS   = 'mass'
        MDOT   = 'mdot'
        MDOT_B = 'mdot_bondi'
        POS    = 'pos'
        VEL    = 'vel'
        TASK   = 'task'
        # RHO   = 'rho'
        # CS    = 'cs'

        U_IDS = 'unique_ids'
        U_INDICES = 'unique_indices'
        U_COUNTS = 'unique_counts'

        _INTERP_KEYS = [POT, MASS, MDOT, MDOT_B, POS, VEL]

    def _process(self):
        if self._sim_path is None:
            err = "ERROR: cannot process {} without `sim_path` set!".format(self.__class__)
            logging.error(err)
            raise ValueError(err)

        snap = 0
        KEYS = self.KEYS
        verb = self._verbose
        first = True
        tot_input = 0
        tot_output = 0
        tot_count = 0
        ave_ratio = 0.0

        dets = {}
        while True:
            if verb:
                print("snap = {}".format(snap))

            try:
                dets_snap = Details_Snap_New(snap, sim_path=self._sim_path)
            except FileNotFoundError as err:
                if snap < 2:
                    msg = "ERROR: snap {} details missing! Has `details.py` been run?".format(snap)
                    logging.error(msg)
                    logging.error(str(err))
                    raise err
                else:
                    break

            scales = dets_snap[KEYS.SCALE]
            ids = dets_snap[KEYS.ID]
            tot_num = scales.size
            if verb:
                print("\ttot_num = {}".format(tot_num))

            if tot_num == 0:
                snap += 1
                continue

            u_ids = dets_snap[KEYS.U_IDS]
            u_indices = dets_snap[KEYS.U_INDICES]
            u_counts = dets_snap[KEYS.U_COUNTS]
            for ii, xx, nn in zip(u_ids, u_indices, u_counts):
                # if verb:
                #     print("\t{} - {} - {}:{} - {}:{}".format(
                #         ii, nn, xx, xx+nn-1, scales[xx], scales[xx+nn-1]))
                #     # print("\t{}".format(scales[xx:xx+nn]))

                if not np.all(ids[xx] == ids[xx:xx+nn]):
                    err = "ERROR: snap {}, ids inconsistent for BH {}!".format(snap, ii)
                    logging.error(err)
                    raise ValueError(err)
                if (xx > 0) and (ids[xx] == ids[xx-1]):
                    err = "ERROR: snap {}, ids start before expected for BH {}!".format(snap, ii)
                    logging.error(err)
                    raise ValueError(err)
                if (xx+nn < tot_num) and (ids[xx] == ids[xx+nn]):
                    err = "ERROR: snap {}, ids continue beyond expected for BH {}!".format(snap, ii)
                    logging.error(err)
                    raise ValueError(err)
                if np.any(np.diff(scales[xx:xx+nn]) < 0.0):
                    err = "ERROR: snap {}, BH {} scales not monotonically increasing!".format(
                        snap, ii)
                    logging.error(err)
                    raise ValueError(err)

                if nn > DETS_RESOLUTION_MIN_NUM:
                    span = scales[xx+nn-1] - scales[xx]
                    res = span / nn
                else:
                    res = np.inf
                    span = None

                # if verb:
                #     print("\tres = {} (target={}), span = {}".format(
                #         res, DETS_RESOLUTION_TARGET, span))

                if res < DETS_RESOLUTION_TARGET / (1.0 + DETS_RESOLUTION_TOLERANCE):
                    new_num = int(np.ceil(span / DETS_RESOLUTION_TARGET))
                    if verb and (new_num < DETS_RESOLUTION_MIN_NUM):
                        # msg = "WARNING: target resolution num = {} is too low".format(new_num)
                        # logging.warning(msg)
                        # raise RuntimeError(msg)
                        new_num = np.max([nn/10, DETS_RESOLUTION_MIN_NUM])
                        new_num = int(np.ceil(new_num))

                    new_scales = np.linspace(scales[xx], scales[xx+nn-1], new_num)[1:-1]
                    # if verb:
                    #     print("\tinterpolating {} ==> {}".format(nn, new_num))

                    for kk in KEYS:
                        if kk.startswith('unique'):
                            continue

                        if (kk in KEYS._INTERP_KEYS):
                            xvals = scales[xx:xx+nn]
                            yvals = dets_snap[kk][xx:xx+nn]
                            ndim = np.ndim(yvals)
                            if ndim == 1:
                                temp = [
                                    [yvals[0]],
                                    np.interp(new_scales, xvals, yvals),
                                    [yvals[-1]],
                                ]
                                temp = np.concatenate(temp)
                            elif ndim == 2:
                                nvars = yvals.shape[1]
                                temp = np.zeros((new_num, nvars))
                                for jj in range(nvars):
                                    temp[0, jj] = yvals[0, jj]
                                    temp[-1, jj] = yvals[-1, jj]
                                    temp[1:-1, jj] = np.interp(new_scales, xvals, yvals[:, jj])
                            else:
                                err = "ERROR: unexpected shape for {} ({})!".format(kk, yvals.shape)
                                logging.error(err)
                                raise ValueError(err)

                        elif kk == KEYS.SCALE:
                            temp = np.concatenate([[scales[xx]], new_scales, [scales[xx+nn-1]]])
                        else:
                            temp = dets_snap[kk][xx:xx+new_num]

                        if first:
                            dets[kk] = temp
                        else:
                            prev = dets[kk]
                            dets[kk] = np.concatenate([prev, temp])

                    first = False
                    tot_output += new_num
                    ave_ratio += (new_num / nn)

                else:
                    for kk in KEYS:
                        temp = dets_snap[kk][xx:xx+nn]
                        if first:
                            dets[kk] = temp
                        else:
                            prev = dets[kk]
                            dets[kk] = np.concatenate([prev, temp])

                    first = False
                    tot_output += nn
                    ave_ratio += 1.0

                tot_input += nn
                tot_count += 1

            snap += 1

        ave_ratio /= tot_count
        tot_ratio = tot_output / tot_input
        if verb:
            print("Downsampled from {:.2e} ==> {:.2e}".format(tot_input, tot_output))
            print("Total compression: {:.2e}, average: {:.2e}".format(tot_ratio, ave_ratio))

        self._finalize_and_save(dets)

        return

    def _finalize_and_save(self, dets):
        KEYS = self.KEYS
        idx = np.lexsort((dets[KEYS.SCALE], dets[KEYS.ID]))
        for kk, vv in dets.items():
            if kk.startswith('unique'):
                continue
            dets[kk] = vv[idx, ...]

        u_ids, u_inds, u_counts = np.unique(dets[KEYS.ID], return_index=True, return_counts=True)
        num_unique = u_ids.size
        dets[KEYS.U_IDS] = u_ids
        dets[KEYS.U_INDICES] = u_inds
        dets[KEYS.U_COUNTS] = u_counts

        # -- Save values to file
        # Get output filename for this snapshot
        #   path should have already been created in `_process` by rank=0
        fname = self.filename
        with h5py.File(fname, 'w') as save:
            utils._save_meta_to_hdf5(save, self._sim_path, VERSION, __file__)
            for key in KEYS:
                save.create_dataset(str(key), data=dets[key])

        if self._verbose:
            msg = "Saved data for {} details ({} unique) to '{}' size {}".format(
                len(dets[KEYS.SCALE]), num_unique, fname, zio.get_file_size(fname))
            print(msg)

        return

    def _load_from_save(self):
        fname = self.filename
        with h5py.File(fname, 'r') as load:
            vers = load.attrs['version']
            if vers != str(VERSION):
                msg = "WARNING: loaded version '{}' does not match current '{}'!".format(
                    vers, VERSION)
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
            size = load[self.KEYS.SCALE].size
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
                print("Loaded {:10d} details from '{}', created '{}'".format(size, fname, dt))

        self._size = size
        return


class Details_Snap_New(Details_New):

    _PROCESSED_FILENAME = "bh-details_{snap:03d}.hdf5"

    def __init__(self, snap, *args, **kwargs):
        self._snap = snap
        super().__init__(*args, **kwargs)
        return

    def _process(self):
        snap = self._snap
        types = [np.float, np.uint64, np.float64, np.float64, np.float64, np.float64,
                 np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]

        def parse_detail_line(line):
            """
            "%g %llu  %g  %g %g %g  %g %g %g  %g %g %g\n",
            All.Time, (long long) P[n].ID, pot, BPP(n).BH_Mass, BPP(n).BH_MdotBondi, BPP(n).BH_Mdot,
            P[n].Pos[0], P[n].Pos[1], P[n].Pos[2], P[n].Vel[0], P[n].Vel[1], P[n].Vel[2])
            """
            line = line.strip().split()
            if len(line) != len(types):
                raise ValueError(f"unexpected line length: '{line}'!")

            temp = [tt(ll) for tt, ll in zip(types, line)]
            return temp

        ddir = 'details_{:03d}'.format(snap)
        ddir = os.path.join(self._sim_path, 'output', 'blackholes', ddir)

        if not os.path.isdir(ddir):
            err = "ERROR: `ddir` '{}' does not exist!".format(ddir)
            logging.error(err)
            raise FileNotFoundError(err)

        dfils = os.path.join(ddir, "blackhole_details_*.txt")
        dfils = sorted(glob.glob(dfils))

        num_files = len(dfils)
        if self._verbose:
            print(f"snap {snap}, dir {ddir}, files: {num_files}")
        if len(dfils) == 0:
            err = "ERROR: snap={} : no files found in '{}'".format(snap, ddir)
            raise FileNotFoundError(err)

        details = []
        tasks = []
        num_lines = 0
        for ii, dfil in enumerate(dfils):
            _task = os.path.basename(dfil).split('_')[-1].split('.')[0]
            _task = int(_task)

            prev = None
            for ll, line in enumerate(open(dfil, 'r').readlines()):
                try:
                    vals = parse_detail_line(line)
                except Exception as err:
                    msg = "ERROR: failed to parse line {} from {}: '{}'".format(ll, dfil, line)
                    logging.error(msg)
                    logging.error(str(err))
                    raise

                sc = vals[0]
                if prev is None:
                    prev = sc
                elif sc < prev:
                    err = f"Snap: {snap}, file: {ii} - scale:{sc:.8f} is before prev:{prev:.8f}!"
                    logging.error(err)
                    raise ValueError(err)

                details.append(vals)
                tasks.append(ii)
                num_lines += 1

        KEYS = self.KEYS
        dets = {
            KEYS.SCALE  : np.zeros(num_lines, dtype=types[0]),
            KEYS.ID     : np.zeros(num_lines, dtype=types[1]),
            KEYS.POT    : np.zeros(num_lines, dtype=types[2]),
            KEYS.MASS   : np.zeros(num_lines, dtype=types[3]),
            KEYS.MDOT_B : np.zeros(num_lines, dtype=types[4]),
            KEYS.MDOT   : np.zeros(num_lines, dtype=types[5]),
            KEYS.POS    : np.zeros((num_lines, 3), dtype=types[6]),
            KEYS.VEL    : np.zeros((num_lines, 3), dtype=types[9]),

            KEYS.TASK   : np.zeros(num_lines, dtype=np.uint32),
        }
        for ii in range(num_lines):
            dd = details[ii]

            dets[KEYS.SCALE][ii] = dd[0]
            dets[KEYS.ID][ii] = dd[1]
            dets[KEYS.POT][ii] = dd[2]
            dets[KEYS.MASS][ii] = dd[3]
            dets[KEYS.MDOT_B][ii] = dd[4]
            dets[KEYS.MDOT][ii] = dd[5]
            dets[KEYS.POS][ii] = [dd[6+kk] for kk in range(3)]
            dets[KEYS.VEL][ii] = [dd[9+kk] for kk in range(3)]
            dets[KEYS.TASK][ii] = tasks[ii]

        self._finalize_and_save(dets)
        return

    @property
    def filename(self):
        if self._filename is None:
            # `sim_path` has already been checked to exist in initializer
            sim_path = self._sim_path
            temp = self._PROCESSED_FILENAME.format(snap=self._snap)
            self._filename = os.path.join(sim_path, *PATH_PROCESSED, temp)

        return self._filename


class Merger_Details_New(Details_New):

    _PROCESSED_FILENAME = "bh-merger-details.hdf5"

    def _process(self):
        if self._sim_path is None:
            err = "ERROR: cannot process {} without `sim_path` set!".format(self.__class__)
            logging.error(err)
            raise ValueError(err)

        mrgs = mergers.Mergers_New(sim_path=self._sim_path, verbose=self._verbose)
        mkeys = mrgs.KEYS
        num_mrgs = mrgs.size
        max_snap = np.max(mrgs[mkeys.SNAP])

        snap = 0
        KEYS = self.KEYS
        verb = self._verbose
        mdets = [[None, None, None] for ii in range(num_mrgs)]
        while True:
            if verb:
                print("snap = {}".format(snap))

            try:
                dets_snap = Details_Snap_New(snap, sim_path=self._sim_path)
            except FileNotFoundError as err:
                if snap < max_snap:
                    msg = "ERROR: snap {} details missing! Has `details.py` been run?".format(snap)
                    logging.error(msg)
                    logging.error(str(err))
                    raise err
                else:
                    break

            scales = dets_snap[KEYS.SCALE]
            tot_num = scales.size
            if verb:
                print("\ttot_num = {}".format(tot_num))

            if tot_num == 0:
                snap += 1
                continue

            u_ids = dets_snap[KEYS.U_IDS]
            u_indices = dets_snap[KEYS.U_INDICES]
            u_counts = dets_snap[KEYS.U_COUNTS]
            numu = u_ids.size
            for mm in range(num_mrgs):
                merger_ids = mrgs[mkeys.ID][mm]
                merger_scale = mrgs[mkeys.SCALE][mm]
                merger_mass = mrgs[mkeys.MASS][mm]
                next_merger = mrgs[mkeys.NEXT][mm]
                if next_merger > 0:
                    next_merger = mrgs[mkeys.SCALE][next_merger]
                else:
                    next_merger = np.inf

                for bh, bhid in enumerate(merger_ids):
                    dd = np.searchsorted(u_ids, bhid)
                    if (dd >= numu) or (u_ids[dd] != bhid):
                        continue

                    err = "ERROR: snap {}, merger {}, bh {} : ".format(snap, mm, bhid)
                    beg = u_indices[dd]
                    end = beg + u_counts[dd]
                    det_scales = dets_snap[KEYS.SCALE][beg:end]
                    det_masses = dets_snap[KEYS.MASS][beg:end]
                    det_ids = dets_snap[KEYS.ID][beg:end]
                    if not np.all(det_ids == bhid):
                        raise RuntimeError()

                    if bh == BH_TYPE.IN:
                        if np.any(det_scales > merger_scale):
                            err += "in-bh found after merger!"
                            logging.error(err)
                            raise RuntimeError(err)

                        first = (mdets[mm][bh] is None)
                        if first:
                            mdets[mm][bh] = dict()

                        for kk in KEYS:
                            if kk.startswith('unique'):
                                continue
                            temp = dets_snap[kk][beg:end]
                            if first:
                                mdets[mm][bh][kk] = temp
                            else:
                                prev = mdets[mm][bh][kk]
                                mdets[mm][bh][kk] = np.concatenate([prev, temp])

                            if (kk == KEYS.MASS) and np.any(temp > merger_mass[BH_TYPE.IN]):
                                err += "in bh larger than merger mass!"
                                logging.error(err)
                                raise RuntimeError(err)

                    elif bh == BH_TYPE.OUT:

                        idx_bef = (det_scales < merger_scale)
                        idx_aft = (det_scales > merger_scale) & (det_scales < next_merger)
                        for loc, idx in zip([BH_TYPE.OUT, 2], [idx_bef, idx_aft]):
                            if not np.any(idx):
                                continue

                            print("merger scale = {:.8f}".format(merger_scale))
                            print("merger mass  = {:.8e}, {:.8e}".format(*merger_mass))
                            print("SCALES {} : {}".format(
                                loc, zmath.stats_str(det_scales[idx], format=':.8f')))
                            print("MASSES {} : {}".format(
                                loc, zmath.stats_str(det_masses[idx], format=':.8e')))

                            first = (mdets[mm][loc] is None)
                            if first:
                                mdets[mm][loc] = dict()

                            for kk in KEYS:
                                if kk.startswith('unique'):
                                    continue
                                temp = dets_snap[kk][beg:end][idx]
                                if first:
                                    mdets[mm][loc][kk] = temp
                                else:
                                    prev = mdets[mm][loc][kk]
                                    mdets[mm][loc][kk] = np.concatenate([prev, temp])

                                if kk == KEYS.MASS:
                                    # After the merger, details should be MORE massive
                                    if (loc == 2):
                                        if np.any(temp < merger_mass.sum()):
                                            msg = "merger_mass = {:.6e}, {:.6e}".format(
                                                *merger_mass)
                                            logging.error(msg)
                                            msg = "details masses = {}".format(
                                                zmath.stats_str(temp, format=':.8e'))
                                            logging.error(msg)
                                            msg = "merger scale = {:.8f}".format(merger_scale)
                                            logging.error(msg)
                                            msg = "details scales = {}".format(
                                                zmath.stats_str(det_scales[idx], format=':.8f'))
                                            logging.error(msg)
                                            err += "remnant mass less than merger mass!"
                                            logging.error(err)
                                            # raise RuntimeError(err)
                                    # Before the merger, details should be LESS massive
                                    else:
                                        if np.any(temp > merger_mass[BH_TYPE.OUT]):
                                            err += "out bh larger than merger mass!"
                                            logging.error(err)
                                            raise RuntimeError(err)

                                elif kk == KEYS.ID:
                                    if not np.all(temp == bhid):
                                        err += "out bh IDs dont match!"
                                        logging.error(err)
                                        raise RuntimeError(err)

                            # for kk
                        # for loc
                    # elif bh
                # for bh

            snap += 1
            # while true

        if verb:

            cosmo = illpy_lib.illcosmo.Simulation_Cosmology(self._sim_path, verbose=False)
            seed_mass = float(cosmo._params['SeedBlackHoleMass'])

            print("\n\n\n")
            counts = np.zeros((num_mrgs, 3), dtype=int)
            for mm in range(num_mrgs):
                for jj in range(3):
                    if mdets[mm][jj] is not None:
                        counts[mm, jj] = mdets[mm][jj][KEYS.SCALE].size
                        continue

                    okay = False
                    print("Merger {} type {} with zero matches".format(
                        mm, BH_TYPE.from_value(jj)))
                    print(mrgs.scale[mm], mrgs.id[mm], mrgs.mass[mm])
                    if jj == BH_TYPE.REMNANT:
                        nn = mrgs.next[mm]
                        print("next = ", nn)
                        if nn > 0:
                            print("\t", mrgs.scale[nn], mrgs.id[nn], mrgs.mass[nn])

                        tnext = mrgs.scale[nn] - mrgs.scale[mm]
                        if (tnext < 1e-4):
                            print("\t>>Remnant merged almost immediately")
                            okay = True
                        # else:
                        #     print("\tWARNING: time till next merger = {:.8f}".format(tnext))

                    else:
                        pp = mrgs.prev[mm, jj]
                        print("prev = ", pp)
                        if pp >= 0:
                            print("\t", mrgs.scale[pp], mrgs.id[pp])

                            tprev = mrgs.scale[mm] - mrgs.scale[pp]
                            if (tprev < 1e-4):
                                print("\t>>Previous remnant merged almost immediately")
                                okay = True
                            # else:
                            #     print("\tWARNING: time till next merger = {:.8f}".format(tnext))
                        elif np.isclose(mrgs.mass[mm, jj], seed_mass, rtol=0.1):
                            print("\t>>BH is near seed mass")
                            okay = True

                    if not okay:
                        print("\tWARNING: no explanation for lack of details entries!")

                print("")

            num_zero = np.count_nonzero(counts == 0, axis=0)
            med_nums = np.median(counts, axis=0)
            ave_nums = np.average(counts, axis=0)

            names = ['zero', 'med', 'ave']
            types = ['', '', '']
            types[BH_TYPE.IN] = 'in '
            types[BH_TYPE.OUT] = 'out'
            types[BH_TYPE.REMNANT] = 'rem'

            table = []
            for vals in [num_zero, med_nums, ave_nums]:
                row = []
                for jj in range(3):
                    temp = "{:g}".format(vals[jj])
                    row.append(temp)
                table.append(row)

            print(zio.ascii_table(table, rows=names, cols=types))

        self._finalize_and_save(mdets)
        return

    def _finalize_and_save(self, mdets):
        KEYS = self.KEYS
        num = len(mdets)
        # for mm in range(num):



        idx = np.lexsort((dets[KEYS.SCALE], dets[KEYS.ID]))
        for kk, vv in dets.items():
            if kk.startswith('unique'):
                continue
            dets[kk] = vv[idx, ...]

        u_ids, u_inds, u_counts = np.unique(dets[KEYS.ID], return_index=True, return_counts=True)
        num_unique = u_ids.size
        dets[KEYS.U_IDS] = u_ids
        dets[KEYS.U_INDICES] = u_inds
        dets[KEYS.U_COUNTS] = u_counts

        # -- Save values to file
        # Get output filename for this snapshot
        #   path should have already been created in `_process` by rank=0
        fname = self.filename
        with h5py.File(fname, 'w') as save:
            utils._save_meta_to_hdf5(save, self._sim_path, VERSION, __file__)
            for key in KEYS:
                save.create_dataset(str(key), data=dets[key])

        if self._verbose:
            msg = "Saved data for {} details ({} unique) to '{}' size {}".format(
                len(dets[KEYS.SCALE]), num_unique, fname, zio.get_file_size(fname))
            print(msg)

        return

def process_details_snaps(sim_path, recreate=False, verbose=VERBOSE):

    if comm.rank == 0:
        temp = 'details_' + '[0-9]' * 3
        path_bhs = os.path.join(sim_path, 'output', 'blackholes')
        temp = os.path.join(path_bhs, temp)
        det_dirs = sorted(glob.glob(temp))
        if verbose:
            print("Found {} details directories".format(len(det_dirs)))
        if len(det_dirs) == 0:
            err = "ERROR: no details directories found (pattern: '{}')".format(temp)
            logging.error(err)
            raise FileNotFoundError(err)

        snap_list = []
        num_procs = None
        prev = None
        for dd in det_dirs:
            sn = int(os.path.basename(dd).split('_')[-1])
            snap_list.append(sn)
            if prev is None:
                if sn != 0:
                    err = "WARNING: first snapshot ({}) is not zero!".format(sn)
                    logging.warning(err)
            elif prev + 1 != sn:
                err = "WARNING: snapshot {} does not follow previous {}!".format(sn, prev)
                logging.warning(err)

            pattern = os.path.join(dd, "blackhole_details_*.txt")
            num = len(glob.glob(pattern))
            if num == 0:
                err = "ERROR: found no files matching details pattern ({})!".format(pattern)
                logging.error(err)
                raise FileNotFoundError(err)
            else:
                if num_procs is None:
                    num_procs = num
                    if verbose:
                        print("Files found for {} processors".format(num))
                elif num_procs != num:
                    err = "WARNING: num of files ({}) in snap {} does not match previous!".format(
                        num, sn)
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
    for snap in snap_list:
        details = Details_Snap_New(snap, sim_path=sim_path, recreate=recreate, verbose=verbose)
        num_lines += details[details.KEYS.SCALE].size

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


if __name__ == "__main__":
    beg = datetime.now()
    recreate = ('-r' in sys.argv) or ('--recreate' in sys.argv)
    verbose = ('-v' in sys.argv) or ('--verbose' in sys.argv) or VERBOSE

    if comm.rank == 0:
        this_fname = os.path.abspath(__file__)
        head = f"{this_fname} : {str(beg)} - rank: {comm.rank}/{comm.size}"
        print("\n" + head + "\n" + "=" * len(head) + "\n")

        if (len(sys.argv) < 2) or ('-h' in sys.argv) or ('--help' in sys.argv):
            logging.warning("USAGE: `python {} <PATH>`\n\n".format(__file__))
            sys.exit(0)

        sim_path = os.path.abspath(sys.argv[1]).rstrip('/')
        if os.path.basename(sim_path) == 'output':
            sim_path = os.path.split(sim_path)[0]

    else:
        sim_path = None

    sim_path = comm.bcast(sim_path, root=0)

    comm.barrier()

    process_details_snaps(sim_path, recreate=recreate, verbose=verbose)
