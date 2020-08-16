"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import logging

import numpy as np
import h5py

import parse

import zcode.inout as zio
import zcode.math as zmath

import illpy_lib  # noqa
from illpy_lib.illbh import BH_TYPE, Processed, utils


VERSION = 0.5


class Mergers_New(Processed):

    _PROCESSED_FILENAME = "bh-mergers.hdf5"

    class KEYS(Processed.KEYS):
        SCALE = 'scale'
        SNAP = 'snap'
        TASK = 'task'

        ID   = 'id'
        MASS = 'mass'
        BH_MASS = 'bh_mass'
        POS  = 'pos'
        VEL  = 'vel'

        NEXT = 'next'
        PREV = 'prev'

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
                hi = scales[snap]

            for ii, mf in enumerate(mfils):
                prev = None
                for line in open(mf, 'r').readlines():
                    vals = self._parse_raw_file_line(line)
                    vals[KEYS.SNAP] = snap

                    sc = vals[KEYS.SCALE]
                    if prev is None:
                        prev = sc
                    elif sc < prev:
                        err = f"Snap: {snap}, file: {ii} - scale:{sc:.8f} is before previous:{prev:.8f}!"
                        logging.error(err)
                        raise ValueError(err)

                    if (scales is not None) and ((sc < lo) or (sc > hi)):
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

        # Construct merger tree ('next' and 'prev' arrays) and do some consistency checks
        mrgs = self._build_tree(mrgs)
        if self._verbose:
            print("Merger tree constructed with {} next and {} prev links".format(
                np.count_nonzero(mrgs['next'] >= 0), np.count_nonzero(mrgs['prev'] >= 0)))

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
            msg = "{:6d} a={:.8f} s={:3d} ".format(mm, mrgs[KEYS.SCALE][mm], mrgs[KEYS.SNAP][mm])
            ids = mrgs[KEYS.ID][mm]
            mass = mrgs[KEYS.MASS][mm]
            bh_mass = mrgs[KEYS.MASS][mm]
            for name, tt in zip([' in', 'out'], [BH_TYPE.IN, BH_TYPE.OUT]):
                try:
                    msg += "{} id={:14d} bh_mass={:.8f} mass={:.8f} ".format(
                        name, ids[tt], bh_mass[tt], mass[tt])
                except:
                    print("ids  = ", ids[tt], type(ids[tt]))
                    print("mass = ", mass[tt], type(mass[tt]))
                    raise

            return msg

        num = mrgs['scale'].size
        next = np.ones(num, np.int32) * -1
        prev = np.ones((num, 2), np.int32) * -1
        use_bh_mass = True
        for mm in range(num):
            # mids = [mrgs['id1'][mm], mrgs['id2'][mm]]
            # ma = mrgs['scale'][mm]
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
                # nids = [mrgs['id1'][nn], mrgs['id2'][nn]]
                # na = mrgs['scale'][nn]
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

        if self._verbose:

            def count_prev(idx, cnt):
                if cnt >= next.size:
                    raise RuntimeError()

                temp = 0
                for bh in range(2):
                    if prev[idx, bh] >= 0:
                        # msg = " " * (cnt + temp)
                        temp += 1
                        # msg += "{} : {} ==> {} ({})".format(idx, bh, prev[idx, bh], temp + cnt)
                        # print(msg)
                        temp += count_prev(prev[idx, bh], cnt)

                # print(" " * (cnt + temp) + "{} = {}".format(idx, temp))
                return temp + cnt

            num_next = (next >= 0)
            print("Mergers with subsequent mergers: {}".format(zmath.frac_str(num_next)))
            print("Mergers that are final remnants: {}".format(zmath.frac_str(~num_next)))
            mult = -1 * np.ones_like(next)
            for mm in range(num):
                # Only consider final mergers
                if next[mm] >= 0:
                    continue
                mult[mm] = count_prev(mm, 0)

            mult = mult[mult >= 0]
            print("Remnant merger-multiplicity:")
            print("\tzero = {}".format(zmath.frac_str(mult == 0)))
            print("\tave  = {:.2f}".format(mult.mean()))
            print("\tmed  = {:.2f}".format(np.median(mult)))
            print("\t{}".format(zmath.stats_str(mult)))

        return mrgs
