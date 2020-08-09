"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
# import enum
# import sys
# import shutil
import glob
import logging
from datetime import datetime

import numpy as np
import h5py

import zcode.inout as zio
import zcode.math as zmath

import illpy_lib  # noqa
from illpy_lib.illbh import BH_TYPE, Processed, _ENUM
# import illpy_lib.illcosmo


VERSION = 0.5


class Mergers_New(Processed):

    _PROCESSED_FILENAME = "bh-mergers.hdf5"

    class KEYS(_ENUM):
        SCALE = 'scale'
        SNAP = 'snap'
        TASK = 'task'

        ID = 'id'
        MASS = 'mass'
        POS = 'pos'

    def _load_raw_mergers(self):

        sim_path = self._sim_path

        if sim_path is None:
            err = "ERROR: `sim_path` is not set, cannot load raw mergers!"
            logging.error(err)
            raise ValueError(err)

        if not os.path.isdir(sim_path):
            err = "ERROR: `sim_path` '{}' does not exist!".format(sim_path)
            logging.error(err)
            raise ValueError(err)

        def parse_merger_line(line):
            """
            "%g %d  %llu %g %g %g %g  %llu %g %g %g %g\n",
                All.Time, ThisTask,
                /* remnant BH */
                (long long) id, mass, pos[0], pos[1], pos[2],
                /* consumed BH */
                (long long) P[no].ID, BPP(no).BH_Mass, P[no].Pos[0], P[no].Pos[1], P[no].Pos[2]);
            """
            types = [np.float, np.uint,
                     np.uint64, np.float64, np.float64, np.float64, np.float64,
                     np.uint64, np.float64, np.float64, np.float64, np.float64]
            line = line.strip().split()
            if len(line) != len(types):
                raise ValueError(f"unexpected line length: '{line}'!")

            temp = [tt(ll) for tt, ll in zip(types, line)]
            return temp, types

        cosmo = illpy_lib.illcosmo.Simulation_Cosmology(sim_path, verbose=self._verbose)
        scale = cosmo.scale
        mergers = []
        msnaps = []

        num_mrgs = 0
        for snap in range(len(scale)):
            mdir = 'mergers_{:03d}'.format(snap)
            mdir = os.path.join(self._sim_path, 'output', 'blackholes', mdir)
            mfils = os.path.join(mdir, "blackhole_mergers_*.txt")
            mfils = sorted(glob.glob(mfils))
            if len(mfils) == 0:
                err = "ERROR: snap={} : no files found in '{}'".format(snap, mdir)
                raise FileNotFoundError(err)

            lo = scale[snap-1] if snap > 0 else 0.0
            hi = scale[snap]

            for ii, mf in enumerate(mfils):
                prev = None
                for line in open(mf, 'r').readlines():
                    vals, types = parse_merger_line(line)
                    sc = vals[0]
                    if prev is None:
                        prev = sc
                    elif sc < prev:
                        err = f"Snap: {snap}, file: {ii} - scale:{sc:.8f} is before previous:{prev:.8f}!"
                        logging.error(err)
                        raise ValueError(err)

                    if (sc < lo) or (sc > hi):
                        # print(num_mrgs)
                        err = f"Snap: {snap}, file: {ii} - scale:{sc:.8f} is not between [{lo:.8f}, {hi:.8f}]!"
                        logging.error(err)
                        # raise ValueError(err)
                    mergers.append(vals)
                    msnaps.append(snap)
                    num_mrgs += 1

        mrgs = {
            'scale': np.zeros(num_mrgs, dtype=types[0]),
            'snap': np.zeros(num_mrgs, dtype=np.uint32),
            'task': np.zeros(num_mrgs, dtype=types[1]),
            'id1': np.zeros(num_mrgs, dtype=types[2]),
            'mass1': np.zeros(num_mrgs, dtype=types[3]),
            'pos1': np.zeros((num_mrgs, 3), dtype=types[3]),
            'id2': np.zeros(num_mrgs, dtype=types[2]),
            'mass2': np.zeros(num_mrgs, dtype=types[3]),
            'pos2': np.zeros((num_mrgs, 3), dtype=types[3]),
        }
        for ii in range(num_mrgs):
            mm = mergers[ii]
            mrgs['scale'][ii] = mm[0]
            mrgs['task'][ii] = mm[1]
            mrgs['snap'][ii] = msnaps[ii]

            for kk in range(2):
                jj = 2 + kk*5
                mrgs[f'id{kk+1}'][ii] = mm[jj]
                mrgs[f'mass{kk+1}'][ii] = mm[jj+1]
                mrgs[f'pos{kk+1}'][ii, :] = [mm[jj+2+kk] for kk in range(3)]

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
        KEYS = self.KEYS
        with h5py.File(fname, 'w') as save:
            save.attrs['created'] = str(datetime.now())
            save.attrs['version'] = str(VERSION)
            save.attrs['script'] = str(os.path.abspath(__file__))
            save.attrs['sim_path'] = str(self._sim_path)

            ids = np.array([mrgs['id1'], mrgs['id2']]).T
            mass = np.array([mrgs['mass1'], mrgs['mass2']]).T
            pos = np.moveaxis(np.array([mrgs['pos1'], mrgs['pos2']]), 0, 1)

            save.create_dataset(str(KEYS.SCALE), data=mrgs['scale'])
            save.create_dataset(str(KEYS.SNAP), data=mrgs['snap'])
            save.create_dataset(str(KEYS.TASK), data=mrgs['task'])

            save.create_dataset(str(KEYS.ID), data=ids)
            save.create_dataset(str(KEYS.MASS), data=mass)
            save.create_dataset(str(KEYS.POS), data=pos)

        if self._verbose:
            msg = "Saved data for {} mergers to '{}' size {}".format(
                len(mrgs['scale']), fname, zio.get_file_size(fname))
            print(msg)

        return

    def _build_tree(self, mrgs):
        """
        The first BH is the "out" (remaining) BH, second BH is "in" (removed) BH
        """

        def mstr(mm):
            msg = "{:6d} a={:.8f} s={:3d} ".format(mm, mrgs['scale'][mm], mrgs['snap'][mm])
            msg += " in id={:20s} mass={:.8f} ".format(str(mrgs['id1'][mm]), mrgs['mass1'][mm])
            msg += "out id={:20s} mass={:.8f} ".format(str(mrgs['id2'][mm]), mrgs['mass2'][mm])
            return msg

        num = mrgs['scale'].size
        next = np.ones(num, np.int32) * -1
        prev = np.ones((num, 2), np.int32) * -1

        for mm in range(num):
            mids = [mrgs['id1'][mm], mrgs['id2'][mm]]
            ma = mrgs['scale'][mm]
            for nn in range(mm+1, num):
                nids = [mrgs['id1'][nn], mrgs['id2'][nn]]
                na = mrgs['scale'][nn]
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
                    raise ValueError(err)

                # Check if 'out' BH is in a subsequent merger
                if mids[BH_TYPE.OUT] in nids:
                    next[mm] = nn
                    for jj in range(2):
                        if mids[BH_TYPE.OUT] == nids[jj]:
                            prev[nn, jj] = mm
                            # Make sure masses increased
                            if mrgs['mass' + str(jj+1)][nn] < mrgs['mass2'][mm]:
                                err = "ERROR: BH mass loss between merger {} and {}".format(mm, nn)
                                logging.error(err)
                                logging.error(mstr(mm))
                                logging.error(mstr(nn))
                                raise ValueError(err)

                            break  # if

                    break  # for jj

        mrgs['next'] = next
        mrgs['prev'] = prev
        # num_next = np.zeros_like(next)
        # num_prev = np.zeros_like(prev)

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

            size = None
            keys = self.keys()
            for kk in keys:
                try:
                    vals = load[str(kk)][:]
                    ss = np.shape(vals)[0]
                    if size is None:
                        size = ss
                    elif size != ss:
                        msg = (
                            "WARNING: loaded array for '{}' has size {} ".format(kk, ss) +
                            "different from expected {}".format(size)
                        )
                        logging.warning(msg)

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
                print("Loaded {} mergers from '{}', created '{}'".format(size, fname, dt))

        self._size = size
        return
