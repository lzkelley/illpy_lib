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

import parse

import zcode.inout as zio
import zcode.math as zmath

import illpy_lib  # noqa
import illpy_lib.illcosmo
from illpy_lib.illbh import Processed, utils, mergers, BH_TYPE, VERBOSE, KEYS

DETS_RESOLUTION_LIMIT = False                  # Control flag for downsampling
DETS_RESOLUTION_TARGET = 1.0e-3                # in units of scale-factor
DETS_RESOLUTION_TOLERANCE = 0.5              # allowed fraction below `DETS_RESOLUTION_TARGET`
DETS_RESOLUTION_MIN_NUM = 10

ALLOW_LINE_PARSE_ERROR = True

if DETS_RESOLUTION_LIMIT and (DETS_RESOLUTION_MIN_NUM < 10):
    raise ValueError("ERROR: `DETS_RESOLUTION_MIN_NUM` must be >= 10!")

TOS_PARAMS_FNAME = os.path.join('setup', 'curie', 'param.txt-usedvalues')


class Details(Processed):

    _PROCESSED_FILENAME = "bh-details.hdf5"

    def _process(self):
        raise NotImplementedError()

    def _sort_add_unique(self, dets):
        # KEYS = self.KEYS
        idx = np.lexsort((dets[KEYS.SCALE], dets[KEYS.ID]))
        # skip = [KEYS.U_IDS, KEYS.U_INDICES, KEYS.U_COUNTS]
        for kk, vv in dets.items():
            # if kk in skip:
            #     continue
            dets[kk] = vv[idx, ...]

        u_ids, u_inds, u_counts = np.unique(dets[KEYS.ID], return_index=True, return_counts=True)
        # num_unique = u_ids.size
        dets[KEYS.U_IDS] = u_ids
        dets[KEYS.U_INDICES] = u_inds
        dets[KEYS.U_COUNTS] = u_counts
        return dets

    def _finalize_and_save(self, dets, fname=None, **header):
        dets = self._sort_add_unique(dets)

        # -- Save values to file
        # Get output filename for this snapshot
        #   path should have already been created in `_process` by rank=0
        if fname is None:
            fname = self.filename()
        self._save_to_hdf5(fname, __file__, dets, **header)
        if self._verbose:
            num_unique = dets[KEYS.U_IDS].size
            msg = "Saved data for {} details ({} unique) to '{}' size {}".format(
                len(dets[KEYS.SCALE]), num_unique, fname, zio.get_file_size(fname))
            print(msg)

        return


#   ============================================================================================
#   ===================================       TNG       ========================================
#   ============================================================================================


class Details_TNG_Task(Details):

    # _PROCESSED_FILENAME = "bh-details_task{task:04d}{res:}.hdf5"
    _PROCESSED_FILENAME = "bh-details_{task:04d}.hdf5"
    _SNAP_DIR_NAME = "details_{snap:04d}"

    def __init__(self, task, sim_path, *args, cosmo=None, **kwargs):
        self._task = task
        if cosmo is None:
            cosmo = illpy_lib.illcosmo.Simulation_Cosmology(sim_path)
        self._cosmo = cosmo

        if (self._cosmo is None) or (len(getattr(self._cosmo, 'scale', [])) == 0):
            err = "Failed to load cosmo, required for {}".format(self.__class__.__name__)
            logging.error(err)
            raise RuntimeError(err)

        super().__init__(sim_path, *args, **kwargs)
        # fname = self.filename()
        # self._load(fname, self._recreate)

        self._size = None
        num_snaps = len(self._cosmo.scale)
        recreate = self._recreate
        verbose = self._verbose
        exists = True
        for snap in range(num_snaps):
            fname = self.filename(snap)
            if not os.path.isfile(fname):
                if self._verbose:
                    print("Task {}: snap {} does not exist '{}'".format(task, snap, fname))
                exists = False
                break

        if recreate or (not exists):
            if verbose:
                print("Running `_process()`; recreate: {}, exists: {} ({})".format(
                    recreate, exists, fname))
            self._process()
        elif verbose:
            print("Files exist")

        return

    def _parse_raw_file_line(self, line):
        """
        TNG:
        fprintf(FdBlackHolesDetails, "BH=%llu %g %g %g %g %g\n",
            (long long) P[n].ID, All.Time, BPP(n).BH_Mass, mdot, rho, soundspeed);
        """

        format = (
            "BH={{{id}:d}} {{{scale}:g}} " +
            "{{{bh_mass}:g}} {{{bh_mdot}:g}} {{{bh_density}:g}} {{{bh_soundspeed}:g}}"
        ).format(
            id=KEYS.ID, scale=KEYS.SCALE,
            bh_mass=KEYS.BH_MASS, bh_mdot=KEYS.BH_Mdot,
            bh_density=KEYS.BH_Density, bh_soundspeed=KEYS.BH_SoundSpeed
        )

        # in TNG some soundspeed values are nan (from zero densities at seeding??)
        #    NOTE: big-s 'S' means any non-whitespace characters
        backup = (
            "BH={{{id}:d}} {{{scale}:g}} " +
            "{{{bh_mass}:g}} {{{bh_mdot}:g}} {{{bh_density}:g}} {{{bh_soundspeed}:4S}}"
        ).format(
            id=KEYS.ID, scale=KEYS.SCALE,
            bh_mass=KEYS.BH_MASS, bh_mdot=KEYS.BH_Mdot,
            bh_density=KEYS.BH_Density, bh_soundspeed=KEYS.BH_SoundSpeed
        )

        line = line.strip()
        det = parse.parse(format, line)
        # Try the backup parser and manually convert from str to float
        if det is None:
            det = parse.parse(backup, line).named
            if det is None:
                err = "ERROR: failed to parse line '{}'".format(line)
                logging.error(err)
                raise ValueError(err)

            det[KEYS.BH_SoundSpeed] = np.float(det[KEYS.BH_SoundSpeed])
        else:
            det = det.named

        return det

    def _finalize_and_save(self, dets):
        # KEYS = self.KEYS
        # dets = self._sort_add_unique(dets)
        verbose = self._verbose
        cosmo = self._cosmo
        snap_scales = cosmo.scale
        num_snaps = len(snap_scales)

        # KEYS = self.KEYS
        idx = np.argsort(dets[KEYS.SCALE])
        # skip = [KEYS.U_IDS, KEYS.U_INDICES, KEYS.U_COUNTS]
        for kk, vv in dets.items():
            dets[kk] = vv[idx, ...]

        det_scales = dets[KEYS.SCALE]
        SFIGS = 6
        # logging.warning("Rounding snapshot scalefactors up at {} decimal places".format(SFIGS))
        # Determine the number of decimal places for this number of sig-figs
        temp = SFIGS - 1 + np.fabs(np.floor(np.log10(snap_scales))).astype(int)
        # Round up at this number of sig-figs
        temp = 10**temp
        snap_scales = np.ceil(temp * snap_scales) / temp

        last_scale = 0.0
        check = np.zeros(det_scales.size, dtype=int)
        if verbose:
            tot_num = det_scales.size
            print("Organizing into {:3d} snapshots [{:.10e}...{:.10e}]".format(
                snap_scales.size, snap_scales[0], snap_scales[-1]))

        final_snap = num_snaps - 1
        # print_first_snap = verbose
        for snap, sca in enumerate(snap_scales):
            # idx_hi = (det_scales > last_scale)
            # idx_lo_1 = (det_scales <= sca)
            # idx_lo_2 = (np.isclose(det_scales, sca, rtol=1e-8) & (snap == final_snap))
            # idx = idx_hi & (idx_lo_1 | idx_lo_2)
            idx = (det_scales > last_scale) & (det_scales <= sca)
            check[idx] += 1
            snap_dets = {}
            for kk, vv in dets.items():
                snap_dets[kk] = vv[idx, ...]

            snap_dets = self._sort_add_unique(snap_dets)
            fname = self.filename(snap)
            self._save_to_hdf5(fname, __file__, snap_dets)
            if (snap == final_snap) and verbose:
                _num = len(snap_dets[KEYS.SCALE])
                _num_uniq = snap_dets[KEYS.U_IDS].size
                _frac = _num / tot_num
                msg = "task {:4d} :: saved {:.2e}/{:.2e} = {:.2e} details ({:.2e} unique) to '{}' size {}".format(
                    self._task, _num, tot_num, _frac, _num_uniq, fname, zio.get_file_size(fname))
                print(msg)

            last_scale = sca

        # --- Make sure each entry was written once and only once ---
        err = ""
        if np.any(check > 1):
            err += "ERROR: `check` values greater than one!  "
        if np.any(check < 1):
            err += "ERROR: `check` values less    than one!  "

        if len(err) > 0:
            bads = np.where(check != 1)[0]
            logging.error(err)

            logging.error("{} bad values: {}".format(len(bads), bads))
            logging.error("bad scales = {:.10e}   = {}".format(
                det_scales[bads][0], det_scales[bads]))

            # Delete files on failure
            for snap in range(num_snaps):
                fname = self.filename(snap)
                os.remove(fname)

            raise RuntimeError(err)

        return

    def _get_details_file_list(self, sim_path):
        task = self._task
        fname_in = 'blackhole_details_{:d}.txt'.format(task)
        fname_in = os.path.join(self._sim_path, 'output', 'blackhole_details', fname_in)

        if not os.path.isfile(fname_in):
            err = "ERROR: `fname` '{}' does not exist!".format(fname_in)
            logging.error(err)
            raise FileNotFoundError(err)

        # return as list to match expectations in `_process()` (for generalizeability)
        return [fname_in]

    def _process(self):
        verbose = self._verbose
        cosmo = self._cosmo
        scales = cosmo.scale
        num_snaps = len(scales)
        if verbose:
            print("{} snapshots from cosmo".format(num_snaps))

        # --- Make sure output directories for each snapshot exist ---
        for snap in range(num_snaps):
            fname_out = self.filename(snap)
            path = os.path.dirname(fname_out)
            if not os.path.exists(path):
                if verbose:
                    print("Creating path '{}'".format(path))
                os.makedirs(path)
            if not os.path.isdir(path):
                err = "ERROR: filename '{}' path '{}' is not a directory!".format(fname_out, path)
                logging.error(err)
                raise FileNotFoundError(err)

        # task = self._task
        # fname_in = 'blackhole_details_{:d}.txt'.format(task)
        # fname_in = os.path.join(self._sim_path, 'output', 'blackhole_details', fname_in)
        #
        # if not os.path.isfile(fname_in):
        #     err = "ERROR: `fname` '{}' does not exist!".format(fname_in)
        #     logging.error(err)
        #     raise FileNotFoundError(err)
        fname_in_list = self._get_details_file_list(self._sim_path)

        details = []
        for fname_in in fname_in_list:
            prev = None
            print_first_line = verbose
            for ll, line in enumerate(open(fname_in, 'r').readlines()):
                if print_first_line:
                    print("First line: '{}'".format(line.strip()))
                    print_first_line = False

                try:
                    vals = self._parse_raw_file_line(line)
                except Exception as err:
                    logging.error("File '{}', line {}: {}".format(fname_in, ll, str(err)))
                    raise err

                sc = vals[KEYS.SCALE]
                if prev is None:
                    prev = sc
                elif sc < prev:
                    err = "Task: {}, {} - scale:{:.8f} is before prev:{:.8f}!".format(
                        self._task, fname_in, sc, prev)
                    logging.error(err)
                    raise ValueError(err)

                details.append(vals)

        num_dets = len(details)
        self._size = num_dets
        if verbose:
            print("Loaded {} details from '{}'".format(num_dets, fname_in))

        dets = {}
        for dd, temp in enumerate(details):
            if dd == 0:
                for kk, vv in temp.items():
                    if np.isscalar(vv):
                        shp = num_dets
                        tt = type(vv)
                    else:
                        shp = (num_dets,) + np.shape(vv)
                        tt = vv.dtype

                    if kk == KEYS.ID:
                        tt = np.uint64

                    # print(f"initializing {kk} to shape {shp} type {tt}  - shape(vv) = {np.shape(vv)}")
                    dets[kk] = np.zeros(shp, dtype=tt)

            for kk, vv in temp.items():
                try:
                    dets[kk][dd, ...] = vv
                except OverflowError as err:
                    print(kk)
                    print(err)
                    print(f"FAILED on kk={kk}, type={vv.dtype}")
                    print(f"dets[{kk}].dtype = {dets[kk].dtype}")
                    raise

        self._finalize_and_save(dets)
        return

    '''
    def path_processed_snap(self, snap):
        fname = self._PROCESSED_FILENAME.format(task=self._task)
        temp = self._SNAP_DIR_NAME.format(snap=snap)
        path = os.path.join(self._processed_path, temp, fname)
        return path
    '''

    @classmethod
    def _snap_path(cls, snap, processed_path):
        temp = cls._SNAP_DIR_NAME.format(snap=snap)
        path = os.path.join(processed_path, temp, '')
        return path

    def filename(self, snap):
        # if not DETS_RESOLUTION_LIMIT:
        #     res = ""
        # else:
        #     res = "_res{+:.2f}".format(np.log10(DETS_RESOLUTION_TARGET))
        # temp = self._PROCESSED_FILENAME.format(task=self._task, res=res)

        # temp = self._PROCESSED_FILENAME.format(task=self._task)
        # fname = os.path.join(self._processed_path, temp)
        path = self._snap_path(snap, self._processed_path)
        fname = self._PROCESSED_FILENAME.format(task=self._task)
        fname = os.path.join(path, fname)

        return fname


class Details_TNG_Snap(Details):

    _PROCESSED_FILENAME = "bh-details_snap-{snap:04d}.hdf5"
    _SNAP_DIR_NAME = "details_{snap:04d}"
    _SKIP_DERIVED_KEYS = KEYS._DERIVED

    def __init__(self, snap, sim_path, *args, cosmo=None, **kwargs):
        self._snap = snap
        self._cosmo = cosmo
        super().__init__(sim_path, *args, **kwargs)
        self._load(self.filename(snap), self._recreate)
        return

    def _process(self):
        verbose = self._verbose

        # Get list of files
        snap = self._snap
        snap_path = self._snap_path(snap, self._processed_path)
        pattern = Details_TNG_Task._PROCESSED_FILENAME.replace('{task:04d}', '*')
        pattern = os.path.join(snap_path, pattern)
        files = sorted(glob.glob(pattern))
        if verbose:
            print(f"found {len(files)} files matching '{pattern}'")

        if len(files) == 0:
            raise FileNotFoundError(f"No files found for snap {snap} matching '{pattern}'!")

        details = {}
        first = True
        size = 0
        for ii, fname in enumerate(files):
            new_size = None
            new_size_key = None
            with h5py.File(fname, 'r') as h5:
                for kk, vv in h5.items():
                    if kk in self._SKIP_DERIVED_KEYS:
                        continue

                    temp = vv[()]
                    # if np.ndim(temp) > 0:
                    temp_size = np.shape(temp)[0]
                    if new_size is None:
                        new_size = temp_size
                        new_size_key = kk
                    elif (kk not in KEYS._U_KEYS) and (new_size != temp_size):
                        msg = (
                            f"Size of '{kk}'={temp_size} does not match "
                            f"previous size {new_size} from '{new_size_key}!"
                        )
                        logging.warning(msg)

                    if kk not in details:
                        if not first:
                            err = f"WARNING: key '{kk}' found first in file {ii}, {fname}!"
                            logging.error(err)
                            raise RuntimeError(err)

                        details[kk] = temp
                    else:
                        prev = details[kk]
                        details[kk] = np.concatenate([prev, temp], axis=0)

                if first and len(h5.keys()) > 0:
                    first = False

            if new_size is not None:
                size += new_size
                if verbose:
                    print(f"Loaded {new_size} entires from '{fname}'")

        self._size = size
        if verbose:
            print(f"Loaded {size} details")

        self._details = details
        try:
            self._finalize_and_save(details)
        except Exception as err:
            logging.exception(err)
            print(err)
            raise

        return

    @classmethod
    def _snap_path(cls, snap, processed_path):
        temp = cls._SNAP_DIR_NAME.format(snap=snap)
        path = os.path.join(processed_path, temp, '')
        return path

    def filename(self, snap=None):
        if snap is None:
            snap = self._snap
        path = self._snap_path(snap, self._processed_path)
        fname = self._PROCESSED_FILENAME.format(snap=snap)
        fname = os.path.join(path, fname)
        return fname


class Details_TOS_Task(Details_TNG_Task):

    def __init__(self, task, sim_path, *args, cosmo=None, **kwargs):
        if cosmo is None:
            cosmo = illpy_lib.illcosmo.Simulation_Cosmology(sim_path, fname_params=TOS_PARAMS_FNAME)

        super().__init__(task, sim_path, *args, cosmo=cosmo, **kwargs)
        return

    def _get_details_file_list(self, sim_path):
        files = []
        path = os.path.join(sim_path, 'txt-files', 'txtfiles_new', '')
        task = self._task
        fname = 'blackhole_details_{:d}.txt'.format(task)

        for dirpath, dirnames, filenames in os.walk(path):
            if fname in filenames:
                temp = os.path.join(dirpath, fname)
                files.append(temp)

        if self._verbose:
            print(f"Found {len(files)} details files matching {fname}")

        return files


class Details_TOS_Snap(Details_TNG_Snap):
    pass


def process_details_snaps(mode, sim_path, proc_path, recreate=False, verbose=VERBOSE):
    try:
        comm = MPI.COMM_WORLD
    except NameError:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    comm.barrier()

    beg = datetime.now()
    if comm.rank == 0:

        pattern = os.path.join(sim_path, 'output', 'snapdir_*')
        num_snaps = len(glob.glob(pattern))
        if num_snaps == 0:
            err = f"Failed to find snapshots matching pattern '{pattern}'!"
            logging.exception(err)
            raise RuntimeError(err)

        if mode == 'tng':
            Det_Class_Task = Details_TNG_Task
            Det_Class_Snap = Details_TNG_Snap
            fname_params = None

            pattern = os.path.join(
                sim_path, 'output', 'blackhole_details', 'blackhole_details_*.txt')

            det_files = sorted(glob.glob(pattern))
            num_tasks = len(det_files)
            if verbose:
                print("Found {} details files (tasks)".format(num_tasks))
            if len(det_files) == 0:
                err = "ERROR: no details directories found (pattern: '{}')".format(pattern)
                logging.error(err)
                raise FileNotFoundError(err)

        elif mode == 'tos':
            Det_Class_Task = Details_TOS_Task
            Det_Class_Snap = Details_TOS_Snap
            fname_params = TOS_PARAMS_FNAME
            num_tasks = 8192

        else:
            raise ValueError(f"Unrecognized `mode` '{mode}'!")

        task_list = np.arange(num_tasks).tolist()
        snap_list = np.arange(num_snaps).tolist()
        np.random.seed(1234)
        np.random.shuffle(task_list)
        np.random.shuffle(snap_list)
        task_list = np.array_split(task_list, comm.size)
        snap_list = np.array_split(snap_list, comm.size)

        cosmo = illpy_lib.illcosmo.Simulation_Cosmology(sim_path, fname_params=fname_params)
        # Make sure all paths exist
        for snap in range(len(cosmo.scale)):
            snap_path = Det_Class_Task._snap_path(snap, proc_path)
            if not os.path.isdir(snap_path):
                os.makedirs(snap_path)

    else:
        task_list = None
        snap_list = None
        cosmo = None
        Det_Class_Task = None
        Det_Class_Snap = None

    task_list = comm.scatter(task_list, root=0)
    snap_list = comm.scatter(snap_list, root=0)
    cosmo = comm.bcast(cosmo, root=0)
    Det_Class_Task = comm.bcast(Det_Class_Task, root=0)
    Det_Class_Snap = comm.bcast(Det_Class_Snap, root=0)

    # ---- Process All Processor/Task Output Files ----
    comm.barrier()
    print(f"rank: {comm.rank}, {len(task_list)} tasks: {task_list}".format())
    num_lines = 0
    for task in task_list:
        details = Det_Class_Task(task, sim_path, proc_path,
                                 cosmo=cosmo, recreate=recreate, verbose=verbose)
        temp = details.size
        if (temp is None) and verbose:
            print("No details loaded for task {}".format(task))
            continue

        num_lines += temp

    end = datetime.now()
    if verbose:
        print(f"Rank: {comm.rank}, {num_lines:.2e} lines done at {end}, after {(end-beg)}")

    num_lines = comm.gather(num_lines, root=0)

    if comm.rank == 0:
        tot_num_lines = np.sum(num_lines)
        ave = np.mean(num_lines)
        med = np.median(num_lines)
        std = np.std(num_lines)
        if verbose:
            print("Tot lines={:.2e}, med={:.2e}, ave={:.2e}±{:.2e}".format(
                tot_num_lines, med, ave, std))

    # ---- Process All Snapshots Output Files ----
    comm.barrier()
    print(f"rank: {comm.rank}, {len(snap_list)} snaps: {snap_list}".format())
    num_lines = 0
    for snap in snap_list:
        details = Det_Class_Snap(snap, sim_path, proc_path,
                                 cosmo=cosmo, recreate=recreate, verbose=verbose)
        temp = details.size
        if (temp is None) and verbose:
            print("No details loaded for task {}".format(task))
            continue

        num_lines += temp

    end = datetime.now()
    if verbose:
        print(f"Rank: {comm.rank}, {num_lines:.2e} lines done at {end}, after {(end-beg)}")

    num_lines = comm.gather(num_lines, root=0)
    if comm.rank == 0:
        tot_num_lines = np.sum(num_lines)
        ave = np.mean(num_lines)
        med = np.median(num_lines)
        std = np.std(num_lines)
        if verbose:
            print("Tot lines={:.2e}, med={:.2e}, ave={:.2e}±{:.2e}".format(
                tot_num_lines, med, ave, std))

    if comm.rank == 0:
        tail = f"Done at {str(end)} after {str(end-beg)} ({(end-beg).total_seconds()})"
        print("\n" + "=" * len(tail) + "\n" + tail + "\n")

    comm.barrier()
    return


if __name__ == "__main__":
    logging.warning("Loading MPI...")
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    beg = datetime.now()
    recreate = ('-r' in sys.argv) or ('--recreate' in sys.argv)
    verbose = ('-v' in sys.argv) or ('--verbose' in sys.argv) or VERBOSE

    if comm.rank == 0:
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

    else:
        mode = None
        sim_path = None
        proc_path = None

    mode = comm.bcast(mode, root=0)
    sim_path = comm.bcast(sim_path, root=0)
    proc_path = comm.bcast(proc_path, root=0)
    comm.barrier()

    process_details_snaps(mode, sim_path, proc_path, recreate=recreate, verbose=verbose)


def OLD_OLD_OLD_OLD():

    '''
    class Details_TNG(Details):

        class KEYS(Details_TNG_Task.KEYS):
            pass

        def _process(self):
            verbose = self._verbose
            if self._sim_path is None:
                err = "ERROR: cannot process {} without `sim_path` set!".format(self.__class__)
                logging.error(err)
                raise ValueError(err)

            # Check output filename
            fname_out = self.filename
            if verbose:
                print("Output filename: '{}'".format(fname_out))
            path = os.path.dirname(fname_out)
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.isdir(path):
                err = "ERROR: filename '{}' path '{}' is not a directory!".format(fname_out, path)
                logging.error(err)
                raise FileNotFoundError(err)

            task = 0
            KEYS = self.KEYS
            first = True
            # tot_input = 0
            # tot_output = 0
            tot_count = 0
            # ave_ratio = 0.0
            num_tasks = 0

            dets = {}
            while True:
                if verbose:
                    print("task = {}".format(task))

                try:
                    dets_task = Details_TNG_Task(
                        task, sim_path=self._sim_path, processed_path=self._processed_path,
                        verbose=verbose)
                except FileNotFoundError as err:
                    if verbose:
                        print("Could not find files for task {}; ending.".format(task))
                    break

                num_tasks += 1
                scales = dets_task[KEYS.SCALE]
                ids = dets_task[KEYS.ID]
                num_this_task = scales.size
                if verbose:
                    print("\tnum_this_task = {}".format(num_this_task))

                if num_this_task == 0:
                    task += 1
                    continue

                tot_count += num_this_task
                u_ids = dets_task[KEYS.U_IDS]
                u_indices = dets_task[KEYS.U_INDICES]
                u_counts = dets_task[KEYS.U_COUNTS]
                for ii, xx, nn in zip(u_ids, u_indices, u_counts):
                    if not np.all(ids[xx] == ids[xx:xx+nn]):
                        err = "ERROR: task {}, ids inconsistent for BH {}!".format(task, ii)
                        logging.error(err)
                        raise ValueError(err)
                    if (xx > 0) and (ids[xx] == ids[xx-1]):
                        err = "ERROR: task {}, ids start before expected for BH {}!".format(task, ii)
                        logging.error(err)
                        raise ValueError(err)
                    if (xx+nn < num_this_task) and (ids[xx] == ids[xx+nn]):
                        err = "ERROR: task {}, ids continue beyond expected for BH {}!".format(task, ii)
                        logging.error(err)
                        raise ValueError(err)
                    if np.any(np.diff(scales[xx:xx+nn]) < 0.0):
                        err = "ERROR: task {}, BH {} scales not monotonically increasing!".format(
                            task, ii)
                        logging.error(err)
                        raise ValueError(err)

                    # if DETS_RESOLUTION_LIMIT:
                    #     dets = self._downsample()
                    #
                    # else:

                    for kk in KEYS:
                        if kk.startswith('unique'):
                            continue

                        temp = dets_task[kk][xx:xx+nn]
                        if first:
                            dets[kk] = temp
                        else:
                            prev = dets[kk]
                            dets[kk] = np.concatenate([prev, temp])

                    first = False

                task += 1
                if task > 5:
                    logging.warning("BREAKING at task {}!".format(task))
                    break

            if verbose:
                print("{:.4e} total entries from {:d} tasks".format(tot_count, num_tasks))

            # ave_ratio /= tot_count
            # tot_ratio = tot_output / tot_input
            # if verb:
            #     print("Downsampled from {:.2e} ==> {:.2e}".format(tot_input, tot_output))
            #     print("Total compression: {:.2e}, average: {:.2e}".format(tot_ratio, ave_ratio))

            self._finalize_and_save(dets)
            return


    #   ============================================================================================
    #   ================================       New Seeds       =====================================
    #   ============================================================================================


    class Details_New(Details):

        class KEYS(Details.KEYS):
            MASS   = 'mass'
            MDOT_B = 'mdot_bondi'
            POT    = 'potential'

            POS    = 'pos'
            VEL    = 'vel'

            SNAP   = 'snap'
            TASK   = 'task'

            _DERIVED = Details.KEYS._DERIVED + [TASK, SNAP]
            _INTERP_KEYS = Details.KEYS._INTERP_KEYS + [MASS, MDOT_B, POT, POS, VEL]

        def _process(self):
            if self._sim_path is None:
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
                    downsample_flag = DETS_RESOLUTION_LIMIT
                    if downsample_flag:
                        downsample_flag = (res < DETS_RESOLUTION_TARGET / (1.0 + DETS_RESOLUTION_TOLERANCE))

                    if downsample_flag:
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
                            if kk.startswith('unique'):
                                continue

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


    class Details_Snap_New(Details_New):

        _PROCESSED_FILENAME = "bh-details_{snap:03d}.hdf5"

        def __init__(self, snap, *args, **kwargs):
            self._snap = snap
            super().__init__(*args, **kwargs)
            return

        def _parse_raw_file_line(self, line):
            """

            "%.8f %llu  %.8e %.8e  %.8e %.8e %.8e %.8e  %.8e %.8e %.8e  %.8e %.8e %.8e"
                All.Time, (long long) P[n].ID, P[n].Mass, BPP(n).BH_Mass,
                BPP(n).BH_MdotBondi, BPP(n).BH_Mdot, pot, BPP(n).BH_Density,
                P[n].Pos[0], P[n].Pos[1], P[n].Pos[2], P[n].Vel[0], P[n].Vel[1], P[n].Vel[2]

            """
            KEYS = self.KEYS

            format = (
                "{scale:.8f} {id:d}  " +
                "{mass:.8e} {bh_mass:.8e}  " +
                "{mdot_bondi:.8e} {mdot:.8e} {potential:.8e} {density:.8e}  " +
                "{px:.8e} {py:.8e} {pz:.8e}  " +
                "{vx:.8e} {vy:.8e} {vz:.8e}"
            )

            format_g = (
                "{scale:g} {id:d}  " +
                "{mass:g} {bh_mass:g}  " +
                "{mdot_bondi:g} {mdot:g} {potential:g} {density:g}  " +
                "{px:g} {py:g} {pz:g}  " +
                "{vx:g} {vy:g} {vz:g}"
            )

            line = line.strip()
            for form in [format, format_g]:
                temp = parse.parse(form, line)
                if temp is not None:
                    break
            else:
                err = "ERROR: failed to parse line '{}'".format(line)
                # logging.error(err)
                raise ValueError(err)

            rv = {}
            for kk in KEYS:
                if (kk in KEYS._DERIVED) or (kk in [KEYS.POS, KEYS.VEL]):
                    continue
                rv[kk] = temp[kk]

            rv[KEYS.POS] = np.array([temp['px'], temp['py'], temp['pz']])
            rv[KEYS.VEL] = np.array([temp['vx'], temp['vy'], temp['vz']])

            return rv

        def _process(self):
            KEYS = self.KEYS
            snap = self._snap
            ddir = 'details_{:03d}'.format(snap)
            # ddir = os.path.join(self._sim_path, 'output', 'blackholes', ddir)
            ddir = os.path.join(self._sim_path, 'blackholes', ddir)

            if not os.path.isdir(ddir):
                err = "ERROR: `ddir` '{}' does not exist!".format(ddir)
                logging.error(err)
                raise FileNotFoundError(err)

            # Check output filename
            fname = self.filename
            path = os.path.dirname(fname)
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.isdir(path):
                err = "ERROR: filename '{}' path '{}' is not a directory!".format(fname, path)
                logging.error(err)
                raise FileNotFoundError(err)

            try:
                cosmo = illpy_lib.illcosmo.Simulation_Cosmology(self._sim_path, verbose=self._verbose)
                scales = cosmo.scale
                num_snaps = len(scales)
                lo = scales[snap-1] if snap > 0 else 0.0
                try:
                    hi = scales[snap]
                except IndexError:
                    # Last snapshot may not be present
                    if snap != len(scales):
                        raise
                    else:
                        hi = 1.0
                        msg = "Did not find snapshot {}, setting `hi` = {}".format(snap, hi)
                        logging.warning(msg)

            except FileNotFoundError as err:
                num_snaps = None
                msg = "WARNING: could not load cosmology (snapshots missing?): '{}'".format(str(err))
                logging.warning(msg)
                logging.warning("WARNING: Cannot compare merger times to snapshot times!")

            dfils = os.path.join(ddir, "blackhole_details_*.txt")
            dfils = sorted(glob.glob(dfils))

            num_files = len(dfils)
            if self._verbose:
                print(f"snap {snap}, dir {ddir}, files: {num_files}")
            if len(dfils) == 0:
                err = "ERROR: snap={} : no files found in '{}'".format(snap, ddir)
                raise FileNotFoundError(err)

            details = []
            num_dets = 0
            for ii, dfil in enumerate(dfils):
                task = os.path.basename(dfil).split('_')[-1].split('.')[0]
                task = int(task)

                prev = None
                for ll, line in enumerate(open(dfil, 'r').readlines()):
                    try:
                        vals = self._parse_raw_file_line(line)
                    except ValueError as err:
                        if ALLOW_LINE_PARSE_ERROR:
                            continue
                        else:
                            logging.error(str(err))
                            raise err

                    vals[KEYS.SNAP] = snap
                    vals[KEYS.TASK] = task

                    sc = vals[KEYS.SCALE]
                    if prev is None:
                        prev = sc
                    elif sc < prev:
                        err = f"Snap: {snap}, file: {ii} - scale:{sc:.8f} is before prev:{prev:.8f}!"
                        logging.error(err)
                        raise ValueError(err)

                    if (num_snaps is not None):
                        lo_flag = (sc < lo) & (not np.isclose(sc, lo, rtol=1e-6, atol=0.0))
                        hi_flag = (sc > hi) & (not np.isclose(sc, hi, rtol=1e-6, atol=0.0))
                        if (lo_flag or hi_flag):
                            err = "Snap: {}, file: {}, scale:{:.8f} not in [{:.8f}, {:.8f}]!".format(
                                snap, ii, sc, lo, hi)
                            logging.error(err)
                            raise ValueError(err)

                    details.append(vals)
                    num_dets += 1

            dets = {}
            for dd, temp in enumerate(details):
                if dd == 0:
                    for kk, vv in temp.items():
                        if np.isscalar(vv):
                            shp = num_dets
                            tt = type(vv)
                        else:
                            shp = (num_dets,) + np.shape(vv)
                            tt = vv.dtype

                        dets[kk] = np.zeros(shp, dtype=tt)

                for kk, vv in temp.items():
                    dets[kk][dd, ...] = vv

            if len(details) == 0:
                dets = {kk: np.array([]) for kk in KEYS}

            self._finalize_and_save(dets)
            return

        @property
        def filename(self):
            if self._filename is None:
                temp = self._PROCESSED_FILENAME.format(snap=self._snap)
                self._filename = os.path.join(self._processed_path, temp)

            return self._filename

    '''

    '''
    def process_details_snaps__new(sim_path, proc_path, recreate=False, verbose=VERBOSE):
        try:
            comm = MPI.COMM_WORLD
        except NameError:
            logging.warning("Loading MPI...")
            from mpi4py import MPI
            comm = MPI.COMM_WORLD

        beg = datetime.now()
        if comm.rank == 0:
            temp = 'details_' + '[0-9]' * 3
            # path_bhs = os.path.join(sim_path, 'output', 'blackholes')
            path_bhs = os.path.join(sim_path, 'blackholes')
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
    '''

    '''
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
    '''


    '''
    def _downsample(self):
        raise NotImplementedError("NOT TESTED!")

        if nn > DETS_RESOLUTION_MIN_NUM:
            span = scales[xx+nn-1] - scales[xx]
            res = span / nn
        else:
            res = np.inf
            span = None

        if downsample_flag:
            downsample_flag = (res < DETS_RESOLUTION_TARGET / (1.0 + DETS_RESOLUTION_TOLERANCE))

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
                yvals = dets_task[kk][xx:xx+nn]
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
                temp = dets_task[kk][xx:xx+new_num]

            if first:
                dets[kk] = temp
            else:
                prev = dets[kk]
                dets[kk] = np.concatenate([prev, temp])

        first = False
        tot_output += new_num
        ave_ratio += (new_num / nn)
    '''
