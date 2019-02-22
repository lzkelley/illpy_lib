"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import shutil
from datetime import datetime

import numpy as np
import h5py

import zcode.inout as zio
import zcode.math as zmath


try:
    import illpy_lib
except ImportError:
    PATH_ILLPY_LIB = "/n/home00/lkelley/illustris/redesign/illpy_lib/"
    if PATH_ILLPY_LIB not in sys.path:
        print("Added path to `illpy_lib`: '{}'".format(PATH_ILLPY_LIB))
        sys.path.append(PATH_ILLPY_LIB)

    import illpy_lib  # noqa


from illpy_lib.illbh import Core, MERGERS, BH_TYPE, load_hdf5_to_mem, MERGERS_PHYSICAL_KEYS
from illpy_lib.constants import DTYPE


VERSION = 0.4


def main(reorganize_flag=True, crosscheck_flag=True):

    core = Core(sets=dict(LOG_FILENAME='log_illbh-mergers.log'))
    log = core.log
    log.info("mergers.main()")

    log.debug("`reorganize_flag` = {}".format(reorganize_flag))
    if reorganize_flag:
        load_temp_mergers(core=core)

    log.debug("`crosscheck_flag` = {}".format(crosscheck_flag))
    if crosscheck_flag:
        load_fixed_mergers(core=core)

    return


def load_temp_mergers(core=None, recreate=None):
    core = Core.load(core)
    log = core.log
    log.debug("load_temp_mergers()")

    if recreate is None:
        recreate = core.sets.RECREATE

    fname_temp = core.paths.fname_mergers_temp()
    exists = os.path.exists(fname_temp)
    log.debug("Temporary merger file '{}' exists: {}".format(fname_temp, exists))

    if recreate or not exists:
        log.warning("Reoganizing merger files to temporary file '{}'".format(fname_temp))
        _reorganize_files(core, fname_temp)

    data = load_hdf5_to_mem(fname_temp)
    keys = list(data.keys())
    log.info("Loaded {} temporary mergers, keys: {}".format(
        data[MERGERS.SCALES].size, keys))

    return data


def _reorganize_files(core, fname_out):
    """
    """

    core = Core.load(core)
    log = core.log
    log.debug("mergers._reorganize_files()")

    # Use a size guaranteed to be larger than needed (need ~ 3e4)
    BUFFER_SIZE = int(1e5)
    scales = np.zeros(BUFFER_SIZE, dtype=DTYPE.SCALAR)
    masses = np.zeros((BUFFER_SIZE, 2), dtype=DTYPE.SCALAR)
    mids = np.zeros((BUFFER_SIZE, 2), dtype=DTYPE.ID)

    raw_fnames = core.paths.mergers_input
    cnt = 0
    for ii, raw in enumerate(core.tqdm(raw_fnames, desc='Raw files')):
        log.debug("File {}: '{}'".format(ii, raw))

        file_cnt = cnt
        with open(raw, 'r') as data:
            for line in data.readlines():
                time, out_id, out_mass, in_id, in_mass = _parse_merger_line(line)

                scales[cnt] = time
                # mids[cnt, :] = [out_id, in_id]
                # masses[cnt, :] = [out_mass, in_mass]
                mids[cnt, BH_TYPE.OUT] = out_id
                mids[cnt, BH_TYPE.IN] = in_id
                masses[cnt, BH_TYPE.OUT] = out_mass
                masses[cnt, BH_TYPE.IN] = in_mass
                cnt += 1

            file_cnt = cnt - file_cnt
            log.debug("\tRead {} lines".format(file_cnt))

    num_mergers = cnt
    log.info("Read {} mergers".format(num_mergers))

    # Cutoff unused section of arrays
    scales = scales[:cnt]
    mids = mids[:cnt]
    masses = masses[:cnt]

    # Sort by scale-factors
    inds = np.argsort(scales)
    scales = scales[inds]
    mids = mids[inds, :]
    masses = masses[inds, :]

    mrgs = {
        MERGERS.SCALES: scales,
        MERGERS.IDS: mids,
        MERGERS.MASSES: masses
    }

    _save_mergers(core, mrgs, fname_out)

    # Create Mapping Between Mergers and Snapshots
    # mapM2S, mapS2M, ontop = _map_to_snapshots(scales)

    # Find the snapshot that each merger directly precedes
    '''
    log.debug("Calculating merger snapshots")
    snap_nums, ontop_next, ontop_prev = _map_to_snapshots(scales)

    fname_temp = zio.modify_filename(fname_out, prepend='_')
    log.info("Writing to file '{}'".format(fname_temp))
    with h5py.File(fname_temp, 'w') as out:

        out.attrs[MERGERS.RUN] = core.sets.RUN_NUM
        out.attrs[MERGERS.NUM] = len(scales)
        out.attrs[MERGERS.CREATED] = str(datetime.now().ctime())
        out.attrs[MERGERS.VERSION] = VERSION
        out.attrs[MERGERS.FILE] = fname_out

        out.create_dataset(MERGERS.SCALES, data=scales)
        out.create_dataset(MERGERS.IDS, data=mids)
        out.create_dataset(MERGERS.MASSES, data=masses)

        out.create_dataset(MERGERS.SNAP_NUMS, data=snap_nums)
        out.create_dataset(MERGERS.ONTOP_NEXT, data=ontop_next)
        out.create_dataset(MERGERS.ONTOP_PREV, data=ontop_prev)

    log.info("Renaming temporary file")
    log.debug("\t'{}' ==> '{}'".format(fname_temp, fname_out))
    shutil.move(fname_temp, fname_out)

    size_str = zio.get_file_size(fname_out)
    log.info("Saved {} mergers to '{}', size {}".format(num_mergers, fname_out, size_str))
    '''

    return fname_out


def _map_to_snapshots(scales):
    import illpy_lib.illcosmo
    cosmo = illpy_lib.illcosmo.cosmology.Cosmology()
    snap_scales = cosmo.scales()

    snap_nums = np.searchsorted(snap_scales, scales, side='left')
    # Find mergers which are at almost exactly the same time as their subsequent snapshot
    ontop_next = np.isclose(scales, snap_scales[snap_nums], rtol=1e-4, atol=1e-6)
    # Find mergers which are at almost exactly the same time as their previous snapshot
    ontop_prev = np.isclose(scales, snap_scales[snap_nums-1], rtol=1e-4, atol=1e-6)

    return snap_nums, ontop_next, ontop_prev


def load_fixed_mergers(core=None, recreate=None):
    core = Core.load(core)
    log = core.log
    log.info("load_fixed_mergers()")

    if recreate is None:
        recreate = core.sets.RECREATE

    fname_fixed = core.paths.fname_mergers_fixed()
    exists = os.path.exists(fname_fixed)
    log.debug("Fixed merger file '{}' exists: {}".format(fname_fixed, exists))

    if recreate or not exists:
        log.warning("Crosschecking mergers")
        _crosscheck_mergers(core, fname_fixed)

    data = load_hdf5_to_mem(fname_fixed)
    keys = list(data.keys())
    log.info("Loaded {} fixed mergers, keys: {}".format(
        data[MERGERS.NUM], keys))

    return data


def _crosscheck_mergers(core, fname_out):
    log = core.log
    log.info("mergers._crosscheck_mergers()")

    from illpy_lib.illbh import matcher

    mrgs = load_temp_mergers(core=core)
    scales = mrgs[MERGERS.SCALES]
    mids = mrgs[MERGERS.IDS]
    masses = mrgs[MERGERS.MASSES]

    num_mergers_temp = scales.size
    log.info("Loaded {} raw mergers".format(num_mergers_temp))

    # Remove entries where IDs match a second time (IS THIS ENOUGH?!)

    # First sort by ``BH_TYPE.IN`` then ``BH_TYPE.OUT`` (reverse of given order)
    sort = np.lexsort((mids[:, BH_TYPE.OUT], mids[:, BH_TYPE.IN]))

    bads = np.zeros_like(scales, dtype=bool)
    mismatch = np.zeros_like(bads)

    # Iterate over all entries
    for ii in range(num_mergers_temp - 1):
        this_ind = sort[ii]

        this = mids[this_ind]
        jj = ii+1
        next_ind = sort[jj]

        # Look through all examples of same BH_TYPE.IN
        while (mids[next_ind, BH_TYPE.IN] == this[BH_TYPE.IN]):
            # If BH_TYPE.OUT also matches, this is a duplicate -- store first entry as bad
            if (mids[next_ind, BH_TYPE.OUT] == this[BH_TYPE.OUT]):

                # Double check that time also matches
                if (not np.isclose(scales[this_ind], scales[next_ind])):
                    # num_mismatch += 1
                    mismatch[ii] = True

                # bad_inds.append(this_ind)
                bads[ii] = True
                break

            jj += 1
            next_ind = sort[jj]

    log.info("Num duplicates       = {}".format(zio.frac_str(bads)))
    log.info("Num mismatched times = {}".format(zio.frac_str(mismatch)))

    # Remove Duplicate Entries
    goods = ~bads
    scales = scales[goods]
    mids = mids[goods]
    masses = masses[goods]

    # Recalculate maps
    mapM2S, mapS2M, ontop = _map_to_snapshots(scales)
    snap_nums, ontop_next, ontop_prev = _map_to_snapshots(scales)

    # Fix Merger 'Out' Masses
    masses = mrgs[MERGERS.MASSES]
    bef = zmath.stats_str(masses[:, BH_TYPE.OUT])

    # WARNING: Probably dont really need this???
    masses = matcher.infer_merger_out_masses(core=core, mrgs=mrgs)
    aft = zmath.stats_str(masses[:, BH_TYPE.OUT])
    log.info("Out masses: {} ==> {}".format(bef, aft))
    mrgs[MERGERS.MASSES] = masses

    _save_mergers(core, mrgs, fname_out)

    return


def _save_mergers(core, mrgs, fname_out):

    log = core.log
    log.debug("Calculating merger snapshots")
    scales = mrgs[MERGERS.SCALES]
    snap_nums, ontop_next, ontop_prev = _map_to_snapshots(scales)
    num_mergers = len(scales)

    fname_temp = zio.modify_filename(fname_out, prepend='_')
    log.info("Writing to file '{}'".format(fname_temp))
    with h5py.File(fname_temp, 'w') as out:

        # out.attrs[MERGERS.RUN] = core.sets.RUN_NUM
        # out.attrs[MERGERS.NUM] = len(scales)
        # out.attrs[MERGERS.CREATED] = str(datetime.now().ctime())
        # out.attrs[MERGERS.VERSION] = VERSION
        # out.attrs[MERGERS.FILE] = fname_out
        out[MERGERS.RUN] = core.sets.RUN_NUM
        out[MERGERS.NUM] = num_mergers
        out[MERGERS.CREATED] = str(datetime.now().ctime())
        out[MERGERS.VERSION] = VERSION
        out[MERGERS.FILE] = fname_out

        for key in MERGERS_PHYSICAL_KEYS:
            out.create_dataset(key, data=mrgs[key])

        # out.create_dataset(MERGERS.SCALES, data=scales)
        # out.create_dataset(MERGERS.IDS, data=mids)
        # out.create_dataset(MERGERS.MASSES, data=masses)

        out.create_dataset(MERGERS.SNAP_NUMS, data=snap_nums)
        out.create_dataset(MERGERS.ONTOP_NEXT, data=ontop_next)
        out.create_dataset(MERGERS.ONTOP_PREV, data=ontop_prev)

    log.info("Renaming temporary file")
    log.debug("\t'{}' ==> '{}'".format(fname_temp, fname_out))
    shutil.move(fname_temp, fname_out)

    size_str = zio.get_file_size(fname_out)
    log.info("Saved {} mergers to '{}', size {}".format(num_mergers, fname_out, size_str))
    return fname_out


def _parse_merger_line(line):
    """
    Get target quantities from each line of the merger files.

    See 'http://www.illustris-project.org/w/index.php/Blackhole_Files' for
    dets regarding the illustris BH file structure.

    The format of each line is:
        "PROC-NUM  TIME  ID1  MASS1  ID2  MASS2"
        see: http://www.illustris-project.org/w/index.php/Blackhole_Files
        where
            '1' corresponds to the 'out'/'accretor'/surviving BH
            '2' corresponds to the 'in' /'accreted'/eliminated BH
        NOTE: that MASS1 is INCORRECT (dynamical mass, instead of BH)

    Returns
    -------
    time     : scalar, redshift of merger
    out_id   : long, id number of `out` BH
    out_mass : scalar, mass of `out` BH in simulation units (INCORRECT VALUE)
    in_id    : long, id number of `in` BH
    in_mass  : scalar, mass of `in` BH in simulation units

    """

    strs     = line.split()
    # Convert to proper types
    time     = DTYPE.SCALAR(strs[1])
    out_id   = DTYPE.ID(strs[2])
    out_mass = DTYPE.SCALAR(strs[3])
    in_id    = DTYPE.ID(strs[4])
    in_mass  = DTYPE.SCALAR(strs[5])

    return time, out_id, out_mass, in_id, in_mass


if __name__ == "__main__":
    main(reorganize_flag=True, crosscheck_flag=False)
