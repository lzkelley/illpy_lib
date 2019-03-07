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


from illpy_lib.illbh import (
    Core, load_hdf5_to_mem,
    MERGERS, BH_TYPE, MERGERS_PHYSICAL_KEYS, BH_TREE
)
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
    cosmo = illpy_lib.illcosmo.Illustris_Cosmology()
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


def load_tree(core=None, mrgs=None, recreate=None):
    """
    """

    core = Core.load(core)
    log = core.log
    log.debug("load_tree()")

    if recreate is None:
        recreate = core.sets.RECREATE

    fname_tree = core.paths.fname_merger_tree()

    exists = os.path.exists(fname_tree)
    log.debug("Merger tree file '{}' exists: {}".format(fname_tree, exists))

    if recreate or not exists:
        log.warning("Creating merger tree")

        if mrgs is None:
            mrgs = load_fixed_mergers(core=core)

        _construct_tree(core, mrgs, fname_tree)

    data = load_hdf5_to_mem(fname_tree)
    keys = list(data.keys())
    log.info("Loaded {} merger tree elements, keys: {}".format(
        data[BH_TREE.NUM], keys))

    return data


def _construct_tree(core, mrgs, fname_tree):
    """Use merger data to find and connect BHs which merge multiple times.

    Arguments
    ---------
        run     : <int>, Illlustris run number {1, 3}
        mrgs : <dict>, mergers dictionary
        verbose : <bool>, (optional=True), Print verbose output

    Returns
    -------
        tree : <dict>  container for tree data - see BHTree doc

    """
    log = core.log
    log.debug("_construct_tree()")

    # from . import tree
    import illpy_lib.illcosmo
    cosmo = illpy_lib.illcosmo.Illustris_Cosmology()

    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
    from . import tree

    NUM_BH_TYPES = len(BH_TYPE)

    num_mergers = mrgs[MERGERS.NUM]
    last_ind = -1 * np.ones([num_mergers, NUM_BH_TYPES], dtype=DTYPE.INDEX)
    next_ind = -1 * np.ones([num_mergers], dtype=DTYPE.INDEX)
    last_time = -1 * np.ones([num_mergers, NUM_BH_TYPES], dtype=DTYPE.SCALAR)
    next_time = -1 * np.ones([num_mergers], dtype=DTYPE.SCALAR)

    # Convert merger scale factors to ages
    scales = mrgs[MERGERS.SCALES]
    # times = np.array([cosmo.age(sc) for sc in scales], dtype=DTYPE.SCALAR)
    redz = cosmo._a_to_z(scales)
    times = cosmo.z_to_tage(redz)

    # Construct Merger Tree from node IDs
    log.info("Building BH Merger Tree")
    mids = mrgs[MERGERS.IDS]
    tree.build_tree(mids, times, last_ind, next_ind, last_time, next_time)

    num_bad = np.count_nonzero(last_ind < 0)
    log.info("{:d} Missing 'last_ind'".format(num_bad))

    num_bad = np.count_nonzero(next_ind < 0)
    log.info("{:d} Missing 'next_ind'".format(num_bad))

    # Create dictionary to store data
    tree = {
        BH_TREE.LAST: last_ind,
        BH_TREE.NEXT: next_ind,
        BH_TREE.LAST_TIME: last_time,
        BH_TREE.NEXT_TIME: next_time,

        BH_TREE.RUN: core.sets.RUN_NUM,
        BH_TREE.NUM: num_mergers,
    }

    log.warning("Saving merger-tree to '{}'".format(fname_tree))

    _save_hdf5(core, fname_tree, tree, backup=True)

    return tree


def _save_hdf5(core, fname_out, data, backup=True):
    log = core.log
    log.debug("_save_hdf5()")

    exists = os.path.exists(fname_out)
    if backup and exists:
        fname_back = zio.modify_filename(fname_out, prepend='_backup_')
        log.info("File exists, moving to backup: '{}' ==> '{}'".format(fname_out, fname_back))
        shutil.move(fname_out, fname_back)

    fname_temp = zio.modify_filename(fname_out, prepend='_')
    log.info("Writing to temporary file '{}'".format(fname_temp))

    with h5py.File(fname_temp, 'w') as out:
        out['created'] = str(datetime.now().ctime())
        out['version'] = VERSION
        out['filename'] = fname_out

        for key in data.keys():
            out.create_dataset(key, data=data[key])

    log.info("Renaming temporary file")
    log.debug("\t'{}' ==> '{}'".format(fname_temp, fname_out))
    shutil.move(fname_temp, fname_out)

    size_str = zio.get_file_size(fname_out)
    log.info("Saved to '{}', size {}".format(fname_out, size_str))
    return


'''

def analyzeTree(tree, verbose=True):
    """Analyze the merger tree data to obtain typical number of repeats, etc.

    Arguments
    ---------
        tree : <dict> container for tree data - see BHTree doc
        verbose : <bool>, Print verbose output

    Returns
    -------


    """

    if verbose: print(" - - BHTree.analyzeTree()")

    last         = tree[BH_TREE.LAST]
    next         = tree[BH_TREE.NEXT]
    timeNext     = tree[BH_TREE.NEXT_TIME]
    num_mergers   = len(next)

    aveFuture    = 0.0
    avePast      = 0.0
    aveFutureNum = 0
    avePastNum   = 0
    numPast      = np.zeros(num_mergers, dtype=int)
    numFuture    = np.zeros(num_mergers, dtype=int)

    if verbose: print((" - - - {:d} Mergers".format(num_mergers)))

    # Find number of unique merger BHs (i.e. no previous mrgs)
    inds = np.where((last[:, BH_TYPE.IN] < 0) & (last[:, BH_TYPE.OUT] < 0) & (next[:] < 0))
    numTwoIsolated = len(inds[0])
    # Find those with one or the other
    inds = np.where(((last[:, BH_TYPE.IN] < 0) ^ (last[:, BH_TYPE.OUT] < 0)) & (next[:] < 0))
    numOneIsolated = len(inds[0])

    if verbose:
        print((" - - - Mergers with neither  BH previously merged = {:d}".format(numTwoIsolated)))
        print((" - - - Mergers with only one BH previously merged = {:d}".format(numOneIsolated)))

    for ii in range(num_mergers):
        # Count Forward from First Mergers #
        #    If this is a first merger
        if all(last[ii, :] < 0):
            # Count the number of mrgs that the 'out' BH  from this merger, will later be in
            numFuture[ii] = _countFutureMergers(next, ii)
            # Accumulate for averaging
            aveFuture += numFuture[ii]
            aveFutureNum += 1

        # Count Backward from Last Mergers #
        #    If this is a final merger
        if next[ii] < 0:
            # Count the number of mrgs along the longest branch of past merger tree
            numPast[ii] = _countPastMergers(last, ii)
            # Accumulate for averaging
            avePast += numPast[ii]
            avePastNum += 1

    # Calculate averages
    if avePastNum   > 0:
        avePast /= avePastNum
    if aveFutureNum > 0:
        aveFuture /= aveFutureNum

    inds = np.where(next >= 0)[0]
    numRepeats = len(inds)
    fracRepeats = 1.0*numRepeats/num_mergers

    indsInt = np.where(timeNext >= 0.0)[0]
    numInts = len(indsInt)
    timeStats = np.average(timeNext[indsInt]), np.std(timeNext[indsInt])
    inds = np.where(timeNext == 0.0)[0]
    numZeroInts = len(inds)

    if verbose:
        print((" - - - Repeated mergers = {:d}/{:d} = {:.4f}".format(
            numRepeats, num_mergers, fracRepeats)))
        print((" - - - Average number past, future  =  {:.3f}, {:.3f}".format(avePast, aveFuture)))
        print((" - - - Number of merger intervals    = {:d}".format(numInts)))
        print((" - - - - Time between = {:.4e} +- {:.4e} [Myr]".format(
            timeStats[0]/MYR, timeStats[1]/MYR)))
        print((" - - - Number of zero time intervals = {:d}".format(numZeroInts)))

    timeBetween = timeNext[indsInt]

    # Store data to tree dictionary
    tree[BH_TREE.NUM_PAST] = numPast
    tree[BH_TREE.NUM_FUTURE] = numFuture
    tree[BH_TREE.TIME_BETWEEN] = timeBetween

    return timeBetween, numPast, numFuture


def allIDsForTree(run, mrg, tree=None, mrgs=None):
    """Get all of the ID numbers for BH in the same merger-tree as the given merger.

    Arguments
    ---------
    run : int
        Illustris simulation run number {1,3}.
    mrg : int
        Index of the target BH merger.  Any merger number in the same tree will yield the same
        results.
    tree : dict or `None`
        BHTree object will merger-tree data.  Loaded if not provided.
    mrgs : dict or `None`
        mergers object will merger data.  Loaded if not provided.

    Returns
    -------
    fin : int
        Index of the final merger this bh-tree participates in.  Acts as a unique identifier.
    allIDs : list of int
        List of all ID numbers of BHs which participate in this merger tree.

    """
    if not tree:
        tree = loadTree(run)

    nextMerg = tree[BH_TREE.NEXT]
    lastMerg = tree[BH_TREE.LAST]

    if not mrgs:
        from illpy_lib.illbh import mergers
        mrgs = mergers.load_fixed_mergers(run)

    m_ids = mrgs[MERGERS.IDS]

    # Go to the last merger
    fin = mrg
    while nextMerg[fin] >= 0:
        fin = nextMerg[fin]

    # Go backwards to get all IDs
    allIDs, mrgInds = _getPastIDs(m_ids, lastMerg, fin)
    return fin, allIDs, mrgInds


def _countFutureMergers(next, ind):
    """Use the map of `next` mrgs and a starting index to count the future number of mrgs.
    """
    count = 0
    ii = ind
    while next[ii] >= 0:
        count += 1
        ii = next[ii]
    return count


def _countPastMergers(last, ind):
    """Use the map of `last` mrgs and a starting index to count the past number of mrgs.
    """
    last_in  = last[ind, BH_TYPE.IN]
    last_out = last[ind, BH_TYPE.OUT]
    num_in   = 0
    num_out  = 0
    if last_in >= 0:
        num_in = _countPastMergers(last, last_in)
    if last_out >= 0:
        num_out = _countPastMergers(last, last_out)
    return np.max([num_in, num_out])+1


def _getPastIDs(m_ids, lastMerg, ind, idlist=[], mrglist=[]):
    """Get all BH IDs in past-mrgs of this BHTree.

    Arguments
    ---------
    m_ids : (N,2) array of int
        Merger BH ID numbers.
    last : (N,2) array of int
        For a given merger, give the index of the merger for each of the constituent BHs.
        `-1` if there was no previous merger.
    ind : int
        Index of merger to follow.
    idlist : list of int
        Existing list of merger IDs to append to.  Uses a `set` type intermediate to assure unique
        values.

    Used by: `allIDsForTree`.
    """
    ids_in = [m_ids[ind, BH_TYPE.IN]]
    ids_out = [m_ids[ind, BH_TYPE.OUT]]
    mrg_in = [ind]
    mrg_out = [ind]
    last_in  = lastMerg[ind, BH_TYPE.IN]
    last_out = lastMerg[ind, BH_TYPE.OUT]
    if last_in >= 0:
        ids_in, mrg_in = _getPastIDs(m_ids, lastMerg, last_in, ids_in, mrg_in)
    if last_out >= 0:
        ids_out, mrg_out = _getPastIDs(m_ids, lastMerg, last_out, ids_out, mrg_out)
    return list(set(ids_in + ids_out + idlist)), list(set(mrg_in + mrg_out + mrglist))
'''


if __name__ == "__main__":
    main(reorganize_flag=True, crosscheck_flag=False)
