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


try:
    import illpy_lib
except ImportError:
    PATH_ILLPY_LIB = "/n/home00/lkelley/illustris/redesign/illpy_lib/"
    if PATH_ILLPY_LIB not in sys.path:
        print("Added path to `illpy_lib`: '{}'".format(PATH_ILLPY_LIB))
        sys.path.append(PATH_ILLPY_LIB)

    import illpy_lib  # noqa


from illpy_lib.illbh.bh_constants import MERGERS, BH_TYPE, load_hdf5_to_mem
from illpy_lib.illbh import Core
from illpy_lib.constants import DTYPE


# VERSION_MAP = 0.21
# VERSION_FIX = 0.31
VERSION = 0.4


def main(reorganize_flag=True, crosscheck_flag=True):

    core = Core(sets=dict(LOG_FILENAME='log_illbh-mergers.log'))
    log = core.log
    log.info("mergers.main()")
    print(log.filename)

    log.debug("`reorganize_flag` = {}".format(reorganize_flag))
    if reorganize_flag:
        reorganize(core=core)

    log.debug("`crosscheck_flag` = {}".format(crosscheck_flag))
    if crosscheck_flag:
        crosscheck(core=core)

    return


def reorganize(core=None, recreate=None):

    core = Core()
    log = core.log
    log.info("mergers.reorganize()")

    if recreate is None:
        recreate = core.sets.RECREATE

    loadsave = (not recreate)
    fname_temp = core.paths.fname_mergers_temp()
    exists = os.path.exists(fname_temp)
    log.debug("Temporary merger file '{}' exists: {}".format(fname_temp, exists))

    if loadsave and exists:
        log.info("Temporary merger file exists.")
        return fname_temp

    log.warning("Reoganizing merger files to temporary file '{}'".format(fname_temp))
    _reorganize_files(core, fname_temp)

    return fname_temp


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

    # Create Mapping Between Mergers and Snapshots
    # mapM2S, mapS2M, ontop = _map_to_snapshots(scales)

    # Find the snapshot that each merger directly precedes
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


def crosscheck(core=None, recreate=None):   # run, mrgs, verbose=True):
    """
    """
    core = Core.load(core)
    log = core.log
    log.info("mergers.crosscheck()")

    if recreate is None:
        recreate = core.sets.RECREATE

    loadsave = (not recreate)
    fname_out = core.paths.fname_mergers_fixed()
    exists = os.path.exists(fname_out)

    log.debug("Fixed merger file '{}' exists: {}".format(fname_out, exists))

    if loadsave and exists:
        log.info("Fixed merger file exists.")
        return fname_out

    log.warning("Crosschecking mergers")
    _crosscheck_mergers(core, fname_out)

    return fname_out


def load_temp_mergers(core=None):
    core = Core.load(core)
    log = core.log
    log.debug("mergers.load_temp_mergers()")

    fname_temp = core.paths.fname_mergers_temp()
    if not os.path.exists(fname_temp):
        log.error("ERROR: File '{}' does not exist!".format(fname_temp))
        return None

    data = load_hdf5_to_mem(fname_temp)
    log.info("Loaded {} temporary mergers, keys: {}".format(
        data[MERGERS.SCALES].size, list(data.keys())))
    return data


def load_fixed_mergers(core=None):
    core = Core.load(core)
    log = core.log
    log.debug("mergers.load_fixed_mergers()")

    fname_fixed = core.paths.fname_mergers_fixed()
    if not os.path.exists(fname_fixed):
        log.error("File '{}' does not exist!".format(fname_fixed))
        return None

    data = load_hdf5_to_mem(fname_fixed)
    log.info("Loaded {} fixed mergers, keys: {}".format(
        data[MERGERS.SCALES].size, list(data.keys())))
    return data


def _crosscheck_mergers(core, fname_out):
    log = core.log
    log.info("mergers._crosscheck_mergers()")

    from illpy_lib.illbh import BHMatcher

    # Make copy to modify
    # mrgs_fixed = dict(mrgs)
    fname_temp = core.paths.fname_mergers_temp()
    with h5py.File(fname_temp, 'r') as data:
        scales = data[MERGERS.SCALES][:]
        mids = data[MERGERS.IDS][:]
        masses = data[MERGERS.MASSES][:]

    num_mergers_temp = scales.size
    log.info("Loaded {} raw mergers".format(num_mergers_temp))

    # Remove entries where IDs match a second time (IS THIS ENOUGH?!)

    # First sort by ``BH_TYPE.IN`` then ``BH_TYPE.OUT`` (reverse of given order)
    sort = np.lexsort((mids[:, BH_TYPE.OUT], mids[:, BH_TYPE.IN]))

    # bad_inds = []
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

    # Change number, creation date, and version

    # Fix Merger 'Out' Masses
    #  =======================
    masses = mrgs_fixed[MERGERS.MASSES]
    aveBef = np.average(masses[:, BH_TYPE.OUT])
    massOut = BHMatcher.inferMergerOutMasses(run, mrgs=mrgs_fixed, verbose=verbose)
    masses[:, BH_TYPE.OUT] = massOut
    aveAft = np.average(masses[:, BH_TYPE.OUT])
    if verbose: print((" - - - - Ave mass:  {:.4e} ===> {:.4e}".format(aveBef, aveAft)))

    return mrgs_fixed


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


'''
def _import_raw_mergers(files, verbose=True):
    """
    Fill the given arrays with merger data from the given target files.

    Arrays ``ids`` and ``masses`` are shaped [N, 2], for ``N`` total mrgs.
    By convention the
        'in'  (accreted) BH is index ``BH_TYPE.IN`` (0?)
        'out' (accretor) BH is index ``BH_TYPE.OUT`` (1?)

    Arguments
    ---------

    """

    if verbose: print(" - - mergers._import_raw_mergers()")

    # Make sure argument is a list
    # if (not aux.iterableNotString(files)): files = [files]

    # Count Mergers and Prepare Storage for Data
    # ------------------------------------------
    numLines = zio.countLines(files)
    if verbose: print((" - - - Lines : {:d}".format(numLines)))

    # Initialize Storage
    scales = np.zeros(numLines,               dtype=DTYPE.SCALAR)
    ids    = np.zeros([numLines, NUM_BH_TYPES], dtype=DTYPE.ID)
    masses = np.zeros([numLines, NUM_BH_TYPES], dtype=DTYPE.SCALAR)

    # Load Lines from Files
    # ---------------------
    if verbose: pbar = zio.getProgressBar(numLines)
    count = 0
    for fil in files:
        for line in open(fil):
            # Get target elements from each line of file
            time, out_id, out_mass, in_id, in_mass = _parse_merger_line(line)
            # Store values
            scales[count] = time
            ids[count, BH_TYPE.IN] = in_id
            ids[count, BH_TYPE.OUT] = out_id
            masses[count, BH_TYPE.IN] = in_mass
            masses[count, BH_TYPE.OUT] = out_mass
            # Increment Counter
            count += 1

            # Print Progress
            if verbose: pbar.update(count)

    if verbose: pbar.finish()

    return scales, ids, masses


def load_fixed_mergers(run, verbose=True, loadsave=True):
    """
    Load BH Merger data with duplicats removes, and masses corrected.

    Arguments
    ---------
       run      <int>  : illustris simulation run number {1, 3}
       verbose  <bool> : optional, print verbose output
       loadsave <bool> : optional, load existing save file (recreate if `False`)

    Returns
    -------
       mergersFixed <dict> : dictionary of 'fixed' mrgs, most entries shaped [N, 2] for `N`
                             mrgs, and an entry for each {``BH_TYPE.IN``, ``BH_TYPE.OUT``}

    """

    if verbose: print(" - - mergers.load_fixed_mergers()")

    fixedFilename = GET_MERGERS_FIXED_FILENAME(run, VERSION_FIX)

    # Try to Load Existing Mapped Mergers
    if loadsave:
        if verbose: print((" - - - Loading from save '{:s}'".format(fixedFilename)))
        if os.path.exists(fixedFilename):
            mergersFixed = zio.npzToDict(fixedFilename)
        else:
            print((" - - - - '{:s}' does not exist.  Recreating.".format(fixedFilename)))
            loadsave = False

    # Recreate Fixed Mergers
    if (not loadsave):
        if verbose: print(" - - - Creating Fixed Mergers")

        # Load Mapped Mergers
        mergersMapped = load_mapped_mergers(run, verbose=verbose)
        # Fix Mergers
        mergersFixed = _fix_mergers(run, mergersMapped, verbose=verbose)
        # Save
        zio.dictToNPZ(mergersFixed, fixedFilename, verbose=verbose)

    return mergersFixed


def _map_to_snapshots(scales, verbose=True):
    """
    Find the snapshot during which, or following each merger
    """

    if verbose:
        print(" - - mergers._map_to_snapshots()")

    numMergers = len(scales)

    # Load Cosmology
    import illpy_lib.illcosmo
    cosmo = illpy_lib.illcosmo.cosmology.Cosmology()
    snapScales = cosmo.scales()

    # Map Mergers-2-Snapshots: snapshot before (or ontop) of each merger
    mapM2S = np.zeros(numMergers, dtype=DTYPE.INDEX)
    # Map Snapshots-2-Mergers: list of mrgs just-after (or ontop) of each snapshot
    mapS2M = [[] for ii in range(cosmo.num)]
    # Flags if merger happens exactly on a snapshot (init to False=0)
    ontop  = np.zeros(numMergers, dtype=bool)

    # Find snapshots on each side of merger time #

    # Find the snapshot just below and above each merger.
    #     each entry (returned) is [low, high, dist-low, dist-high]
    #     low==high if the times match (within function's default uncertainty)
    snapBins = [_findBoundingBins(sc, snapScales) for sc in scales]

    # Create Mappings
    # ---------------

    if verbose:
        print(" - - - Creating mappings")
        pbar = zio.getProgressBar(numMergers)

    nums = len(snapBins)
    for ii, bins in enumerate(snapBins):
        tsnap = bins[1]     # Set snapshot to upper bin
        mapM2S[ii] = tsnap  # Set snapshot for this merger
        mapS2M[tsnap].append(ii)  # Add merger to this snapshot
        # If this merger takes place ontop of snapshot, set flag
        if (bins[0] == bins[1]):
            ontop[ii] = True

        # Print Progress
        if verbose:
            pbar.update(ii)

    if verbose:
        pbar.finish()

    # Find the most mrgs in a snapshot
    numPerSnap = np.array([len(s2m) for s2m in mapS2M])
    mostMergers = np.max(numPerSnap)
    mostIndex = np.where(mostMergers == numPerSnap)[0]
    # Find the number of ontop mrgs
    numOntop = np.count_nonzero(ontop)
    if verbose:
        print((" - - - Snapshot {:d} with the most ({:d}) mrgs".format(mostIndex, mostMergers)))
    if verbose:
        print((" - - - {:d} ({:.2f}) ontop mrgs".format(numOntop, 1.0*numOntop/nums)))

    return mapM2S, mapS2M, ontop


def _findBoundingBins(target, bins, thresh=1.0e-5):
    """
    Find the array indices (of "bins") bounding the "target"

    If target is outside bins, the missing bound will be 'None'
    low and high will be the same, if the target is almost exactly[*1] equal to a bin

    [*1] : How close counds as effectively the same is set by 'DEL_TIME_THRESH' below

    arguments
    ---------
        target  : [] value to be compared
        bins    : [] list of values to compare to the 'target'

    output
    ------
        low  : [int] index below target (or None if none)
        high : [int] index above target (or None if none)

    """

    # deltat  : test whether the fractional difference between two values is less than threshold
    #           This function allows the later conditions to accomodate smaller numerical
    #           differences, between effectively the same value  (e.g.   1.0 vs. 0.9999999999989)
    #
    if (thresh == 0.0):
        deltat = lambda x, y: False
    else:
        deltat = lambda x, y: np.abs(x-y)/np.abs(x) <= thresh

    # nums = len(bins)
    # Find bin above (or equal to) target
    high = np.where((target <= bins) | deltat(target, bins))[0]
    if (len(high) == 0):
        high = None
    # Select first bin above target
    else:
        high = high[0]
        dhi  = bins[high] - target

    # Find bin below (or equal to) target
    low  = np.where((target >= bins) | deltat(target, bins))[0]
    if (len(low)  == 0): low  = None
    # Select  last bin below target
    else:
        low  = low[-1]
        dlo  = bins[low] - target

    # Print warning on error
    if (low is None) or (high is None):
        print("mergers._findBoundingBins: target = {:e}, bins = {{:e}, {:e}}; low, high = {:s}, {:s} !".format(target, bins[0], bins[-1], str(low), str(high)))
        raise RuntimeError("Could not find bins!")

    return [low, high, dlo, dhi]
'''


if __name__ == "__main__":
    main(reorganize_flag=True, crosscheck_flag=False)
