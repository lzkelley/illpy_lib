"""Routines to match BH Mergers with BH Details entries based on times and IDs.

Functions
---------
    main
    loadMergerDetails
    detailsForMergers
    checkMatches
    inferMergerOutMasses

MergerDetails Dictionary
------------------------
   The ``MergerDetails`` dictionary contains all of the same parameters as that of the normal
   ``BHDetails`` dictionary, but with more elements.  In particular, every parameter entry in the
   dictionary has shape `[N, 2, 3]`, where each axis is as follows:
      0: which merger, for ``N`` total mergers
      1: which BH, either ``BH_TYPE.IN`` or ``BH_TYPE.OUT``
      2: time of details entry, one of
         ``BH_TIME.FIRST``  the first details entry matching this BH
         ``BH_TIME.BEFORE`` details entry immediately before the merger
         ``BH_TIME.AFTER``  details entry immediately after  the merger (only for ``BH_TYPE.OUT``)

   { DETAILS.RUN       : <int>, illustris simulation number in {1, 3}
     DETAILS.NUM       : <int>, total number of mergers `N`
     DETAILS.FILE      : <str>, name of save file from which mergers were loaded/saved
     DETAILS.CREATED   : <str>, datetime this file was created
     DETAILS.VERSION   : <flt>, version of BHDetails used to create file

     DETAILS.IDS       : <uint64>[N, 2, 3], BH particle ID numbers for each entry
     DETAILS.SCALES    : <flt64> [N, 2, 3], scale factor at which each entry was written
     DETAILS.MASSES    : <flt64> [N, 2, 3], BH mass
     DETAILS.MDOTS     : <flt64> [N, 2, 3], BH Mdot
     DETAILS.RHOS      : <flt64> [N, 2, 3], ambient mass-density
     DETAILS.CS        : <flt64> [N, 2, 3], ambient sound-speed
   }

"""
from __future__ import absolute_import, division, print_function, unicode_literals

# import os
import numpy as np
from datetime import datetime

from mpi4py import MPI

import illpy.illbh.BHDetails
import illpy.illbh.BHMergers
import illpy.illbh.BHConstants
from illpy.illbh.BHConstants import MERGERS, DETAILS, _LOG_DIR, GET_MERGER_DETAILS_FILENAME
# from illpy.illbh.BHConstants import MERGERS, DETAILS, BH_TYPE, BH_TIME, NUM_BH_TYPES, \
#     NUM_BH_TIMES, _LOG_DIR, GET_MERGER_DETAILS_FILENAME, DETAILS_PHYSICAL_KEYS
# from illpy.illbh.MatchDetails import getDetailIndicesForMergers
import illpy.Constants
from illpy.Constants import DTYPE, NUM_SNAPS

import zcode.inout as zio
import zcode.math as zmath

__version__ = '0.23'
_GET_KEYS = [DETAILS.SCALES, DETAILS.MASSES, DETAILS.MDOTS, DETAILS.RHOS, DETAILS.CS]
_MAX_DETAILS_PER_SNAP = 10


def main(run=1, verbose=True, debug=True):
    # Initialization
    # --------------
    #     MPI Parameters
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    beg_all = datetime.now()
    name = __file__
    header = "\n%s\n%s\n%s" % (name, '='*len(name), str(beg_all))

    if(rank == 0):
        zio.checkPath(_LOG_DIR)

    comm.Barrier()

    # Initialize log
    log = illpy.illbh.BHConstants._loadLogger(
        __file__, debug=debug, verbose=verbose, run=run, rank=rank, version=__version__)
    log.debug(header)

    bhIDsUnique = None
    if(rank == 0):
        print("Log filename = ", log.filename)

        # Load Mergers
        log.debug("Loading Mergers")
        mergers = illpy.illbh.BHMergers.loadFixedMergers(run)
        numMergers = mergers[MERGERS.NUM]
        mergerIDs = mergers[MERGERS.IDS]

        # Find unique BH IDs from set of merger BHs
        #    these are now 1D and sorted
        bhIDsUnique, reconInds = np.unique(mergerIDs, return_inverse=True)
        numUnique = np.size(bhIDsUnique)
        numTotal = np.size(reconInds)
        frac = numUnique/numTotal
        log.debug(" - %d/%d = %.4f Unique BH IDs" % (numUnique, numTotal, frac))

    # Send unique IDs to all processors
    bhIDsUnique = comm.bcast(bhIDsUnique, root=0)

    # Distribute snapshots to each processor
    mySnaps = np.arange(NUM_SNAPS)
    if(size > 1):
        # Randomize which snapshots go to which processor for load-balancing
        mySnaps = np.random.permutation(mySnaps)
        # Make sure all ranks are synchronized on initial (randomized) list before splitting
        mySnaps = comm.bcast(mySnaps, root=0)
        mySnaps = np.array_split(mySnaps, size)[rank]

    log.info("Rank {:d}/{:d} with {:d} Snapshots [{:d} ... {:d}]".format(
        rank, size, mySnaps.size, mySnaps.min(), mySnaps.max()))

    # Get details entries for unique merger IDs in snapshot list
    # ----------------------------------------------------------
    nums, scales, masses, mdots, dens, csnds, ids = \
        _detailsForMergers_snapshots(run, mySnaps, bhIDsUnique, log)

    # Collect results and organize
    # ----------------------------
    if(size > 1):
        log.debug(" - Gathering")
        beg = datetime.now()
        # Gather results from each processor into ``rank=0``
        tempScales = comm.gather(scales, root=0)
        tempMasses = comm.gather(masses, root=0)
        tempMdots = comm.gather(mdots, root=0)
        tempDens = comm.gather(dens, root=0)
        tempCsnds = comm.gather(csnds, root=0)
        tempIds = comm.gather(ids, root=0)
        tempNums = comm.gather(nums, root=0)
        end = datetime.now()
        log.debug(" - - Done after %s" % (str(end-beg)))

        # Gather snapshot numbers for ordering
        mySnaps = comm.gather(mySnaps, root=0)

        # Organize results appropriately
        if(rank == 0):
            log.debug(" - Stacking")
            beg = datetime.now()

            nums = np.zeros(numUnique)
            scales = numUnique*[[None]]
            masses = numUnique*[[None]]
            mdots = numUnique*[[None]]
            dens = numUnique*[[None]]
            csnds = numUnique*[[None]]
            ids = numUnique*[[None]]

            # Iterate over each black-hole and processor, collect results into single arrays
            for ii, mm in enumerate(bhIDsUnique):
                for jj in xrange(size):
                    errStr = ""
                    if(tempIds[jj][ii][0] is not None):
                        dd = tempIds[jj][ii][0]
                        # Make sure all of the details IDs are consistent
                        if(np.any(tempIds[jj][ii] != dd)):
                            errStr += "ii = {}, jj = {}, mm = {}; tempIds[0] = {}".format(ii, jj, mm, dd)
                            errStr += " tempIds = {}".format(str(tempIds[ii]))

                        # Make sure details IDs match expected merger ID
                        if(dd != mm):
                            errStr += "\nii = {}, jj = {}, mm = {}; dd = {}".format(ii, jj, mm, dd)

                        # If no entries have been stored yet, replace with first entries
                        if(ids[ii][0] is None):
                            ids[ii] = tempIds[jj][ii]
                            scales[ii] = tempScales[jj][ii]
                            masses[ii] = tempMasses[jj][ii]
                            mdots[ii] = tempMdots[jj][ii]
                            dens[ii] = tempDens[jj][ii]
                            csnds[ii] = tempCsnds[jj][ii]
                        # If entries already exist, append new ones
                        else:
                            # Double check that all existing IDs are consistent with new ones
                            #    This should be redundant, but whatevs
                            if(np.any(ids[ii] != dd)):
                                errStr += "\nii = {}, jj = {}, mm = {}, dd = {}, ids = {}".format(ii, jj, mm, dd, str(ids))
                            ids[ii] = np.append(ids[ii], tempIds[jj][ii])
                            scales[ii] = np.append(scales[ii], tempScales[jj][ii])
                            masses[ii] = np.append(masses[ii], tempMasses[jj][ii])
                            mdots[ii] = np.append(mdots[ii], tempMdots[jj][ii])
                            dens[ii] = np.append(dens[ii], tempDens[jj][ii])
                            csnds[ii] = np.append(csnds[ii], tempCsnds[jj][ii])

                        # Count the number of entries for each BH from each processor
                        nums[ii] += tempNums[jj][ii]

                    if(len(errStr) > 0):
                        log.error(errStr)
                        zio.mpiError(comm, log=log, err=errStr)

            # Merge lists of snapshots, and look for any missing
            mySnaps = np.hstack(mySnaps)
            log.debug("Obtained %d Snapshots" % (mySnaps.size))
            missingSnaps = []
            for ii in xrange(NUM_SNAPS):
                if(ii not in mySnaps):
                    missingSnaps.append(ii)

            if(len(missingSnaps) > 0):
                log.warning("WARNING: snaps %s not in results!" % (str(missingSnaps)))

            log.debug("Total entries stored = %d" % (np.sum([np.sum(nn) for nn in nums])))

    # Convert from uinique BH IDs back to full mergers list.  Sort results by time (scalefactor)
    if(rank == 0):
        log.debug(" - Reconstructing")
        beg = datetime.now()
        ids = np.array([ids[ii] for ii in reconInds]).reshape(numMergers, 2)
        scales = np.array([scales[ii] for ii in reconInds]).reshape(numMergers, 2)
        masses = np.array([masses[ii] for ii in reconInds]).reshape(numMergers, 2)
        dens = np.array([dens[ii] for ii in reconInds]).reshape(numMergers, 2)
        mdots = np.array([mdots[ii] for ii in reconInds]).reshape(numMergers, 2)
        csnds = np.array([csnds[ii] for ii in reconInds]).reshape(numMergers, 2)
        nums = np.array([nums[ii] for ii in reconInds]).reshape(numMergers, 2)
        end = datetime.now()
        log.debug(" - - Done after %s" % (str(end-beg)))

        # Sort entries for each BH by scalefactor
        log.debug(" - Sorting")
        beg = datetime.now()
        for ii in xrange(numMergers):
            for jj in xrange(2):
                if(nums[ii, jj] == 0): continue
                # Check ID numbers yet again
                if(not np.all(ids[ii, jj] == mergerIDs[ii, jj])):
                    errStr = "Error!  ii = {}, jj = {}.  Merger ID = {}, det ID = {}"
                    errStr = errStr.format(ii, jj, mergerIDs[ii, jj], ids[ii, jj])
                    errStr += "nums[ii,jj] = {:s}, shape {:s}".format(str(nums[ii, jj]), str(np.shape(nums[ii, jj])))
                    log.error(errStr)
                    zio.mpiError(comm, log=log, err=errStr)

                inds = np.argsort(scales[ii, jj])
                scales[ii, jj] = scales[ii, jj][inds]
                masses[ii, jj] = masses[ii, jj][inds]
                dens[ii, jj] = dens[ii, jj][inds]
                mdots[ii, jj] = mdots[ii, jj][inds]
                csnds[ii, jj] = csnds[ii, jj][inds]

        end = datetime.now()
        log.debug(" - - Done after %s" % (str(end-beg)))

        log.debug(" - Packaging")
        beg = datetime.now()
        data = {DETAILS.IDS: ids,
                DETAILS.SCALES: scales,
                DETAILS.MASSES: masses,
                DETAILS.RHOS: dens,
                DETAILS.MDOTS: mdots,
                DETAILS.CS: csnds,
                'nums': nums}

        end = datetime.now()
        log.debug(" - - Done after %s" % (str(end-beg)))

        log.info(" - Saving to dictionary.")
        beg = datetime.now()
        fname = 'temp.npz'
        zio.dictToNPZ(data, fname, verbose=True)
        end = datetime.now()
        log.info(" - - Saved to '%s' after %s" % (fname, str(end-beg)))

        end_all = datetime.now()
        log.debug(" - Done after %s" % (str(end_all - beg_all)))

    return


def _detailsForMergers_snapshots(run, snapshots, bhIDsUnique, log):
    """Find details entries for mergers in particular snapshots.

    For each snapshot, store at most `_MAX_DETAILS_PER_SNAP` entries for each blackhole,
    interpolating values to an even spacing in scale-factor if more entries are found.  The first
    and last entries found are stored as is.

    Arguments
    ---------

    Returns
    -------

    """
    log.debug("_detailsForMergers_snapshot()")

    comm = MPI.COMM_WORLD

    # Initialize
    # ----------
    #     Make sure snapshots are iterable
    snapshots = np.atleast_1d(snapshots)
    numUniqueBHs = bhIDsUnique.size

    numMatchesTotal = np.zeros(numUniqueBHs, dtype=int)   # Number of matches in all snapshots
    numStoredTotal = np.zeros(numUniqueBHs, dtype=int)    # Number of entries stored in all

    #     Empty list for each Unique BH
    ids = numUniqueBHs * [[None]]
    scales = numUniqueBHs * [[None]]
    masses = numUniqueBHs * [[None]]
    mdots = numUniqueBHs * [[None]]
    csnds = numUniqueBHs * [[None]]
    dens = numUniqueBHs * [[None]]

    # Iterate over target snapshots
    # -----------------------------
    for snap in snapshots:
        log.debug("snap = %03d" % (snap))
        numMatchesSnap = np.zeros(numUniqueBHs, dtype=int)    # Num matches in a given snapshot
        numStoredSnap = np.zeros(numUniqueBHs, dtype=int)     # Num entries stored in a snapshot

        # Load `BHDetails`
        dets = illpy.illbh.BHDetails.loadBHDetails(run, snap, verbose=False)
        numDets = dets[DETAILS.NUM]
        log.debug(" - %d Details" % (numDets))
        detIDs = dets[DETAILS.IDS]
        if(np.size(detIDs) == 0): continue

        if(not isinstance(detIDs[0], DTYPE.ID)):
            errStr = "Error: incorrect dtype = %s" % (np.dtype(detIDs[0]))
            log.error(errStr)
            raise RuntimeError(errStr)

        detScales = dets[DETAILS.SCALES]

        # Sort details entries by IDs then by scales (times)
        detSort = np.lexsort((detScales, detIDs))
        count = 0
        # Iterate over and search for each unique BH ID
        for ii, bh in enumerate(bhIDsUnique):
            tempMatches = []
            # Increment up until reaching the target BH ID
            while(count < numDets and detIDs[detSort[count]] < bh):
                count += 1

            # Iterate over all matching BH IDs, storing those details' indices
            while(count < numDets and detIDs[detSort[count]] == bh):
                tempMatches.append(detSort[count])
                count += 1

            # Store values at matching indices
            tempMatches = np.array(tempMatches)
            numMatchesSnap[ii] = tempMatches.size
            #     Only if there are some matches
            if(numMatchesSnap[ii] > 0):
                tempScales = detScales[tempMatches]
                tempMasses = dets[DETAILS.MASSES][tempMatches]
                tempMdots = dets[DETAILS.MDOTS][tempMatches]
                tempDens = dets[DETAILS.RHOS][tempMatches]
                tempCsnds = dets[DETAILS.CS][tempMatches]
                tempIDs = dets[DETAILS.IDS][tempMatches]

                # Interpolate down to only `_MAX_DETAILS_PER_SNAP` entries
                if(numMatchesSnap[ii] > _MAX_DETAILS_PER_SNAP):
                    #    Create even spacing in scale-factor to interpolate to
                    newScales = zmath.spacing(tempScales, scale='lin', num=_MAX_DETAILS_PER_SNAP)
                    tempMasses = np.interp(newScales, tempScales, tempMasses)
                    tempMdots = np.interp(newScales, tempScales, tempMdots)
                    tempDens = np.interp(newScales, tempScales, tempDens)
                    tempCsnds = np.interp(newScales, tempScales, tempCsnds)
                    #    Cant interpolate IDs, select random subset instead...
                    tempIDs = np.random.choice(tempIDs, size=_MAX_DETAILS_PER_SNAP, replace=False)
                    if(not isinstance(tempIDs[0], DTYPE.ID)):
                        errStr = "Error: incorrect dtype for random tempIDs = %s" % (np.dtype(tempIDs[0]))
                        log.error(errStr)
                        raise RuntimeError(errStr)

                    tempScales = newScales

                # Store matches
                #    If this is the first set of entries, replace ``[None]``
                if(scales[ii][0] is None):
                    scales[ii] = tempScales
                    masses[ii] = tempMasses
                    mdots[ii] = tempMdots
                    dens[ii] = tempDens
                    csnds[ii] = tempCsnds
                    ids[ii] = tempIDs
                #    If there are already entries, append new ones
                else:
                    if(tempIDs[0] != ids[ii][-1] or tempIDs[0] != bh):
                        errStr = "Snap {}, ii {}, bh = {}, prev IDs = {}, new = {}!!"
                        errStr.format(snap, ii, bh, ids[ii][-1], tempIDs[0])
                        log.error(errStr)
                        zio.mpiError(comm, log=log, err=errStr)

                    scales[ii] = np.append(scales[ii], tempScales)
                    masses[ii] = np.append(masses[ii], tempMasses)
                    mdots[ii] = np.append(mdots[ii], tempMdots)
                    dens[ii] = np.append(dens[ii], tempDens)
                    csnds[ii] = np.append(csnds[ii], tempCsnds)
                    ids[ii] = np.append(ids[ii], tempIDs)

                numStoredSnap[ii] += tempScales.size

            if(count >= numDets):
                break

        scales = np.array(scales)
        masses = np.array(masses)
        mdots = np.array(mdots)
        dens = np.array(dens)
        csnds = np.array(csnds)
        ids = np.array(ids)

        numMatchesTotal += numMatchesSnap
        numStoredTotal += numStoredSnap
        snapTotal = np.sum(numMatchesSnap)
        total = np.sum(numMatchesTotal)
        snapOcc = np.count_nonzero(numMatchesSnap)
        occ = np.count_nonzero(numMatchesTotal)
        log.debug(" - %7d Matches (%7d total), %4d BHs (%4d)" % (snapTotal, total, snapOcc, occ))
        log.debug(" - Average and total number stored = {:f}, {:d}".format(
            numStoredTotal.mean(), np.sum(numStoredTotal)))

    return numStoredTotal, scales, masses, mdots, dens, csnds, ids


'''
def loadMergerDetails(run, loadsave=True, mergers=None, verbose=True):

    saveFile = GET_MERGER_DETAILS_FILENAME(run, __version__)

    # Try to Load Existing Merger Details
    if(loadsave):
        # if(verbose): print " - - - Loading save from '%s'" % (saveFile)
        # Make sure file exists
        if(os.path.exists(saveFile)):
            mergerDetails = zio.npzToDict(saveFile)
            # if(verbose): print " - - - Loaded Merger Details"
        else:
            loadsave = False
            warnStr = "File '%s' does not exist!" % (saveFile)
            warnings.warn(warnStr, RuntimeWarning)

    # Re-match Mergers with Details
    if(not loadsave):
        # if(verbose): print " - - - Rematching mergers and details"
        if(mergers is None): mergers = illpy.illbh.BHMergers.loadFixedMergers(run)
        # if(verbose): print " - - - - Loaded %d Mergers" % (mergers[MERGERS.NUM])

        # Get Details for Mergers
        mergerDetails = detailsForMergers(run, mergers, verbose=verbose)
        # Add meta-data
        mergerDetails[DETAILS.RUN]     = run
        mergerDetails[DETAILS.CREATED] = datetime.now().ctime()
        mergerDetails[DETAILS.VERSION] = __version__
        mergerDetails[DETAILS.FILE]    = saveFile

        # Save merger details
        zio.dictToNPZ(mergerDetails, saveFile, verbose=verbose)

    return mergerDetails


def detailsForMergers(run, mergers, log):
    """Given a set of mergers, retrieve corresponding 'details' entries for BHs.

    Finds the details entries which occur closest to the 'merger' time, both
    before and after it (after only exists for the 'out' BHs).

    Arguments
    ---------
        run     : int, illustris run number {1, 3}
        mergers : dict, data arrays for mergers
        verbose : bool (optional : ``True``), flag to print verbose output.

    Returns
    -------
        mergDets : dict, data arrays corresponding to each 'merger' BHs

    """
    log.debug("detailsForMergers()")

    numMergers = mergers[MERGERS.NUM]
    numSnaps = NUM_SNAPS
    shape = [numMergers, NUM_BH_TYPES, NUM_BH_TIMES]

    # Search for all Mergers in each snapshot (easier, though less efficient)
    targets = np.arange(numMergers)

    snapNumbers = reversed(xrange(numSnaps))

    matchInds  = -1*np.ones(shape, dtype=DTYPE.INDEX)
    matchTimes = -1*np.ones(shape, dtype=DTYPE.SCALAR)

    # Create Dictionary To Store Results
    mergDets = {}
    # Initialize Dictionary to Invalid entries (-1)
    for KEY in DETAILS_PHYSICAL_KEYS:
        if(KEY == DETAILS.IDS): mergDets[KEY] = np.zeros(shape, dtype=DTYPE.ID)
        else:                   mergDets[KEY] = -1*np.ones(shape, dtype=DTYPE.SCALAR)

    # Iterate over all Snapshots
    # --------------------------
    if(verbose): pbar = zio.getProgressBar(numSnaps)
    for ii, snum in enumerate(snapNumbers):
        # Load appropriate Details File
        dets = illpy.illbh.BHDetails.loadBHDetails(run, snum, verbose=False)
        # If there are no details here, continue to next iter, but save BHs
        if(dets[DETAILS.NUM] == 0): continue
        # Track which matches are 'new' (+1 = new); reset at each snapshot
        matchNew  = -1*np.ones([shape], dtype=int)  # THIS MIGHT NEED TO BE A LONG???? FIX

        # Load Details Information
        numMatches = getDetailIndicesForMergers(targets, mergers[MERGERS.IDS],
                                                mergers[MERGERS.SCALES], matchInds,
                                                matchTimes, matchNew,
                                                dets[DETAILS.IDS], dets[DETAILS.SCALES])

        # Store Matches Data

        # iterate over in/out BHs
        for TYPE in [BH_TYPE.IN, BH_TYPE.OUT]:
            # Iterate over match times
            for TIME in [BH_TIME.FIRST, BH_TIME.BEFORE, BH_TIME.AFTER]:
                # Find valid matches
                inds = np.where(matchNew[:, TYPE, TIME] >= 0)[0]
                if(len(inds) > 0):
                    # Store Each Parameter
                    for KEY in DETAILS_PHYSICAL_KEYS:
                        mergDets[KEY][inds, TYPE, TIME] = dets[KEY][matchInds[inds, TYPE, TIME]]

        if(verbose): pbar.update(ii)

    if(verbose): pbar.finish()

    return mergDets


def checkMatches(matches, mergers):
    """
    Perform basic diagnostic on merger-detail matches.

    Arguments
    ---------

    """

    RAND_NUMS = 4

    numMergers = mergers[MERGERS.NUM]
    bh_str = {BH_TYPE.IN: "In ", BH_TYPE.OUT: "Out"}
    # time_str = {BH_TIME.FIRST: "First ", BH_TIME.BEFORE: "Before", BH_TIME.AFTER: "After "}

    good = np.zeros([numMergers, len(BH_TYPE), len(BH_TIME)], dtype=bool)

    # Count Seemingly Successful Matches
    # print " - - - Number of Matches (%6d Mergers) :" % (numMergers)
    # print "             First  Before After"

    # Iterate over both BHs
    for TYPE in [BH_TYPE.IN, BH_TYPE.OUT]:
        num_str = ""
        # Iterate over match times
        for TIME in [BH_TIME.FIRST, BH_TIME.BEFORE, BH_TIME.AFTER]:
            inds = np.where(matches[DETAILS.SCALES][:, TYPE, TIME] >= 0.0)[0]
            good[inds, TYPE, TIME] = True
            num_str += "%5d  " % (len(inds))

        # print "       %s : %s" % (bh_str[TYPE], num_str)

    # Count Combinations of Matches
    # -----------------------------
    # print "\n - - - Number of Match Combinations :"

    # All Times
    inds = np.where(np.sum(good[:, BH_TYPE.OUT, :], axis=1) == len(BH_TIME))[0]
    # print " - - - - All            : %5d" % (len(inds))

    # Before and After
    inds = np.where(good[:, BH_TYPE.OUT, BH_TIME.BEFORE] &
                    good[:, BH_TYPE.OUT, BH_TIME.AFTER])[0]
    # print " - - - - Before & After : %5d" % (len(inds))

    # First and Before
    inds = np.where(good[:, BH_TYPE.OUT, BH_TIME.FIRST] &
                    good[:, BH_TYPE.OUT, BH_TIME.BEFORE])[0]
    # print " - - - - First & Before : %5d" % (len(inds))

    # print ""
    inds = np.where((good[:, BH_TYPE.OUT, BH_TIME.FIRST]  == True) &
                    (good[:, BH_TYPE.OUT, BH_TIME.BEFORE] == False))[0]

    # print " - - - Number of First without Before : %5d" % (len(inds))
    if(len(inds) > 0):
        # print "\t\t         First      Before    (Merger )   After"
        if(len(inds) <= RAND_NUMS): sel = np.arange(len(inds))
        else:                         sel = np.random.randint(0, len(inds), size=RAND_NUMS)

        for ii in sel:
            sel_id = inds[ii]
            tmat = matches[DETAILS.SCALES][sel_id, BH_TYPE.OUT, :]
            tt = mergers[MERGERS.SCALES][sel_id]
            # print "\t\t%5d : %+f  %+f  (%+f)  %+f" % \
            #     (sel_id, tmat[BH_TIME.FIRST], tmat[BH_TIME.BEFORE], tt, tmat[BH_TIME.AFTER])

    # Check ID Matches
    # print "\n - - - Number of BAD ID Matches:"
    # print "             First  Before After"
    # Iterate over both BHs
    for BH in [BH_TYPE.IN, BH_TYPE.OUT]:
        num_str = ""
        eg = None
        # Iterate over match times
        for FBA in [BH_TIME.FIRST, BH_TIME.BEFORE, BH_TIME.AFTER]:
            dids = matches[DETAILS.IDS][:, BH, FBA]
            mids = mergers[MERGERS.IDS][:, BH]
            inds = np.where((good[:, BH, FBA]) & (dids != mids))[0]

            num_str += "%5d  " % (len(inds))
            if(len(inds) > 0): eg = inds[0]

        # print "       %s : %s" % (bh_str[BH], num_str)
        if(eg is not None):
            tmat = matches[DETAILS.SCALES][sel_id, BH_TYPE.OUT, :]
            tt = mergers[MERGERS.SCALES][sel_id]
            # print "\t\t%5d : %+f  %+f  (%+f)  %+f" % \
            #     (eg, tmat[BH_TIME.FIRST], tmat[BH_TIME.BEFORE], tt, tmat[BH_TIME.AFTER])

    # Check Time Matches
    # print "\n - - - Number of BAD time Matches:"
    # Iterate over both BHs
    for TYPE in [BH_TYPE.IN, BH_TYPE.OUT]:
        num_str = ""
        eg = None
        # Iterate over match times
        for TIME in [BH_TIME.FIRST, BH_TIME.BEFORE, BH_TIME.AFTER]:
            dt = matches[DETAILS.SCALES][:, TYPE, TIME]
            mt = mergers[MERGERS.SCALES]

            if(TIME == BH_TIME.AFTER): inds = np.where((good[:, TYPE, TIME]) & (dt < mt))[0]
            else:                      inds = np.where((good[:, TYPE, TIME]) & (dt >= mt))[0]

            num_str += "%5d  " % (len(inds))
            if(len(inds) > 0): eg = inds[0]

        # print "       %s : %s" % (bh_str[TYPE], num_str)
        if(eg is not None):
            tmat = matches[DETAILS.SCALES][eg, BH_TYPE.OUT, :]
            tt = mergers[MERGERS.SCALES][eg]
            # print "\t\t%5d : %+f  %+f  (%+f)  %+f" % \
            #     (eg, tmat[BH_TIME.FIRST], tmat[BH_TIME.BEFORE], tt, tmat[BH_TIME.AFTER])

    return


def inferMergerOutMasses(run, mergers=None, mdets=None, verbose=True, debug=False):
    """Based on 'merger' and 'details' information, infer the 'out' BH masses at time of mergers.

    The Illustris 'merger' files have the incorrect values output for the 'out' BH mass.  This
    method uses the data included in 'details' entries (via ``BHDetails``), which were matched
    to mergers here in ``BHMatcher``, to infer the approximate mass of the 'out' BH at the
    time of merger.

    The ``mergerDetails`` entries have the details for both 'in' and 'out' BHs both 'before' and
    'after' merger.  First the 'out' BH's entries just 'before' merger are used directly as the
    'inferred' mass.  In the few cases where these entries don't exist, the code falls back on
    calculating the difference between the total mass (given by the 'out' - 'after' entry) and the
    'in' BH mass recorded by the merger event (which should be correct); if this still doesn't
    work (which shouldn't ever happen), then the difference between the 'out'-'after' mass and the
    'in'-'before' mass is used, which should also be good --- but slightly less accurate (because
    the 'in'-'before' mass might have been recorded some small period of time before the actual
    merger event.

    Arguments
    ---------
       run     <int>  :
       mergers <dict> :
       mdets   <dict> :
       verbose <str>  :
       debug   <str>  :

    Returns
    -------
       outMass <flt64>[N] : inferred 'out' BH masses at the time of merger

    """

    # if(verbose): print " - - BHMatcher.inferOutMasses()"

    # Load Mergers
    if(mergers is None):
        # if(verbose): print " - - - Loading Mergers"
        mergers = illpy.illbh.BHMergers.loadFixedMergers(run, verbose=verbose)

    # Load Merger Details
    if(mdets is None):
        # if(verbose): print " - - - Loading Merger Details"
        mdets = loadMergerDetails(run, verbose=verbose)

    numMergers = mergers[MERGERS.NUM]
    mass = mdets[DETAILS.MASSES]
    # scal = mdets[DETAILS.SCALES]

    if(debug):
        inds_inn_bef = np.where(mass[:, BH_TYPE.IN, BH_TIME.BEFORE] <= 0.0)[0]
        inds_out_bef = np.where(mass[:, BH_TYPE.OUT, BH_TIME.BEFORE] <= 0.0)[0]
        inds_out_aft = np.where(mass[:, BH_TYPE.OUT, BH_TIME.AFTER] <= 0.0)[0]
        # print "BHMatcher.inferOutMasses() : %d missing IN  BEFORE" % (len(inds_inn_bef))
        # print "BHMatcher.inferOutMasses() : %d missing OUT BEFORE" % (len(inds_out_bef))
        # print "BHMatcher.inferOutMasses() : %d missing OUT AFTER " % (len(inds_out_aft))

    # Fix Mass Entries
    # ----------------
    # if(verbose): print " - - - Inferring 'out' BH masses at merger"

    # Details entries just before merger are the best option, default to this
    massOut = np.array(mass[:, BH_TYPE.OUT, BH_TIME.BEFORE])
    inds = np.where(massOut <= 0.0)[0]
    # if(verbose):
    #     print " - - - - %d/%d Entries missing details 'out' 'before'" % \
    #         (len(inds), numMergers)

    # If some are missing, use difference from out-after and merger-in
    inds = np.where(massOut <= 0.0)[0]
    if(len(inds) > 0):
        massOutAfter = mass[inds, BH_TYPE.OUT, BH_TIME.AFTER]
        massInDuring = mergers[MERGERS.MASSES][inds, BH_TYPE.IN]
        massOut[inds] = massOutAfter - massInDuring
        inds = np.where(massOut[inds] > 0.0)[0]
        # print " - - - - %d Missing entries replaced with 'out' 'after' minus merger 'in'" % \
        #     (len(inds))

    # If some are still missing, use difference from out-after and in-before
    inds = np.where(massOut <= 0.0)[0]
    if(len(inds) > 0):
        massOutAfter = mass[inds, BH_TYPE.OUT, BH_TIME.AFTER]
        massInBefore = mass[inds, BH_TYPE.IN, BH_TIME.BEFORE]
        massOut[inds] = massOutAfter - massInBefore
        inds = np.where(massOut[inds] > 0.0)[0]
        # print " - - - - %d Missing entries replaced with 'out' 'after' minus 'in' 'before'" % \
        #     (len(inds))

    inds = np.where(massOut <= 0.0)[0]
    # if(verbose): print " - - - - %d/%d Out masses still invalid" % (len(inds), numMergers)

    return massOut
'''

if(__name__ == "__main__"): main()
