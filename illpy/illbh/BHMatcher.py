"""Routines to match BH Mergers with BH Details entries based on times and IDs.

The matched details files should be precomputed by executing this file as a script,
for example:

    $ mpirun -n 64 python -m illpy.illbh.BHMatcher

This will create the 'merger-details' and 'remnant-details' files respectively, which can then be
loaded via the API functions `loadMergerDetails` and `loadRemnantDetails` respectively.


Functions
---------
-   main                     - Check if details files exist, if not manage their creation.
-   loadMergerDetails        - Load a previously calculated merger-details file.
-   loadRemnantDetails       - Load a previously calculated remnant-details file.

-   allDetailsForBHLineage   - Load all of the details entries for a given BH lineage (merger tree).
-   inferMergerOutMasses     - Infer the 'out' BH masses at time of mergers based on available data.

-   _matchMergerDetails      - Find details entries matching merger-BH ID numbers.
-   _matchRemnantDetails     - Combine merger-details entries to obtain an entire remnant's life.
-   _detailsForMergers_snapshots - Find details entries for BH IDs in a particular snapshots.
-   _saveDetails             - Package details into dictionary and save to NPZ file.
-   _findNextMerger          - Find the next merger index in which a particular BH participates.
-   _indBefAft               - Retrieve the index matching the minimum greater-than zero of input.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
from datetime import datetime

from mpi4py import MPI

import illpy.illbh.BHDetails
import illpy.illbh.BHMergers
import illpy.illbh.BHTree
import illpy.illbh.BHConstants
from illpy.illbh.BHConstants import MERGERS, DETAILS, BH_TREE, _LOG_DIR, BH_TYPE, \
    GET_MERGER_DETAILS_FILENAME, GET_REMNANT_DETAILS_FILENAME, _MAX_DETAILS_PER_SNAP, \
    _distributeSnapshots, GET_BLACKHOLE_TREE_DETAILS_FILENAME
import illpy.Constants
from illpy.Constants import DTYPE, NUM_SNAPS

import zcode.inout as zio
import zcode.math as zmath

__version__ = '0.23'
_GET_KEYS = [DETAILS.SCALES, DETAILS.MASSES, DETAILS.MDOTS, DETAILS.RHOS, DETAILS.CS]


def main(run=1, verbose=True, debug=True, loadsave=True, redo_mergers=False, redo_remnants=True):
    # Initialization
    # --------------
    #     MPI Parameters
    comm = MPI.COMM_WORLD
    rank = comm.rank
    name = __file__
    header = "\n%s\n%s\n%s" % (name, '='*len(name), str(datetime.now()))

    if(rank == 0):
        zio.checkPath(_LOG_DIR)
    comm.Barrier()

    # Initialize log
    log = illpy.illbh.BHConstants._loadLogger(
        __file__, debug=debug, verbose=verbose, run=run, rank=rank, version=__version__)
    log.debug(header)
    if(rank == 0):
        print("Log filename = ", log.filename)

    # if rank == 0: log.warning("Running 'allDetailsForBHLineage(run, 0, log)'")
    # allDetailsForBHLineage(run, 336, log)
    # return

    # Check status of files, determine what operations to perform
    create_mergerDets = False
    create_remnantDets = False
    if(rank == 0):
        # Check merger details status
        mergerDetFName = GET_MERGER_DETAILS_FILENAME(
            run, __version__, _MAX_DETAILS_PER_SNAP)
        log.info("Merger Details file: '%s'" % (mergerDetFName))
        if(not os.path.exists(mergerDetFName)):
            log.warning(" - Merger details file does not exist.")
            create_mergerDets = True
        else:
            log.info("Merger Details file exists.")
            if(not loadsave or redo_mergers):
                log.info(" - Recreating anyway.")
                create_mergerDets = True

        # Check remnants details status
        remnantDetFName = GET_REMNANT_DETAILS_FILENAME(run, __version__, _MAX_DETAILS_PER_SNAP)
        log.info("Remnant Details file: '%s'" % (remnantDetFName))

        if(not os.path.exists(remnantDetFName)):
            log.warning(" - Remnant Details file does not exist.")
            create_remnantDets = True
        else:
            log.info("Remnant Details file exists.")
            if(not loadsave or redo_remnants):
                log.info(" - Recreating anyway.")
                create_remnantDets = True

    # Synchronize control-flow flags
    create_mergerDets = comm.bcast(create_mergerDets, root=0)
    create_remnantDets = comm.bcast(create_remnantDets, root=0)
    comm.Barrier()
    mdets = None

    # Match Merger BHs to Details entries
    # -----------------------------------
    if(create_mergerDets):
        log.info("Creating Merger Details")
        beg = datetime.now()
        mdets = _matchMergerDetails(run, log)
        end = datetime.now()
        log.debug(" - Done after %s" % (str(end - beg)))

    # Extend merger-details to account for later mergers
    # --------------------------------------------------
    if(create_remnantDets):
        log.info("Creating Remnant Details")
        beg = datetime.now()
        _matchRemnantDetails(run, log, mdets=mdets)
        end = datetime.now()
        log.debug(" - Done after %s" % (str(end - beg)))

    return


def loadMergerDetails(run, verbose=True, log=None):
    """Load a previously calculated merger-details file.

    Arguments
    ---------
    run : int
        Illustris run numer {1,3}.
    verbose : bool
        Should verbose output be produced if a log is created.
        Only applies if no `log` is provided.
    log : ``logging.Logger`` object or `None`
        Logging object for output.
        If provided, the `verbose` parameter is ignored.

    Returns
    -------
    megerDets : dict
        Dictionary of merger-details data.

    """
    if(log is None):
        log = illpy.illbh.BHConstants._loadLogger(
            __file__, verbose=verbose, debug=False, run=run, tofile=False)

    log.debug("loadMergerDetails()")
    mergerDetFName = GET_MERGER_DETAILS_FILENAME(run, __version__, _MAX_DETAILS_PER_SNAP)
    logStr = "Merger Details File '%s'" % (mergerDetFName)
    mergerDets = None
    if(os.path.exists(mergerDetFName)):
        mergerDets = zio.npzToDict(mergerDetFName)
        logStr += " loaded."
        log.info(logStr)
    else:
        logStr += " does not exist."
        log.warning(logStr)

    return mergerDets


def loadRemnantDetails(run, verbose=True, log=None):
    """Load a previously calculated remnant-details file.

    Arguments
    ---------
    run : int
        Illustris run numer {1,3}.
    verbose : bool
        Should verbose output be produced if a log is created.
        Only applies if no `log` is provided.
    log : ``logging.Logger`` object or `None`
        Logging object for output.
        If provided, the `verbose` parameter is ignored.

    Returns
    -------
    remnantDets : dict
        Dictionary of remnant-details data.

    """
    if(log is None):
        log = illpy.illbh.BHConstants._loadLogger(
            __file__, verbose=verbose, debug=False, run=run, tofile=False)

    log.debug("loadRemnantDetails()")
    remnantDetFName = GET_REMNANT_DETAILS_FILENAME(run, __version__, _MAX_DETAILS_PER_SNAP)
    logStr = "Remnant Details File '%s'" % (remnantDetFName)
    remnantDets = None
    if(os.path.exists(remnantDetFName)):
        remnantDets = zio.npzToDict(remnantDetFName)
        logStr += " loaded."
        log.info(logStr)
    else:
        logStr += " does not exist."
        log.warning(logStr)

    return remnantDets


def allDetailsForBHLineage(run, mrg, log, reload=False):
    """Load all of the details entries for a given BH lineage (merger tree).

    Arguments
    ---------
    run : int
    mrg : int
    log : ``logging.Logger`` object
    reload : bool
        Even if the results file exists, recalculate them.

    """
    log.debug("allDetailsForBHLineage()")
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    bhIDs = None
    fname = None
    if rank == 0:
        # get the final merger number, the unique BH IDs, and the merger indices of this tree
        finMerger, bhIDs, mrgInds = illpy.illbh.BHTree.allIDsForTree(run, mrg)
        # Construct the appropriate file-name
        # fname = "ill-%d_merger-%d_full-tree-details.npz" % (run, mrg)
        fname = GET_BLACKHOLE_TREE_DETAILS_FILENAME(run, finMerger, __version__)
        log.debug(" - Final merger %d, filename: '%s'" % (finMerger, fname))
        if os.path.exists(fname):
            if size == 1:
                log.debug(" - File exists, loading.")
                data = zio.npzToDict(fname)
                return data
            elif not reload:
                fname = None

        bhIDs = np.array(bhIDs)
        numBHs = bhIDs.size
        log.info(" - Merger {} has a tree with {} unique BHs".format(mrg, numBHs))
        if numBHs < 2:
            errStr = "ERROR: only IDs found for merger {} are : {}".format(mrg, str(bhIDs))
            log.error(errStr)
            raise RuntimeError(errStr)

    # Distribute snapshots to each processor
    mySnaps = np.arange(NUM_SNAPS)
    if size > 1:
        # If save-file already exists, discontinue.
        fname = comm.bcast(fname, root=0)
        if not fname:
            log.debug(" - `fname` is invalid.  Terminating.")
            return

        # Send unique IDs to all processors
        bhIDs = comm.bcast(bhIDs, root=0)
        # Randomize which snapshots go to which processor for load-balancing
        mySnaps = np.random.permutation(mySnaps)
        # Make sure all ranks are synchronized on initial (randomized) list before splitting
        mySnaps = comm.bcast(mySnaps, root=0)
        mySnaps = np.array_split(mySnaps, size)[rank]
        comm.Barrier()

    log.info("Rank {:d}/{:d} with {:d} Snapshots [{:d} ... {:d}]".format(
        rank, size, mySnaps.size, mySnaps.min(), mySnaps.max()))

    # Get details entries for unique merger IDs in snapshot list
    # ----------------------------------------------------------
    #    Use ``maxPerSnap = None`` to save ALL details entries
    nums, scales, masses, mdots, dens, csnds, ids = \
        _detailsForMergers_snapshots(run, mySnaps, bhIDs, None, log)

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

            nums = np.zeros(numBHs)
            scales = numBHs*[[None]]
            masses = numBHs*[[None]]
            mdots = numBHs*[[None]]
            dens = numBHs*[[None]]
            csnds = numBHs*[[None]]
            ids = numBHs*[[None]]

            # Iterate over each black-hole and processor, collect results into single arrays
            for ii, mm in enumerate(bhIDs):
                for jj in xrange(size):
                    errStr = ""
                    if(tempIds[jj][ii][0] is not None):
                        dd = tempIds[jj][ii][0]
                        # Make sure all of the details IDs are consistent
                        if(np.any(tempIds[jj][ii] != dd)):
                            errStr += "ii = {}, jj = {}, mm = {}; tempIds[0] = {}".format(
                                ii, jj, mm, dd)
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
                                errStr += "\nii = {}, jj = {}, mm = {}, dd = {}, ids = {}"
                                errStr = errStr.format(ii, jj, mm, dd, str(ids))
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

    # Sort results by time
    if(rank == 0):

        # Sort entries for each BH by scalefactor
        log.debug(" - Sorting")
        beg = datetime.now()
        for ii in xrange(numBHs):
            if(nums[ii] == 0): continue
            # Check ID numbers yet again
            if(not np.all(ids[ii] == ids[ii][0])):
                errStr = "Error!  ii = {}, ID = {}, IDs = {}"
                errStr = errStr.format(ii, ids[ii][0], ids[ii])
                log.error(errStr)
                zio.mpiError(comm, log=log, err=errStr)

            inds = np.argsort(scales[ii])
            scales[ii] = scales[ii][inds]
            masses[ii] = masses[ii][inds]
            dens[ii] = dens[ii][inds]
            mdots[ii] = mdots[ii][inds]
            csnds[ii] = csnds[ii][inds]

        end = datetime.now()
        log.debug(" - - Done after %s" % (str(end-beg)))

        # Save data
        data = _saveDetails(fname, run, ids, scales, masses, dens, mdots, csnds, log)

    return


def inferMergerOutMasses(run, mergers=None, mdets=None, log=None):
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
    if(log is None):
        log = illpy.illbh.BHConstants._loadLogger(
            __file__, verbose=True, debug=False, run=run, tofile=False)

    log.debug("inferMergerOutMasses()")

    # Load Mergers
    if(mergers is None):
        mergers = illpy.illbh.BHMergers.loadFixedMergers(run, verbose=False)
    m_scales = mergers[MERGERS.SCALES]
    m_masses = mergers[MERGERS.MASSES]
    numMergers = mergers[MERGERS.NUM]
    del mergers

    # Load Merger Details
    if(mdets is None):
        mdets = loadMergerDetails(run, log=log)
    d_masses = mdets[DETAILS.MASSES]
    d_scales = mdets[DETAILS.SCALES]
    del mdets

    # Find details entries before and after merger
    massBef = np.zeros_like(m_masses)
    massAft = np.zeros_like(m_masses)
    for ii, sc in enumerate(m_scales):
        for bh in [BH_TYPE.IN, BH_TYPE.OUT]:
            if(d_scales[ii].size == 0):
                log.warning("Merger %s with zero details entries" % str(ii))
            else:
                # 'before' is ``sc > d_scales``
                bef = _indBefAft(sc - d_scales[ii, bh])
                if(bef is not None):
                    if(np.isfinite(d_masses[ii, bh][bef])):
                        massBef[ii, bh] = d_masses[ii, bh][bef]
                    elif(bef > 0):
                        massBef[ii, bh] = d_masses[ii, bh][bef-1]

                # 'after' is ``d_scales > sc``
                aft = _indBefAft(d_scales[ii, bh] - sc)
                if(aft):
                    massAft[ii, bh] = d_masses[ii, bh][aft]

    # Fix Mass Entries
    # ----------------
    massBef = massBef.reshape(2*numMergers)
    massAft = massAft.reshape(2*numMergers)
    masses = np.zeros_like(massBef)
    ntot = masses.size

    bads = np.where(np.isfinite(massBef) == False)[0]
    print("Bads Before = ", bads)
    print("\t", massBef[bads])
    bads = np.where(np.isfinite(massAft) == False)[0]
    print("Bads After = ", bads)
    print("\t", massAft[bads])

    # Use 'before' masses
    inds = np.where(massBef > 0.0)
    masses[inds] = massBef[inds]
    bads = np.where(masses == 0.0)
    nfix = np.size(inds)
    nbad = np.size(bads)
    frac = nfix/ntot
    log.info(" - Used %d/%d = %.4f after masses.  %d remain" % (nfix, ntot, frac, nbad))

    massBef = massBef.reshape(numMergers, 2)
    massAft = massAft.reshape(numMergers, 2)
    masses = masses.reshape(numMergers, 2)

    return masses


def _matchMergerDetails(run, comm, log):
    """Find details entries matching merger-BH ID numbers.

    Stores at most ``_MAX_DETAILS_PER_SNAP`` entries per snapshot for each BH.
    """
    log.debug("_matchMergerDetails()")
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # Load Unique ID numbers to distribute to all tasks
    bhIDsUnique = None
    if rank == 0:
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
    """
    mySnaps = np.arange(NUM_SNAPS)
    if(size > 1):
        # Randomize which snapshots go to which processor for load-balancing
        mySnaps = np.random.permutation(mySnaps)
        # Make sure all ranks are synchronized on initial (randomized) list before splitting
        mySnaps = comm.bcast(mySnaps, root=0)
        mySnaps = np.array_split(mySnaps, size)[rank]
    """
    mySnaps = _distributeSnapshots(comm)

    log.info("Rank {:d}/{:d} with {:d} Snapshots [{:d} ... {:d}]".format(
        rank, size, mySnaps.size, mySnaps.min(), mySnaps.max()))

    # Get details entries for unique merger IDs in snapshot list
    # ----------------------------------------------------------
    nums, scales, masses, mdots, dens, csnds, ids = \
        _detailsForMergers_snapshots(run, mySnaps, bhIDsUnique, _MAX_DETAILS_PER_SNAP, log)

    # Collect results and organize
    # ----------------------------
    if size > 1:
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
        if rank == 0:
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
                    if tempIds[jj][ii][0] is not None:
                        dd = tempIds[jj][ii][0]
                        # Make sure all of the details IDs are consistent
                        if np.any(tempIds[jj][ii] != dd):
                            errStr += "ii = {}, jj = {}, mm = {}; tempIds[0] = {}"
                            errStr = errStr.format(ii, jj, mm, dd)
                            errStr += " tempIds = {}".format(str(tempIds[ii]))

                        # Make sure details IDs match expected merger ID
                        if dd != mm:
                            errStr += "\nii = {}, jj = {}, mm = {}; dd = {}".format(ii, jj, mm, dd)

                        # If no entries have been stored yet, replace with first entries
                        if ids[ii][0] is None:
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
                                errStr += "\nii = {}, jj = {}, mm = {}, dd = {}, ids = {}"
                                errStr = errStr.format(ii, jj, mm, dd, str(ids))
                            ids[ii] = np.append(ids[ii], tempIds[jj][ii])
                            scales[ii] = np.append(scales[ii], tempScales[jj][ii])
                            masses[ii] = np.append(masses[ii], tempMasses[jj][ii])
                            mdots[ii] = np.append(mdots[ii], tempMdots[jj][ii])
                            dens[ii] = np.append(dens[ii], tempDens[jj][ii])
                            csnds[ii] = np.append(csnds[ii], tempCsnds[jj][ii])

                        # Count the number of entries for each BH from each processor
                        nums[ii] += tempNums[jj][ii]

                    if len(errStr) > 0:
                        log.error(errStr)
                        zio.mpiError(comm, log=log, err=errStr)

            # Merge lists of snapshots, and look for any missing
            mySnaps = np.hstack(mySnaps)
            log.debug("Obtained %d Snapshots" % (mySnaps.size))
            missingSnaps = []
            for ii in xrange(NUM_SNAPS):
                if(ii not in mySnaps):
                    missingSnaps.append(ii)

            if len(missingSnaps) > 0:
                log.warning("WARNING: snaps %s not in results!" % (str(missingSnaps)))

            log.debug("Total entries stored = %d" % (np.sum([np.sum(nn) for nn in nums])))

    # Convert from uinique BH IDs back to full mergers list.  Sort results by time (scalefactor)
    if rank == 0:
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
                if nums[ii, jj] == 0:
                    continue
                # Check ID numbers yet again
                if not np.all(ids[ii, jj] == mergerIDs[ii, jj]):
                    errStr = "Error!  ii = {}, jj = {}.  Merger ID = {}, det ID = {}"
                    errStr = errStr.format(ii, jj, mergerIDs[ii, jj], ids[ii, jj])
                    errStr += "nums[ii,jj] = {:s}, shape {:s}"
                    errStr = errStr.format(str(nums[ii, jj]), str(np.shape(nums[ii, jj])))
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

        # Save data
        filename = GET_MERGER_DETAILS_FILENAME(run, __version__, _MAX_DETAILS_PER_SNAP)
        data = _saveDetails(filename, run, ids, scales, masses, dens, mdots, csnds, log)

    return


def _matchRemnantDetails(run, log, mdets=None):
    """Combine merger-details entries to obtain details for an entire merger-remnant's life.

    Each merger 'out'-BH is followed in subsequent mergers to combine the details entries forming
    a continuous chain of details entries for the remnant's entire life after the initial merger.

    Loads `BHMergers` and `MergerDetails` files and uses that data to construct remnant details.
    Runs on a single core (processors with ``rank > 0`` simply return at start).

    Arguments
    ---------
    run : int
        Illustris simulation run number {1,3}.
    log : ``logging.Logger``
        Logging object.
    mdets : dict or `None`
        `MergerDetails` data, loaded if not proveded.

    Returns
    -------
    data : dict
        `RemnantDetails` data.

    """
    log.debug("_matchRemnantDetails()")
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Only use root-processor (for now at least)
    if(rank != 0):
        return

    # Load Mergers
    # ------------
    log.debug("Loading Mergers")
    mergers = illpy.illbh.BHMergers.loadFixedMergers(run)
    m_scales = mergers[MERGERS.SCALES]
    m_ids = mergers[MERGERS.IDS]
    m_masses = mergers[MERGERS.MASSES]
    numMergers = np.size(m_scales)
    del mergers

    # Load merger-details file
    # ------------------------
    loadname = GET_MERGER_DETAILS_FILENAME(run, __version__, _MAX_DETAILS_PER_SNAP)
    log.info(" - Loading from '%s'" % (loadname))
    if(not os.path.exists(loadname)):
        errStr = "ERROR: '%s' does not exist!" % (loadname)
        log.error(errStr)
        zio.mpiError(comm, log=log, err=errStr)

    beg = datetime.now()
    if(mdets is None):
        mdets = zio.npzToDict(loadname)
    # Unpack data
    d_ids = mdets[DETAILS.IDS]
    d_scales = mdets[DETAILS.SCALES]
    d_masses = mdets[DETAILS.MASSES]
    d_dens = mdets[DETAILS.RHOS]
    d_mdots = mdets[DETAILS.MDOTS]
    d_csnds = mdets[DETAILS.CS]
    nums = mdets['nums']
    #     Log number of entries stored
    entries = np.sum(nums)
    bon = np.count_nonzero(nums)
    tot = np.size(nums)
    frac = bon/tot
    end = datetime.now()
    log.debug(" - - Loaded {} entries for {}/{} = {} BHs after {}".format(
              entries, bon, tot, frac, str(end-beg)))

    # Load BH Merger Tree
    tree = illpy.illbh.BHTree.loadTree(run)
    nextBH = tree[BH_TREE.NEXT]
    del tree

    # Initialize data for results
    ids = np.zeros(numMergers, dtype=DTYPE.ID)
    ndups = np.zeros(numMergers, dtype=np.int)
    nents = np.zeros(numMergers, dtype=np.int)
    oscf = np.ones(numMergers, dtype=np.bool)
    nosc = np.zeros(numMergers, dtype=np.int)     # Number of non-monotonic entries deleted
    oscScales = numMergers*[None]                 # Scales with non-monotonicity
    idnums = numMergers*[None]
    mrgnums = numMergers*[None]
    scales = numMergers*[None]
    masses = numMergers*[None]
    mcorrected = numMergers*[None]
    dens = numMergers*[None]
    mdots = numMergers*[None]
    csnds = numMergers*[None]

    # Iterate over all mergers
    # ------------------------
    for ii in xrange(numMergers):
        # First Merger
        #    Store details after merger time for 'out' BH
        inds = np.where(d_scales[ii, BH_TYPE.OUT] > m_scales[ii])[0]
        if(np.size(inds) > 0):
            ids[ii] = d_ids[ii, BH_TYPE.OUT][inds[0]]
            scales[ii] = d_scales[ii, BH_TYPE.OUT][inds]
            masses[ii] = d_masses[ii, BH_TYPE.OUT][inds]
            dens[ii] = d_dens[ii, BH_TYPE.OUT][inds]
            mdots[ii] = d_mdots[ii, BH_TYPE.OUT][inds]
            csnds[ii] = d_csnds[ii, BH_TYPE.OUT][inds]
            idnums[ii] = d_ids[ii, BH_TYPE.OUT][inds]
            mrgnums[ii] = ii*np.ones(inds.size, dtype=int)
        else:
            log.warning("Merger %d without post-merger details entries!" % (ii))
            ids[ii] = m_ids[ii, BH_TYPE.OUT]
            scales[ii] = []
            masses[ii] = []
            dens[ii] = []
            mdots[ii] = []
            csnds[ii] = []
            idnums[ii] = []
            mrgnums[ii] = []

        # Subsequent mergers
        #    Find the next merger that this 'out' BH participates in
        nextMerger = nextBH[ii]
        checkID = m_ids[nextMerger, BH_TYPE.OUT]
        checkScale = m_scales[nextMerger]
        if(ids[ii] >= 0 and nextMerger >= 0):
            # Make sure `nextMerger` is correct, fix if not
            if(ids[ii] not in m_ids[nextMerger]):
                nextMerger = _findNextMerger(ids[ii], m_scales[ii], m_ids, m_scales)
                checkID = m_ids[nextMerger, BH_TYPE.OUT]
                checkScale = m_scales[nextMerger]
                # Error if still not fixed
                if(nextMerger >= 0 and ids[ii] not in m_ids[nextMerger]):
                    errStr = "ERROR: ids[{}] = {}, merger ids {} = {}"
                    errStr = errStr.format(ii, ids[ii], nextMerger, str(m_ids[nextMerger]))
                    log.error(errStr)
                    zio.mpiError(comm, log=log, err=errStr)

        #    while a subsequent merger exists... store those entries
        while(nextMerger >= 0):
            nextIDs = d_ids[nextMerger, BH_TYPE.OUT][:]
            # Make sure ID numbers match
            if(checkID):
                if(np.any(checkID != nextIDs)):
                    errStr = "ERROR: ii = %d, next = %d, IDs don't match!" % (ii, nextMerger)
                    errStr += "\nids[ii] = %d, check = %d" % (ids[ii], checkID)
                    errStr += "\nd_ids = %s" % (str(nextIDs))
                    log.error(errStr)
                    zio.mpiError(comm, log=log, err=errStr)

            if(np.size(scales[ii]) > 0):
                inds = np.where(d_scales[nextMerger, BH_TYPE.OUT] > np.max(scales[ii]))[0]
            else:
                inds = np.where(d_scales[nextMerger, BH_TYPE.OUT] > m_scales[ii])[0]

            scales[ii] = np.append(scales[ii], d_scales[nextMerger, BH_TYPE.OUT][inds])
            masses[ii] = np.append(masses[ii], d_masses[nextMerger, BH_TYPE.OUT][inds])
            dens[ii] = np.append(dens[ii], d_dens[nextMerger, BH_TYPE.OUT][inds])
            mdots[ii] = np.append(mdots[ii], d_mdots[nextMerger, BH_TYPE.OUT][inds])
            csnds[ii] = np.append(csnds[ii], d_csnds[nextMerger, BH_TYPE.OUT][inds])
            idnums[ii] = np.append(idnums[ii], d_ids[nextMerger, BH_TYPE.OUT][inds])
            mrgnums[ii] = np.append(mrgnums[ii], nextMerger*np.ones(inds.size, dtype=int))

            # Get next merger in Tree
            nextMerger = nextBH[nextMerger]
            # Make sure `nextMerger` is correct, fix if not
            if(checkID not in m_ids[nextMerger] and nextMerger >= 0):
                nextMerger = _findNextMerger(checkID, checkScale, m_ids, m_scales)
                # Error if still not fixed
                if(nextMerger >= 0 and ids[ii] not in m_ids[nextMerger]):
                    errStr = "ERROR: ids[{}] = {}, merger ids {} = {}"
                    errStr = errStr.format(ii, ids[ii], nextMerger, str(m_ids[nextMerger]))
                    log.error(errStr)
                    zio.mpiError(comm, log=log, err=errStr)

            #    Get ID of next out-BH
            checkID = m_ids[nextMerger, BH_TYPE.OUT]
            checkScale = m_scales[nextMerger]

        # Appended entries may no longer be sorted, sort them
        if(np.size(scales[ii]) > 0):
            inds = np.argsort(scales[ii])
            scales[ii] = scales[ii][inds]
            masses[ii] = masses[ii][inds]
            dens[ii] = dens[ii][inds]
            mdots[ii] = mdots[ii][inds]
            csnds[ii] = csnds[ii][inds]
            idnums[ii] = idnums[ii][inds]
            mrgnums[ii] = mrgnums[ii][inds]

            # Find and remove duplicates
            dups = np.where(np.isclose(scales[ii][:-1], scales[ii][1:], rtol=1e-8))[0]
            if(np.size(dups) > 0):
                ndups[ii] += np.size(dups)
                scales[ii] = np.delete(scales[ii], dups)
                masses[ii] = np.delete(masses[ii], dups)
                dens[ii] = np.delete(dens[ii], dups)
                mdots[ii] = np.delete(mdots[ii], dups)
                csnds[ii] = np.delete(csnds[ii], dups)
                idnums[ii] = np.delete(idnums[ii], dups)
                mrgnums[ii] = np.delete(mrgnums[ii], dups)

            # Find and remove non-monotonic entries
            bads = np.where(np.diff(masses[ii]) < 0.0)[0]
            if bads.size > 0:
                nosc[ii] = bads.size
                oscScales[ii] = scales[ii][bads]
                # print(" - Nonmonotonic scales = ", scales[ii][bads])
                scales[ii] = np.delete(scales[ii], bads)
                masses[ii] = np.delete(masses[ii], bads)
                dens[ii] = np.delete(dens[ii], bads)
                mdots[ii] = np.delete(mdots[ii], bads)
                csnds[ii] = np.delete(csnds[ii], bads)
                idnums[ii] = np.delete(idnums[ii], bads)
                mrgnums[ii] = np.delete(mrgnums[ii], bads)

            nents[ii] = np.size(scales[ii])
            # Look for non-monotonicity
            if(not zmath.mono(masses[ii], 'ge')):
                oscf[ii] = True

        else:
            log.warning("Merger %d without ANY entries." % (ii))

    log.debug("Remnant details collected.")
    log.debug(" - Average number of entries: %e" % (np.mean(nents)))
    log.debug(" - Average number of duplicates removed: %e" % (np.mean(ndups)))
    log.debug(" - Duplicate entries:")
    log.debug(" - - %d Mergers with duplicates" % (np.count_nonzero(ndups)))
    log.debug(" - - Median and 68% (overall): {}".format(
        str(zmath.confidenceIntervals(ndups, ci=0.68))))
    log.debug(" - - Average num (nonzero): {}".format(
        str(zmath.confidenceIntervals(ndups, ci=0.68, filter='g')))
    log.debug(" - Non-monotonic entries:")
    log.debug(" - - %d Mergers with non-monotonicity" % (np.count_nonzero(nosc)))
    log.debug(" - - Median and 68% (overall): {}".format(
        str(zmath.confidenceIntervals(nosc, ci=0.68))))
    log.debug(" - - Average num (nonzero): {}".fomat(
        str(zmath.confidenceIntervals(nosc, ci=0.68, filter='g')))
    log.debug(" - - Number remaining non-monotonic: %d" % (np.count_nonzero(oscf)))

    fname = 'nonmono-scales.npz'
    np.savez(fname, scales=oscScales)
    print("Saved to '%s'" % (fname))

    # Calculate 'corrected' remnant masses
    # ------------------------------------
    log.debug("'Correcting' remnant masses for later mergers.")
    mcorrected = np.array(masses)
    correctionErrs = []
    goods = 0
    count = 0
    for ii in xrange(numMergers):
        jj = nextBH[ii]
        while(jj >= 0):
            inMass = m_masses[jj, BH_TYPE.IN]
            mscale = m_scales[jj]
            # Find scales after the merger
            inds = np.where(scales[ii] > mscale)[0]
            if(inds.size == 0):
                log.warning("Merger %d, Next %d, no matching scales after %f!" % (ii, jj, mscale))
                correctionErrs.append(ii)
            else:
                useInds = np.arange(np.max([inds[0]-3, 0]), np.min([inds[0]+3, inds[-1]]))

                # If there is only one index, use it
                if(useInds.size == 1):
                    mcorrected[ii][useInds[0]:] -= inMass
                    goods += 1
                # If there are multiple indices, try to find merger
                else:
                    # Get changes in mass in these indices
                    minc = np.diff(masses[ii][useInds])
                    # Look for first time mass increases by more than mass of second-BH
                    #     `+1` because 'True' means ``minc[1] > minc[0]+inMass``
                    mrg = useInds[np.where(minc >= inMass)[0] + 1]
                    if(mrg.size == 0):
                        # log.warning("Merger %d, Next %d, missing secondary mass!" % (ii, jj))
                        correctionErrs.append(ii)
                        # best guess: 'correct' all scales after 'merger' time
                        mcorrected[ii][inds[0]:] -= inMass
                    else:
                        mcorrected[ii][mrg[0]:] -= inMass
                        goods += 1

            jj = nextBH[jj]
            if count > 100: return

    # log.info(" - {} Correction errors: {}".format(len(correctionErrs), str(correctionErrs)))
    log.info(" - {} Correction errors".format(len(correctionErrs)))
    log.info(" - {} Successful corerctions".format(goods))

    savename = GET_REMNANT_DETAILS_FILENAME(run, __version__, _MAX_DETAILS_PER_SNAP)
    data = _saveDetails(savename, run, ids, scales, masses, dens, mdots, csnds, log,
                        mcorrected=mcorrected)

    return data


def _detailsForMergers_snapshots(run, snapshots, bhIDsUnique, maxPerSnap, log):
    """Find details entries for BH IDs in a particular snapshots.

    For each snapshot, store at most `_MAX_DETAILS_PER_SNAP` entries for each blackhole,
    interpolating values to an even spacing in scale-factor if more entries are found.  The first
    and last entries found are stored as is.

    Arguments
    ---------
    run : int
        Illustris run number {1,3}.
    snapshots : int or array_like of int
        Snapshots to search for details entries.
    bhIDsUnique : (N,) array of long
        Sequence of all unique, merger-BH ID numbers.
    maxPerSnap : int or `None`
        The maximum number of details entries to store for each snapshot.  If more entries than
        this are found, the values are interpolated down to a linearly-even spacing of
        scale-factors within the range of the matching details entries.
        The maximum number of entries stored is thus ``maxPerSnap*np.size(snapshots)``
        for each BH.  When loading all merger/remnant details, this should be
        `_MAX_DETAILS_PER_SNAP`.
    log : ``logging.Logger`` object
        Object for logging stream output.

    Returns
    -------
    numStoredTotal : (N,) array of int
        For each unique BH ID, the number of details entries stored.
    scales : (N,) array of arrays of float
        Scale-factor of each entry.
        Each of the `N` entries corresponds to a unique merger-BH.  In that entry is an array
        including all of the matching details entries found for that BH.  The length of each of
        these arrays can be (effectively) any value between zero and
        ``_MAX_DETAILS_PER_SNAP*np.size(snapshots)``.  This applies to all of the following
        returned values as-well.
    masses : (N,) array of arrays of float
        Mass of the BH in each entry.
    mdots : (N,) array of arrays of float
        Mass accretion rate of the BH.
    dens : (N,) array of arrays of float
        Local density around the BH.
    csnds : (N,) array of arrays of float
        Local sound-speed around the bH.
    ids : (N,) array of arrays of long
        ID number of the black-hole for each entry (for error-checking).

    """
    log.debug("_detailsForMergers_snapshot()")

    comm = MPI.COMM_WORLD

    # Initialize
    # ----------
    #     Make sure snapshots are iterable
    snapshots = np.atleast_1d(snapshots)
    numUniqueBHs = bhIDsUnique.size
    log.debug(" - {} Unique IDs, maxPerSnap = {}".format(numUniqueBHs, maxPerSnap))

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

                # Interpolate down to only `maxPerSnap` entries
                if maxPerSnap and numMatchesSnap[ii] > maxPerSnap:
                    #    Create even spacing in scale-factor to interpolate to
                    newScales = zmath.spacing(tempScales, scale='lin', num=maxPerSnap)
                    tempMasses = np.interp(newScales, tempScales, tempMasses)
                    tempMdots = np.interp(newScales, tempScales, tempMdots)
                    tempDens = np.interp(newScales, tempScales, tempDens)
                    tempCsnds = np.interp(newScales, tempScales, tempCsnds)
                    #    Cant interpolate IDs, select random subset instead...
                    tempIDs = np.random.choice(tempIDs, size=maxPerSnap, replace=False)
                    if(not isinstance(tempIDs[0], DTYPE.ID)):
                        errStr = "Error: incorrect dtype for random tempIDs = {}"
                        errStr = errStr.format(np.dtype(tempIDs[0]))
                        log.error(errStr)
                        raise RuntimeError(errStr)

                    tempScales = newScales

                # Store matches
                #    If this is the first set of entries, replace ``[None]``
                if(scales[ii][0] is None):
                    try:
                        scales[ii] = tempScales
                    except:
                        print(np.shape(scales[ii]))
                        print(scales[ii])
                        print(np.shape(tempScales))
                        zio.mpiError(comm)

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

        numMatchesTotal += numMatchesSnap
        numStoredTotal += numStoredSnap
        snapTotal = np.sum(numMatchesSnap)
        total = np.sum(numMatchesTotal)
        snapOcc = np.count_nonzero(numMatchesSnap)
        occ = np.count_nonzero(numMatchesTotal)
        log.debug(" - %7d Matches (%7d total), %4d BHs (%4d)" % (snapTotal, total, snapOcc, occ))
        log.debug(" - Average and total number stored = {:f}, {:d}".format(
            numStoredTotal.mean(), np.sum(numStoredTotal)))

    scales = np.array(scales)
    masses = np.array(masses)
    mdots = np.array(mdots)
    dens = np.array(dens)
    csnds = np.array(csnds)
    ids = np.array(ids)

    return numStoredTotal, scales, masses, mdots, dens, csnds, ids


def _saveDetails(fname, run, ids, scales, masses, dens, mdots, csnds, log, mcorrected=None):
    """Package details into dictionary and save to NPZ file.
    """
    log.debug("_saveDetails()")
    data = {DETAILS.IDS: ids,
            DETAILS.SCALES: scales,
            DETAILS.MASSES: masses,
            DETAILS.RHOS: dens,
            DETAILS.MDOTS: mdots,
            DETAILS.CS: csnds,
            DETAILS.RUN: np.array(run),
            DETAILS.CREATED: np.array(datetime.now().ctime()),
            DETAILS.VERSION: np.array(__version__),
            DETAILS.FILE: np.array(fname),
            'detailsPerSnapshot': np.array(_MAX_DETAILS_PER_SNAP),
            'detailsKeys': _GET_KEYS,
            }

    if mcorrected is not None:
        data['masses_corrected'] = mcorrected

    log.info(" - Saving data to '%s'" % (fname))
    beg = datetime.now()
    zio.dictToNPZ(data, fname, verbose=True)
    end = datetime.now()
    log.info(" - - Saved to '%s' after %s" % (fname, str(end-beg)))

    return data


def _findNextMerger(myID, myScale, ids, scales):
    """Find the next merger index in which a particular BH participates.

    Search the full list of merger-BH ID number for matches to the target BH (`myID`) which take
    place at scale factors after the initial merger scale-factor (`myScale`).
    If no match is found, `-1` is returned.

    Arguments
    ---------
    myID : long
        ID number of the target blackhole.
    myScale : float
        Scalefactor at which the initial merger occurs (look for next merger after this time).
    ids : (N,2) array of long
        ID number of all merger BHs
    scales : (N,) array of float
        Scalefactors at which all of the mergers occur.

    Returns
    -------
    nind : int
        Index of the next merger the `myID` blackhole takes place in.
        If no next merger is found, returns `-1`.


    Used by: `_matchRemnantDetails`
    """
    # Find where this ID matches another, but they dont have the same time (i.e. not same merger)
    search = (((myID == ids[:, 0]) | (myID == ids[:, 1])) & (myScale != scales))
    nind = np.where(search)[0]
    if(np.size(nind) > 0):
        # If multiple, find first
        if(np.size(nind) > 1):
            nind = nind[np.argmin(scales[nind])]
        else:
            nind = nind[0]

    else:
        nind = -1

    return nind


def _indBefAft(scaleDiff):
    """Retrieve the index matching the minimum of `scaleDiff` greater-than zero.

    Used by: `inferMergerOutMasses`
    """
    try:
        ind = zmath.argextrema(scaleDiff, 'min', 'g')
    except ValueError:
        ind = None

    return ind


if(__name__ == "__main__"): main()
