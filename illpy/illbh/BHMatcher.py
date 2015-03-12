"""
Routines to match BH Mergers with BH Details entries based on times and IDs.

The files 'blackhole_mergers_<N>.txt' contain the time and IDs ('in' and 'out' BHs) of BH mergers.
The 'in' BH's are the ones which are absorbed into the 'out' BHs---i.e. the ones which continue to
exist after the merger.  The merger files also contain masses, but these are incorrect for the
'out' BHs - and must be reconstructed from the 'details files: 'blackhole_details_<N>.txt', which
also contain information about the environment (e.g. density, etc).



"""


import random
import sys

import numpy as np
import traceback as tb

from glob import glob
from datetime import datetime

import BHDetails
import BHMergers
import BHConstants
from BHConstants import *

from .. import Constants
from .. import AuxFuncs as aux

RUN = 3
VERBOSE = True




###  ===================================  ###
###  =============  MAIN  ==============  ###
###  ===================================  ###



def main(run=RUN, verbose=VERBOSE):

    if( verbose ): print " - BHMatcher.py"

    # Set Basic Parameters
    mergers = BHMergers.loadMergers(run)

    # Get Details for Mergers
    if( verbose ): print " - - Matching Mergers to Details"
    start = datetime.now()
    mergerDetails = detailsForMergers(mergers, run, verbose=verbose)
    stop = datetime.now()
    if( verbose ): print " - - - Mergers matched after %s" % (str(stop-start))

    # Save merger details
    saveFile = 'ill-%d_matched-merger-details_new.npz' % (run)
    aux.saveDictNPZ(mergerDetails, saveFile, verbose=verbose)

    if( verbose ): print " - Done"

    return







###  ================================================  ###
###  ===============  OTHER FUNCTIONS  ==============  ###
###  ================================================  ###


def detailsForMergers(mergers, run, snapNums=None, verbose=VERBOSE):
    """
    Given a set of mergers, retrieve corresponding 'details' entries for BHs.

    Finds the details entries which occur closest to the 'merger' time, both
    before and after it (after only exists for the 'out' BHs).

    Arguments
    ---------
    mergers : dict, data arrays for mergers
    run : int, illustris run number {1,3}
    snapNums : array_like(int) (optional : None), particular snapshots in which
               to match.  If `None` then all snapshots are used.
    verbose : bool (optional : ``VERBOSE``), flag to print verbose output.

    
    Returns
    -------
    mergDets : dict, data arrays corresponding to each 'merger' BHs

    """

    if( verbose ): print " - - detailsForMergers()"

    numMergers = mergers[MERGERS_NUM]
    numSnaps   = Constants.NUM_SNAPS

    ### Choose target snapshots based on input; NOTE: must iterate in reverse ###
    # Default to all snapshots
    if( snapNums == None        ):
        snapNumbers = reversed(xrange(numSnaps))
        numSnapsHere = numSnaps
    # If a list is given, make sure it is in reverse order
    elif( np.iterable(snapNums) ):
        snapNumbers = reversed(sorted(snapNums))
        numSnapsHere = len(snapNums)
    # If a single number given, make iterable (list)
    else:
        snapNumbers = [ snapNums ]
        numSnapsHere = 1


    ### Allocate arrays for results ###
    #     Row for each merger; Column for each BH In merger [0-in, 1-out], and for before and after merger time
    NUM_BHS = 2                                                                                     # There are 2 BHs, {IN_BH, OUT_BH}
    NUM_TIMES = 3                                                                                   # There are 3 times, {DETAIL_BEFORE, DETAIL_AFTER, DETAIL_FIRST}

    '''
    ids   = -1*np.ones([numMergers, NUM_BHS, NUM_TIMES], dtype=_LONG)
    times = -1*np.ones([numMergers, NUM_BHS, NUM_TIMES], dtype=_DOUBLE)
    mass  = -1*np.ones([numMergers, NUM_BHS, NUM_TIMES], dtype=_DOUBLE)
    mdot  = -1*np.ones([numMergers, NUM_BHS, NUM_TIMES], dtype=_DOUBLE)
    rho   = -1*np.ones([numMergers, NUM_BHS, NUM_TIMES], dtype=_DOUBLE)
    cs    = -1*np.ones([numMergers, NUM_BHS, NUM_TIMES], dtype=_DOUBLE)
    '''

    # Initialize Dictionary to Invalid entries (-1)
    mergDets = {}

    for KEY in DETAILS_PHYSICAL_KEYS:
        if( KEY == DETAILS_IDS ): 
            mergDets[KEY] = -1 * np.ones([numMergers, NUM_BHS, NUM_TIMES], dtype=_LONG  )
        else:
            mergDets[KEY] = -1.0*np.ones([numMergers, NUM_BHS, NUM_TIMES], dtype=_DOUBLE)


    # Maintain a list of Mergers which could not be matched to Details
    missList = []

    ### Iterate over Target Snapshots ###
    if( verbose ): print " - - - Iterating over %d snapshots" % (numSnapsHere)
    startAll = datetime.now()
    numMatched = 0
    prevList = []                                                                                   # These are failed matches, passed on
    nextList = []                                                                                   # These will become 'prevList'

    for ii,snum in enumerate(snapNumbers):
        startOne = datetime.now()
        if( verbose ): print " - - - Snapshot %d/%d   # %d" % (ii, numSnapsHere, snum)

        s2m = mergers[MERGERS_MAP_STOM]

        # If there are no mergers to match, continue to next iteration
        if( len(s2m[snum]) == 0 and len(prevList) == 0 ):
            if( verbose ): print " - - - - No mergers for snapshot %d, or carried over" % (snum)
            continue

        ## Load appropriate Details File ##
        start = datetime.now()
        dets = BHDetails.loadBHDetails_NPZ(run, snum)
        stop = datetime.now()
        if( verbose ): print " - - - - Details loaded after %s" % (str(stop-start))

        # If there are no details here, continue to next iter, but save BHs
        if( dets[DETAILS_NUM] == 0 ):
            if( verbose ): print " - - - - No details for snapshot %d" % (snum)

            if( len(prevList) > 0 ):
                print " - - - - - Pushing back 'prev' list"
                for mmm in prevList: prevList.append(mmm)

            if( len(s2m[snum]) > 0 ): 
                print " - - - - - Pushing back 's2m'  list"
                for mmm in s2m[snum]: prevList.append(mmm)

            continue


        ## Load Details Information ##
        start = datetime.now()
        mrgList = mergers[MERGERS_MAP_STOM][snum]                                                   # Mergers in this Snapshot
        matched = detailsForMergersAtSnapshot(mergers, dets, mrgList, prevList, nextList, missList,
                                              run, snum, mergDets, verbose=verbose)

        numMatched += matched                                                                       # Track successful matches
        numMissing = len(missList)
        prevList = nextList                                                                         # Pass 'next' list on
        nextList = []                                                                               # Clear for next iteration
        stop = datetime.now()
        if( verbose ): print " - - - - Snapshot Mergers Matched after %s" % (str(stop-start))

        stopOne = datetime.now()
        if( verbose ):
            print " - - - - - %d matched, %d missing After %s / %s" % (numMatched, numMissing, str(stopOne-startOne), str(stopOne-startAll))

    # } ii


    # Add Meta data to merger details
    mergDets[DETAILS_RUN] = run
    mergDets[DETAILS_CREATED] = datetime.datetime.now().ctime()

    return mergDets





def detailsForMergersAtSnapshot(mergers, dets, mrgList, prevList, nextList, missList,
                                run, snap, mergDets, verbose=VERBOSE):

    '''
    Get the Details for a list of mergers for this snapshot.

    Details group 'i' corresponds to those which were written between
    snapshot i and snapshot i+1.  mrgList describes the mergers which
    occured in the same interval.

    + Missing Mergers:
    If a merger isn't found in these Details, add it's index to 'nextList' which
    will be passed to the next iteration (i.e. the next [previous in time] Details
    batch.  'prevList' is that list, that wasn't found in the last iteration.
    Also try to find 'prevList' here, if they are not found, add them to 'missList'

    mergers   : IN, <Mergers>    object containing all merger information
    dets      : IN, <Details>    object containing Details information
    mrgList   : IN, list, <int>  indices of target mergers
    prevList  : IN, list, <int>  list of additional mergers not found last time
    nextList  : INOUT, list, <int> list to search in next (previous in time) snapshot
    missList  : INOUT, list, <int> list of Merger indices not found in pair of snaps
    run       : IN, <int>, illustris run number {1,3}
    snap      : IN, <int>, illustris snapshot number {1,136}
    mergDets  : INOUT, <dict>, dictionary to be filled with results
    verbose   : IN, <bool>, print verbose output
    '''

    if( verbose ): print " - - - detailsForMergersAtSnapshot()"

    NORM_MRG = 0                                                                                    # Mergers from this snapshot
    PREV_MRG = 1                                                                                    # Mergers from previous snapshot

    # Concatenate the new list (mrgList) with the previous (prevList);
    #     make array (mrgTypes) to track which element is which
    if( len(prevList) > 0 ): fullList = np.concatenate( (mrgList, prevList) )
    else: fullList = np.array(mrgList)

    numTarget = len(fullList)
    mrgTypes = np.ones( len(mrgList) + len(prevList), dtype=_LONG )*NORM_MRG                        # Set first group as NORM type
    mrgTypes[len(mrgList):] = PREV_MRG                                                              # Next group as PREV type


    detInds = -1*np.ones([numTarget,2,2], dtype=_LONG)

    if( verbose ): print " - - - - targeting %d/%d mergers" % (numTarget, mergers[MERGERS_NUM])

    ### Iterate Over Target Merger Indices ###
    numGood = 0                                                                                     # Number matched right away
    numNext = 0                                                                                     # Num passed on to next iter
    numSaved = 0                                                                                    # Num from past, now matched
    numMissing = 0                                                                                  # Num from past, unmatched
    numIncomp = 0

    missInBef = 0
    missOutBef = 0
    missOutAft = 0

    for ii, (mtype, mnum) in enumerate(zip(mrgTypes,fullList)):

        # Extract target parameters
        mtime = mergers[MERGERS_TIMES][mnum]
        inid  = mergers[MERGERS_IDS][mnum, IN_BH]
        outid = mergers[MERGERS_IDS][mnum, OUT_BH]

        for IO,tid in enumerate([inid, outid]):
            inds = detailsIndForBHAtTime(tid, mtime, dets)

            for BA in [DETAILS_BEFORE, DETAILS_AFTER]:

                # If there are matches
                if( inds[BA] is not None ):

                    # If this hasn't been matched before, store match
                    if( mergDets[DETAILS_IDS][mnum,IO,BA] < 0 ):
                        detInds[ii,IO,BA] = inds[BA]

                    # If already matched, see if this is better
                    else:
                        oldTime = dets[DETAILS_TIMES][ detInds[ii,IO,BA] ]
                        newTime = dets[DETAILS_TIMES][ inds[BA] ]

                        # For *before* matches, later   is better
                        if(   BA == DETAILS_BEFORE ):
                            if( newTime > oldTime ):
                                detInds[ii,IO,BA] = inds[BA]
                        # For *after* matches,  earlier is better
                        elif( BA == DETAILS_AFTER ):
                            if( newTime < oldTime ):
                                detInds[ii,IO,BA] = inds[BA]

            # } BA

        # } IO


        # If any of expected matches are missing
        if( detInds[ii,IN_BH, DETAILS_BEFORE] < 0 or
            detInds[ii,OUT_BH,DETAILS_BEFORE] < 0 or detInds[ii,OUT_BH,DETAILS_AFTER] < 0 ):

            # If this is from the normal list, add to 'next' list
            if( mtype == NORM_MRG ):
                nextList.append(mnum)
                numNext += 1
            # If this is from the previous list, add to 'missing' list
            elif( mtype == PREV_MRG ):

                if( detInds[ii,IN_BH, DETAILS_BEFORE] < 0 ):
                    print " - - - - - - Snap  %d, Merger %d, Missing IN  BEFORE" % (snap, mnum)
                    missInBef += 1

                if( detInds[ii,OUT_BH,DETAILS_BEFORE] < 0 ):
                    print " - - - - - - Snap  %d, Merger %d, Missing OUT BEFORE" % (snap, mnum)
                    missOutBef += 1

                if( detInds[ii,OUT_BH,DETAILS_AFTER ] < 0 ):
                    print " - - - - - - Snap  %d, Merger %d, Missing OUT AFTER " % (snap, mnum)
                    missOutAft += 1
                    numMissing += 1
                    missList.append(mnum)

                numIncomp += 1
            else:
                raise RuntimeError("Unrecognized type %d in merger list!" % (mtype) )

            continue

        # Track if successful match after previous failure
        if( mtype == PREV_MRG ): numSaved += 1

        numGood += 1

    # } ii

    for IO in [IN_BH, OUT_BH]:
        for BA in [DETAILS_BEFORE, DETAILS_AFTER]:

            inds = np.where( detInds[:,IO,BA] >= 0 )[0]

            if( len(inds) > 0 ):
                for KEY in DETAILS_PHYSICAL_KEYS:
                    mergDets[KEY][fullList[inds],IO,BA] = dets[KEY][detInds[inds,IO,BA]]


    if( verbose ):
        print " - - - - - %d good, %d passed, %d saved, %d missing" % (numGood, numNext, numSaved, numMissing)
        print " - - - - - - %d Incomplete : %d in before,  %d out before,  %d out after (missing)" % (numIncomp, missInBef, missOutBef, missOutAft)


    return numGood



def detailsIndForBHAtTime(bhid, mtime, dets, verbose=False):
    '''
    Get the details indices matching the target BH around a given time.

    Finds all indices of matching IDs in the details entries, then finds the
    nearest entry to the time ``mtime`` both before and after.  Returns `None`
    for no match on that side (before or after).  Also finds the first matching
    entry (in this snapshot).

    Arguments
    ---------
    bhid : long, ID number of target BH
    mtime : scalar, time (scale factor) of desired detail entries
    dets : dict, BHDetails details entries
    verbose : bool (optional : False), print verbose output

    Returns
    -------
    indBef : long, index number of match before given time
    indAft : long, index number of match after  given time
    indFst : long, index number of first found match

    '''

    if( verbose ): print " - - - detailsIndForBH()"

    start = datetime.now()

    ### Find the Details index which matched ID and Nearest in Time ###

    # Find details indices with BH ID match
    detInds = np.where( bhid == dets[DETAILS_IDS] )[0]

    # If there are no matches, return None
    if( len(detInds) == 0 ):
        if( verbose ): print " - - - - No ID matches"
        return None, None

    # Get times for matching details
    detTimes = dets[DETAILS_TIMES][detInds]
    # Get indices to sort details times
    sortInd = np.argsort(detTimes)

    # Get the first match
    if( len(sortInd) > 0 ): indFst = detInds[ sortInd[0] ]
    else:                   indFst = None

    # details *before* merger time
    indBef = np.where( detTimes[sortInd] <  mtime )[0]
    # details *after*  merger time
    indAft = np.where( detTimes[sortInd] >= mtime )[0]


    ### Convert Index to usable ###

    # If there are *before* matches
    if( len(indBef) > 0 ):
        # Get the last sorted time entry BEFORE merger
        indBef = indBef[-1]
        # Get the actual time entry
        indBef = detInds[ sortInd[indBef] ]
    else:
        indBef = None


    # If there are *after* matches
    if( len(indAft) > 0 ):
        # Get the first sorted time entry AFTER merger
        indAft = indAft[0]
        # Get the actual time entry
        indAft = detInds[ sortInd[indAft] ]
    else:
        indAft = None


    stop = datetime.now()

    if( verbose ):
        timeBef = mtime - dets[DETAILS_TIMES][indBef]
        timeAft = dets[DETAILS_TIMES][indAft] - mtime
        print " - - - - Indices After %s : Before %.2e   After %.2e" % (str(stop-start), timeBef, timeAft)


    return indBef, indAft, indFst







if __name__ == "__main__": main()
