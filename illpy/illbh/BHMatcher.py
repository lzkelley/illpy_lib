# ==================================================================================================
# BHMatcher.py
# ---------------------------
#
# This code takes in merger (from 'blackhole_mergers_<N>.txt') information in a 'Mergers' object,
# and matches it with details (from 'blackhole_details_<N>.txt') information in a 'Details' object.
#
#
#
# Contains
# --------
# + detailsForMergersAtSnapshot() :
#
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================


### Builtin Modules ###
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

    ### Set Basic Parameters ###
    mergers = BHMergers.loadMergers(run)

    ###  Get Details for Mergers and Save  ###
    if( verbose ): print " - - Matching Mergers to Details"
    start = datetime.now()
    mergerDetails = detailsForMergers(mergers, run, verbose=verbose)
    stop = datetime.now()
    if( verbose ): print " - - - Mergers matched after %s" % (str(stop-start))

    # Save merger details
    saveFile = 'ill-%d_matched-merger-details_raw.npz' % (run)
    aux.saveDictNPZ(mergerDetails, saveFile, verbose=verbose)

    if( verbose ): print " - Done"

    return




###  ================================================  ###
###  ===============  OTHER FUNCTIONS  ==============  ###
###  ================================================  ###


def detailsForMergers(mergers, run, snapNums=None, verbose=VERBOSE):

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
    ids   = -1*np.ones([numMergers,2,2], dtype=_LONG)
    times = -1*np.ones([numMergers,2,2], dtype=_DOUBLE)
    mass  = -1*np.ones([numMergers,2,2], dtype=_DOUBLE)
    mdot  = -1*np.ones([numMergers,2,2], dtype=_DOUBLE)
    rho   = -1*np.ones([numMergers,2,2], dtype=_DOUBLE)
    cs    = -1*np.ones([numMergers,2,2], dtype=_DOUBLE)

    mergDets = {}

    for KEY in DETAIL_PHYSICAL_KEYS:
        if( KEY == DETAIL_IDS ): mergDets[KEY] = -1 * np.ones([numMergers,2,2], dtype=_LONG)
        else:                    mergDets[KEY] = -1.0*np.ones([numMergers,2,2], dtype=_DOUBLE)


    # Maintain a list of Mergers which could not be matched to Details
    missingMergers = []

    ### Iterate over Target Snapshots ###
    if( verbose ): print " - - - Iterating over %d snapshots" % (numSnapsHere)
    startAll = datetime.now()
    numMatched = 0
    lastList = []                                                                                   # These are failed matches, passed on
    nextList = []                                                                                   # These will become 'lastList'

    for ii,snum in enumerate(snapNumbers):
        startOne = datetime.now()
        if( verbose ): print " - - - Snapshot %d/%d   # %d" % (ii, numSnapsHere, snum)

        s2m = mergers[MERGERS_MAP_STOM]

        # If there are no mergers to match, continue to next iteration
        if( len(s2m[snum]) == 0 and len(lastList) == 0 ):
            if( verbose ): print " - - - - No mergers for snapshot %d, or carried over" % (snum)
            continue

        ## Load appropriate Details File ##
        start = datetime.now()
        #dets = aux.loadBHDetails(run, snum, workDir, log=log)
        dets = BHDetails.loadBHDetails_NPZ(run, snum)
        stop = datetime.now()
        if( verbose ): print " - - - - Details loaded after %s" % (str(stop-start))

        # If there are no details here, continue to next iteration
        if( dets[DETAIL_NUM] == 0 ):
            if( verbose ): print " - - - - No details for snapshot %d" % (snum)
            continue


        ## Load Details Information ##
        start = datetime.now()
        matched = detailsForMergersAtSnapshot(mergers, mergers[MERGERS_MAP_STOM][snum], lastList, dets,
                                              nextList, missingMergers, run, snum, mergDets, verbose=verbose)

        numMatched += matched                                                                       # Track successful matches
        numMissing = len(missingMergers)
        lastList = nextList                                                                         # Pass 'next' list on
        nextList = []                                                                               # Clear for next iteration
        stop = datetime.now()
        if( verbose ): print " - - - - Snapshot Mergers Matched after %s" % (str(stop-start))

        stopOne = datetime.now()
        if( verbose ):
            print " - - - - - %d matched, %d missing After %s / %s" % (numMatched, numMissing, str(stopOne-startOne), str(stopOne-startAll))



    # } ii

    # Add Meta data to merger details
    mergDets[DETAIL_RUN] = run
    mergDets[DETAIL_SNAP] = snap
    mergDets[DETAIL_CREATED] = datetime.datetime.now().ctime()

    return mergDets





def detailsForMergersAtSnapshot(mergers, mrgList, prevList, dets,
                                nextList, missList, run, snap,
                                mergDets, verbose=VERBOSE):

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
    mrgList   : IN, list, <int>  indices of target mergers
    prevList  : IN, list, <int>  list of additional mergers not found last time
    dets      : IN, <Details>    object containing Details information
    nextList  : INOUT, list, <int> list to search in next (previous in time) snapshot
    missList  : INOUT, list, <int> list of Merger indices not found in pair of snaps
    mass      : INOUT, ndarray[N,2] <DBL> array to hold masses from Details
    mdot      : INOUT, ndarray[N,2] <DBL> array to hold mdot
    rho       : INOUT, ndarray[N,2] <DBL> array to hold density
    cs        : INOUT, ndarray[N,2] <DBL> array to hold sound speed

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

            for BA in [DETAIL_BEFORE, DETAIL_AFTER]:

                # If there are matches
                if( inds[BA] is not None ):

                    # If this hasn't been matched before, store match
                    if( mergDets[DETAIL_IDS][mnum,IO,BA] < 0 ):
                        detInds[ii,IO,BA] = inds[BA]

                    # If already matched, see if this is better
                    else:
                        oldTime = dets[DETAIL_TIMES][ detInds[ii,IO,BA] ]
                        newTime = dets[DETAIL_TIMES][ inds[BA] ]

                        # For *before* matches, later   is better
                        if(   BA == DETAIL_BEFORE ):
                            if( newTime > oldTime ):
                                detInds[ii,IO,BA] = inds[BA]
                        # For *after* matches,  earlier is better
                        elif( BA == DETAIL_AFTER ):
                            if( newTime < oldTime ):
                                detInds[ii,IO,BA] = inds[BA]

            # } BA

        # } IO


        # If any of expected matches are missing
        if( detInds[ii,IN_BH, DETAIL_BEFORE] < 0 or
            detInds[ii,OUT_BH,DETAIL_BEFORE] < 0 or detInds[ii,OUT_BH,DETAIL_AFTER] < 0 ):

            # If this is from the normal list, add to 'next' list
            if( mtype == NORM_MRG ):
                nextList.append(mnum)
                numNext += 1
            # If this is from the previous list, add to 'missing' list
            elif( mtype == PREV_MRG ):

                if( detInds[ii,IN_BH, DETAIL_BEFORE] < 0 ):
                    print " - - - - - - Snap  %d, Merger %d, Missing IN  BEFORE" % (snap, mnum)
                    missInBef += 1

                if( detInds[ii,OUT_BH,DETAIL_BEFORE] < 0 ):
                    print " - - - - - - Snap  %d, Merger %d, Missing OUT BEFORE" % (snap, mnum)
                    missOutBef += 1

                if( detInds[ii,OUT_BH,DETAIL_AFTER ] < 0 ):
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
        for BA in [DETAIL_BEFORE, DETAIL_AFTER]:

            inds = np.where( detInds[:,IO,BA] >= 0 )[0]

            if( len(inds) > 0 ):
                for KEY in DETAIL_PHYSICAL_KEYS:
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
    for no match on that side (before or after).

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

    '''

    if( verbose ): print " - - - detailsIndForBH()"

    start = datetime.now()

    ### Find the Details index which matched ID and Nearest in Time ###

    # Find details indices with BH ID match
    detInds = np.where( bhid == dets[DETAIL_IDS] )[0]

    # If there are no matches, return None
    if( len(detInds) == 0 ):
        if( verbose ): print " - - - - No ID matches"
        return None, None

    # Get times for matching details
    detTimes = dets[DETAIL_TIMES][detInds]
    # Get indices to sort details times
    sortInd = np.argsort(detTimes)

    # details *before* merger time
    indBef = np.where( detTimes[sortInd] <  mtime )[0]
    # details *after*  merger time
    indAft = np.where( detTimes[sortInd] >= mtime )[0]


    ### Convert Index to usable ###

    # If there are *before* matches
    if( len(indBef) > 0 ):
        # Get the last sorted time entry
        indBef = indBef[-1]
        # Get the actual time entry
        indBef = detInds[ sortInd[indBef] ]
    else:
        indBef = None


    # If there are *after* matches
    if( len(indAft) > 0 ):
        # Get the last sorted time entry
        indAft = indAft[-1]
        # Get the actual time entry
        indAft = detInds[ sortInd[indAft] ]
    else:
        indAft = None


    stop = datetime.now()

    if( verbose ):
        timeBef = mtime - dets[DETAILS_TIMES][indBef]
        timeAft = dets[DETAILS_TIMES][indAft] - mtime
        print " - - - - Indices After %s : Before %.2e   After %.2e" % (str(stop-start), timeBef, timeAft)


    return indBef, indAft







if __name__ == "__main__": main()
