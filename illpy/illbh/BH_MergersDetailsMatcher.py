# ==================================================================================================
# BH_MergersDetailsMatcher.py
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


RUN = 3
VERBOSE = True



###  ===================================  ###
###  =============  MAIN  ==============  ###
###  ===================================  ###



def main(run=RUN, verbose=VERBOSE):

    if( verbose ): print " - BH_MergersDetailsMatcher.py"

    ### Set Basic Parameters ###
    mergers = BHMergers.loadMergers(run)

    ###  Get Details for Mergers and Save  ###
    if( verbose ): print " - - Matching Mergers to Details"
    start = datetime.now()
    detailsForMergers(mergers, mapM2S, mapS2M, run)
    stop = datetime.now()
    if( verbose ): print " - - - Mergers matched after %s" % (str(stop-start))


    if( verbose ): print " - Done"

    return




###  ================================================  ###
###  ===============  OTHER FUNCTIONS  ==============  ###
###  ================================================  ###


def detailsForMergers(mergers, m2s, s2m, run, snapNums=None, verbose=VERBOSE):
    
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
    #     Row for each merger; Column for each BH In merger [0-in, 1-out]
    ids  = np.zeros([numMergers,2], dtype=_LONG)
    mass = np.zeros([numMergers,2], dtype=_DOUBLE)
    mdot = np.zeros([numMergers,2], dtype=_DOUBLE)
    rho  = np.zeros([numMergers,2], dtype=_DOUBLE)
    cs   = np.zeros([numMergers,2], dtype=_DOUBLE)

    # Maintain a list of Mergers which could not be matched to Details
    missingMergers = []

    ### Iterate over Target Snapshots ###
    if( verbose ): print " - - Iterating over %d snapshots" % (numSnapsHere)
    startAll = datetime.now()
    numMatched = 0
    lastList = []                                                                                   # These are failed matches, passed on
    nextList = []                                                                                   # These will become 'lastList'

    for ii,snum in enumerate(snapNumbers):
        startOne = datetime.now()
        if( verbose ): print " - - - Snapshot %d/%d" % (ii, numSnapsHere)

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
        if( len(dets) == 0 ): continue

        
        ## Load Details Information ##
        start = datetime.now()
        matched = detailsForMergersAtSnapshot(mergers, s2m[snum], lastList, dets, 
                                              nextList, missingMergers, run, snum,
                                              ids, mass, mdot, rho, cs, verbose=verbose)
                           
        numMatched += matched                                                                       # Track successful matches
        numMissing = len(missingMergers)                                                            
        lastList = nextList                                                                         # Pass 'next' list on
        nextList = []                                                                               # Clear for next iteration
        stop = datetime.now()
        if( verbose ): print " - - - - Snapshot Mergers Matched retreived after %s" % (str(stop-start))
        
        stopOne = datetime.now()
        if( verbose ): 
            print " - - - - - %d matched, %d missing After %s / %s" % (numMatched, numMissing, str(stopOne-startOne), str(stopOne-startAll))


    # } ii

    '''
    # Create save file name
    saveFile = 'ill-%d_matched-merger-details.npz' % (run)

    # Organize data into dictionary
    mdict = { 
        # Info from Mergers
        'time':mergers.time, 'm_in_id':mergers.in_id, 'm_in_mass':mergers.in_mass, 
        'm_out_id':mergers.out_id, 'm_out_mass':mergers.out_mass, 
        # Info from Details
        'd_in_id':ids[:,0], 'd_out_id':ids[:,1], 
        'd_in_mass':mass[:,0], 'd_out_mass':mass[:,1], 
        'd_in_mdot':mdot[:,0], 'd_out_mdot':mdot[:,1], 
        'd_in_rho':rho[:,0], 'd_out_rho':rho[:,1], 
        'd_in_cs':cs[:,0], 'd_out_cs':cs[:,1]
        }
    
    
    # Save data
    np.savez( saveFile, **mdict )
    '''

    return





def detailsForMergersAtSnapshot(mergers, mrgList, prevList, dets,
                                nextList, missList, run, snap,
                                ids, mass, mdot, rho, cs, verbose=VERBOS):

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

    if( verbose ): print " - - detailsForMergersAtSnapshot()"

    NORM_MRG = 0
    PREV_MRG = 1
    
    # Concatenate the new list (mrgList) with the previous (prevList);
    #     make array (mrgTypes) to track which element is which
    fullList = np.concatenate( (mrgList, prevList) )
    numTarget = len(fullList)
    mrgTypes = np.ones( len(mrgList) + len(prevList), dtype=_LONG )*NORM_MRG                        # Set first group as NORM type
    mrgTypes[len(mrgList):] = PREV_MRG                                                              # Next group as PREV type
    
    if( verbose ): print " - - - targeting %d/%d mergers" % (numTarget, mergers[MERGERS_NUM])

    ### Iterate Over Target Merger Indices ###
    numGood = 0                                                                                     # Number matched right away
    numNext = 0                                                                                     # Num passed on to next iter
    numSaved = 0                                                                                    # Num from past, now matched
    numMissing = 0                                                                                  # Num from past, unmatched
    for ii, (mtype, mnum) in enumerate(zip(mrgTypes,fullList)):

        # Extract target parameters
        '''
        mtime = mergers.time[mnum]
        inid  = mergers.in_id[mnum]
        outid = mergers.out_id[mnum]
        '''
        mtime = mergers[MERGERS_TIMES][mnum]
        inid  = mergers[MERGERS_IDS][mnum, IN_BH]
        outid = mergers[MERGERS_IDS][mnum, OUT_BH]


        for jj,tid in enumerate([inid, outid]):
            #dargs  = detailsForBH(tid,  mtime, dets)
            dargs  = detailsForBH(tid, run, snap, details=dets, side=None, verbose=verbose)

            # If there are no matches
            if( dargs == None ): 
                # If this is from the normal list, add to 'next' list
                if( mtype == NORM_MRG ): 
                    nextList.append(mnum)
                    numNext += 1
                # If this is from the previous list, add to 'missing' list
                elif( mtype == PREV_MRG ): 
                    missList.append(mnum)
                    numMissing += 1
                else: 
                    raise RuntimeError("Unrecognized type %d in merger list!" % (mtype) )

                continue

            # Track if successful match after previous failure
            if( mtype == PREV_MRG ): numSaved += 1

            numGood += 1
            ids[mnum,jj] = tid
            mass[mnum,jj] = dargs[Details.DETAIL_MASS]
            mdot[mnum,jj] = dargs[Details.DETAIL_MDOT]
            rho[mnum,jj] = dargs[Details.DETAIL_RHO]
            cs[mnum,jj] = dargs[Details.DETAIL_CS]


    if( log ): 
        log.log("%d good, %d passed, %d saved, %d missing." % 
                (numGood, numNext, numSaved, numMissing), 1 )



    return numGood
    

'''
def detailsForBH(bhid, mtime, dets, log=None):

    if( log ): log.log("detailsForBH()", 1)

    start = datetime.now()

    ### Find the Details index which matched ID and Nearest in Time ###

    # Find details indices with BH ID match
    detInds = np.where( bhid == dets.id )[0]

    # If there are no matches, return None
    if( len(detInds) == 0 ): 
        if( log ): log.log("No ID matches")
        return None

    # Get times for matching details
    detTimes = dets.time[ detInds ]
    # Get indices to sort details times
    sortInd = np.argsort(detTimes)
    # Find last detail before merger time
    lastInd = np.where( detTimes[sortInd] <= mtime )[0]

    # If there are no matches, return None
    if( len(lastInd) == 0 ): 
        if( log ): log.log("No time matches")
        return None

    ### Convert Index to usable ###

    # Get the last sorted time entry
    lastInd = lastInd[-1]
    # Get the actual time entry
    lastInd = sortInd[lastInd]
    # Get the overall index
    detInd = detInds[lastInd]

    stop = datetime.now()

    if( log ): 
        log.log("Matched time offset = %.2e; After %s" % 
                (mtime-dets.time[detInd], str(stop-start)))

    return dets[detInd]
'''
    





if __name__ == "__main__": main()
