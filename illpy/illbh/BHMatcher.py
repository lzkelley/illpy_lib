"""
Routines to match BH Mergers with BH Details entries based on times and IDs.


"""

import numpy as np

from datetime import datetime

import BHDetails
import BHMergers
from BHConstants import *

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
from MatchDetails import getDetailIndicesForMergers

from .. import Constants
from .. import AuxFuncs as aux


RUN = 3
VERBOSE = True


DF = DETAILS_FIRST
DB = DETAILS_BEFORE
DA = DETAILS_AFTER



###  ===================================  ###
###  =============  MAIN  ==============  ###
###  ===================================  ###



def main(run=RUN, verbose=VERBOSE):

    if( verbose ): print " - BHMatcher.py"

    # Set Basic Parameters
    mergers = BHMergers.loadMergers(run)
    numMergers = mergers[MERGERS_NUM]

    # Get Details for Mergers
    if( verbose ): print " - - Matching Mergers to Details"
    start = datetime.now()
    mergerDetails = detailsForMergers(mergers, run, verbose=verbose)
    stop = datetime.now()
    if( verbose ): print " - - - Mergers matched after %s" % (str(stop-start))

    # Save merger details
    saveFile = 'ill-%d_matched-merger-details_new.npz' % (run)
    aux.saveDictNPZ(mergerDetails, saveFile, verbose=verbose)

    # Check Matches
    if( verbose ): print " - - Checking Matches"
    start = datetime.now()
    checkMatches(mergerDetails, mergers)
    stop = datetime.now()
    if( verbose ): print " - - - Checked after %s" % (str(stop-start))


    if( verbose ): print " - Done"

    return





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

    # Search for all Mergers in each snapshot (easier, though less efficient)
    targets = np.arange(numMergers)

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


    matchInds  = -1  *np.ones([numMergers, NUM_BH_TYPES, NUM_BH_TIMES], dtype=np.int64)
    matchTimes = -1.0*np.ones([numMergers, NUM_BH_TYPES, NUM_BH_TIMES], dtype=np.float64)

    ### Create Dictionary To Store Results ###

    mergDets = {}
    # Initialize Dictionary to Invalid entries (-1)
    for KEY in DETAILS_PHYSICAL_KEYS:
        if( KEY == DETAILS_IDS ): 
            mergDets[KEY] = -1 * np.ones([numMergers, NUM_BH_TYPES, NUM_BH_TIMES], dtype=_LONG  )
        else:
            mergDets[KEY] = -1.0*np.ones([numMergers, NUM_BH_TYPES, NUM_BH_TIMES], dtype=_DOUBLE)


    ### Iterate over Target Snapshots ###
    if( verbose ): print " - - - Iterating over %d snapshots" % (numSnapsHere)
    startAll = datetime.now()

    #for ii,snum in enumerate(range(95,97)):
    for ii,snum in enumerate(snapNumbers):
        startOne = datetime.now()
        if( verbose ): print " - - - Snapshot %d/%d   # %d" % (ii, numSnapsHere, snum)

        ## Load appropriate Details File ##
        start = datetime.now()
        dets = BHDetails.loadBHDetails_NPZ(run, snum)
        stop = datetime.now()
        if( verbose ): print " - - - - Details loaded after %s" % (str(stop-start))

        # If there are no details here, continue to next iter, but save BHs
        if( dets[DETAILS_NUM] == 0 ):
            if( verbose ): print " - - - - No details for snapshot %d" % (snum)
            continue

        # Reset which matches are new
        matchNew  = -1*np.ones([numMergers, NUM_BH_TYPES, NUM_BH_TIMES], dtype=np.int32)

        ## Load Details Information ##
        start = datetime.now()
        numMatches = getDetailIndicesForMergers(targets, mergers[MERGERS_IDS], 
                                                mergers[MERGERS_TIMES], matchInds,
                                                matchTimes, matchNew,
                                                dets[DETAILS_IDS], dets[DETAILS_TIMES])

        stop = datetime.now()
        if( verbose ): print " - - - - Snapshot Mergers Matched after %s" % (str(stop-start))

        ### Store Matches Data ###

        start = datetime.now()
        # iterate over in/out BHs
        for BH in [IN_BH,OUT_BH]:
            # Iterate over match times
            for FBA in [DF, DB, DA]:

                # Find valid matches
                inds = np.where( matchNew[:, BH, FBA] >= 0 )[0]
                if( len(inds) > 0 ):

                    # Store Each Parameter
                    for KEY in DETAILS_PHYSICAL_KEYS:
                        mergDets[KEY][inds, BH, FBA] = dets[KEY][ matchInds[inds, BH, FBA] ]

            # } FBA

        # } BH

        stop = datetime.now()
        if( verbose ): print " - - - - Merger-Details stored after %s" % (str(stop-start))

        stopOne = datetime.now()
        if( verbose ):
            print " - - - - - %d matched After %s / %s" % (numMatches, str(stopOne-startOne), str(stopOne-startAll))

    # } ii

    # Add Meta data to merger details
    mergDets[DETAILS_RUN] = run
    mergDets[DETAILS_CREATED] = datetime.now().ctime()

    return mergDets

# detailsForMergers()




def checkMatches(matches, mergers):
    """
    Perform basic diagnostic on merger-detail matches.

    Arguments
    ---------


    """

    RAND_NUMS = 4

    numMergers = mergers[MERGERS_NUM]
    bh_str = { IN_BH : "In ", OUT_BH : "Out" }
    time_str = { DF : "First ", DB : "Before", DA : "After " }

    good = np.zeros([numMergers, NUM_BH_TYPES, NUM_BH_TIMES], dtype=bool)
    
    ### Count Seemingly Successful Matches ###
    print " - - - Number of Matches (%6d Mergers) :" % (numMergers)
    print "             First  Before After"

    # Iterate over both BHs
    for BH in [IN_BH, OUT_BH]:

        num_str = ""

        # Iterate over match times
        for FBA in [DF, DB, DA]:

            inds = np.where( matches[DETAILS_TIMES][:,BH,FBA] >= 0.0 )[0]
            good[inds, BH, FBA] = True
            num_str += "%5d  " % (len(inds))
    

        print "       %s : %s" % ( bh_str[BH], num_str )


    ### Count Combinations of Matches ###
    print "\n - - - Number of Match Combinations :"

    # All Times
    inds = np.where( np.sum(good[:,OUT_BH,:], axis=1) == NUM_BH_TIMES )[0]
    print " - - - - All            : %5d" % (len(inds))
    
    # Before and After
    inds = np.where( good[:,OUT_BH,DB] & good[:,OUT_BH,DA] )[0]
    print " - - - - Before & After : %5d" % (len(inds))

    # First and Before
    inds = np.where( good[:,OUT_BH,DF] & good[:,OUT_BH,DB] )[0]
    print " - - - - First & Before : %5d" % (len(inds))
    

    print ""
    inds = np.where( (good[:,OUT_BH,DF]  == True ) & 
                     (good[:,OUT_BH,DB] == False)    )[0]

    print " - - - Number of First without Before : %5d" % (len(inds))
    if( len(inds) > 0 ):
        print "\t\t         First      Before    ( Merger  )   After"

        if( len(inds) <= RAND_NUMS ): sel = np.arange(len(inds))
        else:                         sel = np.random.randint(0, len(inds), size=RAND_NUMS)         

        for ii in sel:
            sel_id = inds[ii]
            tmat = matches[DETAILS_TIMES][sel_id,OUT_BH,:]
            tt = mergers[MERGERS_TIMES][sel_id]
            print "\t\t%5d : %+f  %+f  (%+f)  %+f" % (sel_id, tmat[DF], tmat[DB], tt, tmat[DA])



    ### Check ID Matches ###
    print "\n - - - Number of BAD ID Matches:"
    print "             First  Before After"
    # Iterate over both BHs
    for BH in [IN_BH, OUT_BH]:
        num_str = ""
        eg = None
        # Iterate over match times
        for FBA in [DF, DB, DA]:

            dids = matches[DETAILS_IDS][:,BH,FBA]
            mids = mergers[MERGERS_IDS][:,BH]
            inds = np.where( (good[:,BH,FBA]) & (dids != mids) )[0]
            
            num_str += "%5d  " % (len(inds))
            if( len(inds) > 0 ): eg = inds[0]


        print "       %s : %s" % ( bh_str[BH], num_str )
        if( eg is not None ):
            tmat = matches[DETAILS_TIMES][sel_id,OUT_BH,:]
            tt = mergers[MERGERS_TIMES][sel_id]
            print "\t\t%5d : %+f  %+f  (%+f)  %+f" % (eg, tmat[DF], tmat[DB], tt, tmat[DA])


    ### Check Time Matches ###
    print "\n - - - Number of BAD time Matches:"
    # Iterate over both BHs
    for BH in [IN_BH, OUT_BH]:
        num_str = ""
        eg = None
        # Iterate over match times
        for FBA in [DF, DB, DA]:

            dt = matches[DETAILS_TIMES][:,BH,FBA]
            mt = mergers[MERGERS_TIMES]

            if( FBA in [DF,DB] ): inds = np.where( (good[:,BH,FBA]) & (dt >= mt) )[0]
            else:                 inds = np.where( (good[:,BH,FBA]) & (dt <  mt) )[0]

            num_str += "%5d  " % (len(inds))
            if( len(inds) > 0 ): eg = inds[0]
            
        print "       %s : %s" % ( bh_str[BH], num_str )
        if( eg is not None ):
            tmat = matches[DETAILS_TIMES][eg,OUT_BH,:]
            tt = mergers[MERGERS_TIMES][eg]
            print "\t\t%5d : %+f  %+f  (%+f)  %+f" % (eg, tmat[DF], tmat[DB], tt, tmat[DA])



    return





if __name__ == "__main__": main()
