"""
Routines to match BH Mergers with BH Details entries based on times and IDs.


"""

import os
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



FST = BH_FIRST
BEF = BH_BEFORE
AFT = BH_AFTER



###  ===================================  ###
###  =============  MAIN  ==============  ###
###  ===================================  ###



def main(run, verbose=VERBOSE):

    if( verbose ): print " - BHMatcher.py"
    start = datetime.now()

    # Load Mergers
    mergers = BHMergers.loadFixedMergers(run)
    numMergers = mergers[MERGERS_NUM]
    if( verbose ): print " - - %d Mergers loaded" % (numMergers)

    # Load Matching Details
    mergerDetails = loadMergerDetails(run, mergers=mergers, verbose=verbose)
    
    # Check Matches
    if( verbose ): print " - - Checking Matches"
    checkMatches(mergerDetails, mergers)

    stop = datetime.now()
    if( verbose ): print " - Done after %s" % (str(stop-start))

    return



def loadMergerDetails(run, loadsave=True, mergers=None, verbose=VERBOSE):

    if( verbose ): print " - - BHMatcher.loadMergerDetails()"

    saveFile = GET_MERGER_DETAILS_FILENAME(run)

    ### Try to Load Existing Merger Details ###
    if( loadsave ):
        if( verbose ): print " - - - Loading save from '%s'" % (saveFile)

        # Make sure file exists
        if( os.path.exists(saveFile) ):
            mergerDetails = aux.npzToDict(saveFile)
            if( verbose ): print " - - - Loaded Merger Details"

            # Make sure versions match
            loadVers = mergerDetails[DETAILS_VERSION]
            if( loadVers != VERSION ):
                print "BHMatcher.loadMergerDetails() : version %s from '%s'" % \
                    (str(loadVers), saveFile)
                print "BHMatcher.loadMergerDetails() : Current version %s" % (str(VERSION))
                print "BHMatcher.loadMergerDetails() : Rematching !!!"
                loadsave = False

        else:
            loadsave = False
                

    ### Re-match Mergers with Details ###
    if( not loadsave ):
        if( verbose ): print " - - - Rematching mergers and details"

        if( mergers is None ): mergers = BHMergers.loadFixedMergers(run)
        if( verbose ): print " - - - - %d Mergers" % (mergers[MERGERS_NUM])
    
        # Get Details for Mergers
        mergerDetails = detailsForMergers(run, mergers, verbose=verbose)
        # Add meta-data
        mergDets[DETAILS_RUN]     = run
        mergDets[DETAILS_CREATED] = datetime.now().ctime()
        mergDets[DETAILS_VERSION] = VERSION
        mergDets[DETAILS_FILE]    = saveFile

        # Save merger details
        aux.dictToNPZ(mergerDetails, saveFile, verbose=verbose)


    return mergerDetails




def detailsForMergers(run, mergers, verbose=VERBOSE):
    """
    Given a set of mergers, retrieve corresponding 'details' entries for BHs.

    Finds the details entries which occur closest to the 'merger' time, both
    before and after it (after only exists for the 'out' BHs).

    Arguments
    ---------
    mergers : dict, data arrays for mergers
    run : int, illustris run number {1,3}
    verbose : bool (optional : ``VERBOSE``), flag to print verbose output.

    
    Returns
    -------
    mergDets : dict, data arrays corresponding to each 'merger' BHs

    """

    if( verbose ): print " - - BHMatcher.detailsForMergers()"

    numMergers = mergers[MERGERS_NUM]
    numSnaps   = Constants.NUM_SNAPS

    # Search for all Mergers in each snapshot (easier, though less efficient)
    targets = np.arange(numMergers)

    snapNumbers = reversed(xrange(numSnaps))

    matchInds  = -1*np.ones([numMergers, NUM_BH_TYPES, NUM_BH_TIMES], dtype=LNG)
    matchTimes = -1*np.ones([numMergers, NUM_BH_TYPES, NUM_BH_TIMES], dtype=DBL)


    ### Create Dictionary To Store Results ###

    mergDets = {}

    # Initialize Dictionary to Invalid entries (-1)
    for KEY in DETAILS_PHYSICAL_KEYS:
        if( KEY == DETAILS_IDS ): useType = ULNG
        else:                     useType = DBL
        mergDets[KEY] = -1 * np.ones([numMergers, NUM_BH_TYPES, NUM_BH_TIMES], dtype=useType)



    ### Iterate over all Snapshots ###
    startAll = datetime.now()
    for ii,snum in enumerate(snapNumbers):
        startOne = datetime.now()

        ## Load appropriate Details File ##
        dets = BHDetails.loadBHDetails(run, snum)

        # If there are no details here, continue to next iter, but save BHs
        if( dets[DETAILS_NUM] == 0 ): continue

        # Reset which matches are new
        matchNew  = -1*np.ones([numMergers, NUM_BH_TYPES, NUM_BH_TIMES], dtype=LNG)

        print type(mergers[MERGERS_IDS][0,0])
        print type(dets[DETAILS_IDS][0])


        ## Load Details Information ##
        numMatches = getDetailIndicesForMergers(targets, mergers[MERGERS_IDS], 
                                                mergers[MERGERS_SCALES], matchInds,
                                                matchTimes, matchNew,
                                                dets[DETAILS_IDS], dets[DETAILS_SCALES])


        ### Store Matches Data ###

        # iterate over in/out BHs
        for BH in [BH_IN,BH_OUT]:
            # Iterate over match times
            for FBA in [FST, BEF, AFT]:

                # Find valid matches
                inds = np.where( matchNew[:, BH, FBA] >= 0 )[0]
                if( len(inds) > 0 ):

                    # Store Each Parameter
                    for KEY in DETAILS_PHYSICAL_KEYS:
                        mergDets[KEY][inds, BH, FBA] = dets[KEY][ matchInds[inds, BH, FBA] ]

            # } FBA

        # } BH

        
        if( progress ):
            now = datetime.now()
            dur = now-start
            statStr = aux.statusString(ii+1, numSnaps, dur)
            sys.stdout.write('\r - - - %s' % (statStr))
            sys.stdout.flush()
            if( ii+1 == numSnaps ): sys.stdout.write('\n')


    # } ii

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
    bh_str = { BH_IN : "In ", BH_OUT : "Out" }
    time_str = { FST : "First ", BEF : "Before", AFT : "After " }

    good = np.zeros([numMergers, NUM_BH_TYPES, NUM_BH_TIMES], dtype=bool)
    
    ### Count Seemingly Successful Matches ###
    print " - - - Number of Matches (%6d Mergers) :" % (numMergers)
    print "             First  Before After"

    # Iterate over both BHs
    for BH in [BH_IN, BH_OUT]:

        num_str = ""

        # Iterate over match times
        for FBA in [FST, BEF, AFT]:

            inds = np.where( matches[DETAILS_SCALES][:,BH,FBA] >= 0.0 )[0]
            good[inds, BH, FBA] = True
            num_str += "%5d  " % (len(inds))
    

        print "       %s : %s" % ( bh_str[BH], num_str )


    ### Count Combinations of Matches ###
    print "\n - - - Number of Match Combinations :"

    # All Times
    inds = np.where( np.sum(good[:,BH_OUT,:], axis=1) == NUM_BH_TIMES )[0]
    print " - - - - All            : %5d" % (len(inds))
    
    # Before and After
    inds = np.where( good[:,BH_OUT,BEF] & good[:,BH_OUT,AFT] )[0]
    print " - - - - Before & After : %5d" % (len(inds))

    # First and Before
    inds = np.where( good[:,BH_OUT,FST] & good[:,BH_OUT,BEF] )[0]
    print " - - - - First & Before : %5d" % (len(inds))
    

    print ""
    inds = np.where( (good[:,BH_OUT,FST]  == True ) & 
                     (good[:,BH_OUT,BEF] == False)    )[0]

    print " - - - Number of First without Before : %5d" % (len(inds))
    if( len(inds) > 0 ):
        print "\t\t         First      Before    ( Merger  )   After"

        if( len(inds) <= RAND_NUMS ): sel = np.arange(len(inds))
        else:                         sel = np.random.randint(0, len(inds), size=RAND_NUMS)         

        for ii in sel:
            sel_id = inds[ii]
            tmat = matches[DETAILS_SCALES][sel_id,BH_OUT,:]
            tt = mergers[MERGERS_SCALES][sel_id]
            print "\t\t%5d : %+f  %+f  (%+f)  %+f" % (sel_id, tmat[FST], tmat[BEF], tt, tmat[AFT])



    ### Check ID Matches ###
    print "\n - - - Number of BAD ID Matches:"
    print "             First  Before After"
    # Iterate over both BHs
    for BH in [BH_IN, BH_OUT]:
        num_str = ""
        eg = None
        # Iterate over match times
        for FBA in [FST, BEF, AFT]:

            dids = matches[DETAILS_IDS][:,BH,FBA]
            mids = mergers[MERGERS_IDS][:,BH]
            inds = np.where( (good[:,BH,FBA]) & (dids != mids) )[0]
            
            num_str += "%5d  " % (len(inds))
            if( len(inds) > 0 ): eg = inds[0]


        print "       %s : %s" % ( bh_str[BH], num_str )
        if( eg is not None ):
            tmat = matches[DETAILS_SCALES][sel_id,BH_OUT,:]
            tt = mergers[MERGERS_SCALES][sel_id]
            print "\t\t%5d : %+f  %+f  (%+f)  %+f" % (eg, tmat[FST], tmat[BEF], tt, tmat[AFT])


    ### Check Time Matches ###
    print "\n - - - Number of BAD time Matches:"
    # Iterate over both BHs
    for BH in [BH_IN, BH_OUT]:
        num_str = ""
        eg = None
        # Iterate over match times
        for FBA in [FST, BEF, AFT]:

            dt = matches[DETAILS_SCALES][:,BH,FBA]
            mt = mergers[MERGERS_SCALES]

            if( FBA in [FST,BEF] ): inds = np.where( (good[:,BH,FBA]) & (dt >= mt) )[0]
            else:                   inds = np.where( (good[:,BH,FBA]) & (dt <  mt) )[0]

            num_str += "%5d  " % (len(inds))
            if( len(inds) > 0 ): eg = inds[0]
            
        print "       %s : %s" % ( bh_str[BH], num_str )
        if( eg is not None ):
            tmat = matches[DETAILS_SCALES][eg,BH_OUT,:]
            tt = mergers[MERGERS_SCALES][eg]
            print "\t\t%5d : %+f  %+f  (%+f)  %+f" % (eg, tmat[FST], tmat[BEF], tt, tmat[AFT])



    return





if __name__ == "__main__": main()
