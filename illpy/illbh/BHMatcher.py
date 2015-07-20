"""
Routines to match BH Mergers with BH Details entries based on times and IDs.


Functions
---------


MergerDetails Dictionary
------------------------
   The ``MergerDetails`` dictionary contains all of the same parameters as that of the normal
   ``BHDetails`` dictionary, but with more elements.  In particular, every parameter entry in the
   dictionary has shape `[N,2,3]`, where each axis is as follows:
      0: which merger, for ``N`` total mergers
      1: which BH, either ``BH_IN`` or ``BH_OUT``
      2: time of details entry, one of 
         ``BH_FIRST``  the first details entry matching this BH
         ``BH_BEFORE`` details entry immediately before the merger
         ``BH_AFTER``  details entry immediately after  the merger (only exists for ``BH_OUT``)

   { DETAILS_RUN       : <int>, illustris simulation number in {1,3}
     DETAILS_NUM       : <int>, total number of mergers `N`
     DETAILS_FILE      : <str>, name of save file from which mergers were loaded/saved
     DETAILS_CREATED   : <str>, date and time this file was created
     DETAILS_VERSION   : <flt>, version of BHDetails used to create file
   
     DETAILS_IDS       : <uint64>[N,2,3], BH particle ID numbers for each entry
     DETAILS_SCALES    : <flt64> [N,2,3], scale factor at which each entry was written
     DETAILS_MASSES    : <flt64> [N,2,3], BH mass
     DETAILS_MDOTS     : <flt64> [N,2,3], BH Mdot
     DETAILS_RHOS      : <flt64> [N,2,3], ambient mass-density
     DETAILS_CS        : <flt64> [N,2,3], ambient sound-speed
   }


Notes
-----

"""

import os
import sys
import numpy as np

from datetime import datetime

import BHDetails
import BHMergers
from BHConstants import *

from MatchDetails import getDetailIndicesForMergers

from .. import Constants
from .. import AuxFuncs as aux


VERSION = 0.22

FST = BH_FIRST
BEF = BH_BEFORE
AFT = BH_AFTER




def main(run, verbose=True):

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



def loadMergerDetails(run, loadsave=True, mergers=None, verbose=True):

    if( verbose ): print " - - BHMatcher.loadMergerDetails()"

    saveFile = GET_MERGER_DETAILS_FILENAME(run, VERSION)

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
            if( verbose ): print " - - - - File does not exist."
            loadsave = False
                

    ### Re-match Mergers with Details ###
    if( not loadsave ):
        if( verbose ): print " - - - Rematching mergers and details"

        if( mergers is None ): mergers = BHMergers.loadFixedMergers(run)
        if( verbose ): print " - - - - %d Mergers" % (mergers[MERGERS_NUM])
    
        # Get Details for Mergers
        mergerDetails = detailsForMergers(run, mergers, verbose=verbose)
        # Add meta-data
        mergerDetails[DETAILS_RUN]     = run
        mergerDetails[DETAILS_CREATED] = datetime.now().ctime()
        mergerDetails[DETAILS_VERSION] = VERSION
        mergerDetails[DETAILS_FILE]    = saveFile

        # Save merger details
        aux.dictToNPZ(mergerDetails, saveFile, verbose=verbose)


    return mergerDetails




def detailsForMergers(run, mergers, verbose=True):
    """
    Given a set of mergers, retrieve corresponding 'details' entries for BHs.

    Finds the details entries which occur closest to the 'merger' time, both
    before and after it (after only exists for the 'out' BHs).

    Arguments
    ---------
    mergers : dict, data arrays for mergers
    run : int, illustris run number {1,3}
    verbose : bool (optional : ``True``), flag to print verbose output.

    
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
    start = datetime.now()
    for ii,snum in enumerate(snapNumbers):

        ## Load appropriate Details File ##
        dets = BHDetails.loadBHDetails(run, snum, verbose=False)

        # If there are no details here, continue to next iter, but save BHs
        if( dets[DETAILS_NUM] == 0 ): continue

        # Reset which matches are new
        matchNew  = -1*np.ones([numMergers, NUM_BH_TYPES, NUM_BH_TIMES], dtype=LNG)

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
        
        if( verbose ):
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

# checkMatches()


def inferMergerOutMasses(run, mergers=None, mdets=None, verbose=True, debug=False):
    """
    Based on 'merger' and 'details' information, infer the 'out' BH masses at time of mergers.

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

    if( verbose ): print " - - BHMatcher.inferOutMasses()"

    # Load Mergers
    if( mergers is None ):
        if( verbose ): print " - - - Loading Mergers"
        mergers = BHMergers.loadFixedMergers(run, verbose=verbose)

    # Load Merger Details
    if( mdets is None ):
        if( verbose ): print " - - - Loading Merger Details"
        mdets = loadMergerDetails(run, verbose=verbose)


    numMergers = mergers[MERGERS_NUM]
    mass = mdets[DETAILS_MASSES]
    scal = mdets[DETAILS_SCALES]

    if( debug ):
        inds_inn_bef = np.where( mass[:,BH_IN ,BH_BEFORE] <= 0.0 )[0]
        inds_out_bef = np.where( mass[:,BH_OUT,BH_BEFORE] <= 0.0 )[0]
        inds_out_aft = np.where( mass[:,BH_OUT,BH_AFTER ] <= 0.0 )[0]
        print "BHMatcher.inferOutMasses() : %d missing IN  BEFORE" % (len(inds_inn_bef))
        print "BHMatcher.inferOutMasses() : %d missing OUT BEFORE" % (len(inds_out_bef))
        print "BHMatcher.inferOutMasses() : %d missing OUT AFTER " % (len(inds_out_aft))


        
    ## Fix Mass Entries
    #  ----------------
    if( verbose ): print " - - - Inferring 'out' BH masses at merger"
        
    # Details entries just before merger are the best option, default to this
    massOut = np.array( mass[:,BH_OUT,BH_BEFORE] )
    inds = np.where( massOut <= 0.0 )[0]
    if( verbose ): 
        print " - - - - %d/%d Entries missing details 'out' 'before'" % \
            (len(inds), numMergers)

    # If some are missing, use difference from out-after and merger-in
    inds = np.where( massOut <= 0.0 )[0]
    if( len(inds) > 0 ): 
        massOut[inds] = mass[inds,BH_OUT,BH_AFTER] - mergers[MERGERS_MASSES][inds,BH_IN]
        inds = np.where( massOut[inds] > 0.0 )[0]
        print " - - - - %d Missing entries replaced with 'out' 'after' minus merger 'in'" % \
            (len(inds))

    # If some are still missing, use difference from out-after and in-before
    inds = np.where( massOut <= 0.0 )[0]
    if( len(inds) > 0 ): 
        massOut[inds] = mass[inds,BH_OUT,BH_AFTER] - mass[inds,BH_IN,BH_BEFORE]
        inds = np.where( massOut[inds] > 0.0 )[0]
        print " - - - - %d Missing entries replaced with 'out' 'after' minus 'in' 'before'" % \
            (len(inds))

    inds = np.where( massOut <= 0.0 )[0]
    if( verbose ): print " - - - - %d/%d Out masses still invalid" % (len(inds), numMergers)

    return massOut

# inferMergerOutMasses()
