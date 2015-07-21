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
      1: which BH, either ``BH_TYPE.IN`` or ``BH_TYPE.OUT``
      2: time of details entry, one of 
         ``BH_TIME.FIRST``  the first details entry matching this BH
         ``BH_TIME.BEFORE`` details entry immediately before the merger
         ``BH_TIME.AFTER``  details entry immediately after  the merger (only for ``BH_TYPE.OUT``)

   { DETAILS.RUN       : <int>, illustris simulation number in {1,3}
     DETAILS.NUM       : <int>, total number of mergers `N`
     DETAILS.FILE      : <str>, name of save file from which mergers were loaded/saved
     DETAILS.CREATED   : <str>, date and time this file was created
     DETAILS.VERSION   : <flt>, version of BHDetails used to create file
   
     DETAILS.IDS       : <uint64>[N,2,3], BH particle ID numbers for each entry
     DETAILS.SCALES    : <flt64> [N,2,3], scale factor at which each entry was written
     DETAILS.MASSES    : <flt64> [N,2,3], BH mass
     DETAILS.MDOTS     : <flt64> [N,2,3], BH Mdot
     DETAILS.RHOS      : <flt64> [N,2,3], ambient mass-density
     DETAILS.CS        : <flt64> [N,2,3], ambient sound-speed
   }


Notes
-----

"""

import os, sys, warnings
import numpy as np
from datetime import datetime

import BHDetails, BHMergers, BHConstants
from BHConstants import MERGERS, DETAILS, BH_TYPE, BH_TIME
from MatchDetails import getDetailIndicesForMergers
from ..Constants import DTYPE, NUM_SNAPS

import zcode.InOut as zio

VERSION = 0.22



def main(run, verbose=True):

    if( verbose ): print " - BHMatcher.py"
    start = datetime.now()

    # Load Mergers
    mergers = BHMergers.loadFixedMergers(run)
    numMergers = mergers[MERGERS.NUM]
    if( verbose ): print " - - %d Mergers loaded" % (numMergers)

    # Load Matching Details
    mergerDetails = loadMergerDetails(run, mergers=mergers, verbose=verbose)
    
    # Check Matches
    if( verbose ): print " - - Checking Matches"
    checkMatches(mergerDetails, mergers)

    stop = datetime.now()
    if( verbose ): print " - Done after %s" % (str(stop-start))

    return

# main()


def loadMergerDetails(run, loadsave=True, mergers=None, verbose=True):

    if( verbose ): print " - - BHMatcher.loadMergerDetails()"

    saveFile = BHConstants.GET_MERGER_DETAILS_FILENAME(run, VERSION)

    ## Try to Load Existing Merger Details
    if( loadsave ):
        if( verbose ): print " - - - Loading save from '%s'" % (saveFile)
        # Make sure file exists
        if( os.path.exists(saveFile) ):
            mergerDetails = zio.npzToDict(saveFile)
            if( verbose ): print " - - - Loaded Merger Details"
        else:
            loadsave = False
            warnStr = "File '%s' does not exist!" % (saveFile)
            warnings.warn(warnStr, RuntimeWarning)
                

    ## Re-match Mergers with Details
    if( not loadsave ):
        if( verbose ): print " - - - Rematching mergers and details"
        if( mergers is None ): mergers = BHMergers.loadFixedMergers(run)
        if( verbose ): print " - - - - Loaded %d Mergers" % (mergers[MERGERS.NUM])
    
        # Get Details for Mergers
        mergerDetails = detailsForMergers(run, mergers, verbose=verbose)
        # Add meta-data
        mergerDetails[DETAILS.RUN]     = run
        mergerDetails[DETAILS.CREATED] = datetime.now().ctime()
        mergerDetails[DETAILS.VERSION] = VERSION
        mergerDetails[DETAILS.FILE]    = saveFile

        # Save merger details
        zio.dictToNPZ(mergerDetails, saveFile, verbose=verbose)


    return mergerDetails

# loadMergerDetails()



def detailsForMergers(run, mergers, verbose=True):
    """
    Given a set of mergers, retrieve corresponding 'details' entries for BHs.

    Finds the details entries which occur closest to the 'merger' time, both
    before and after it (after only exists for the 'out' BHs).

    Arguments
    ---------
        run     : int, illustris run number {1,3}
        mergers : dict, data arrays for mergers
        verbose : bool (optional : ``True``), flag to print verbose output.

    
    Returns
    -------
        mergDets : dict, data arrays corresponding to each 'merger' BHs

    """

    if( verbose ): print " - - BHMatcher.detailsForMergers()"

    numMergers = mergers[MERGERS.NUM]
    numSnaps   = Constants.NUM_SNAPS

    # Search for all Mergers in each snapshot (easier, though less efficient)
    targets = np.arange(numMergers)

    snapNumbers = reversed(xrange(numSnaps))

    matchInds  = -1*np.ones([numMergers, NUM_BH_TYPES, NUM_BH_TIMES], dtype=DTYPE.INDEX)
    matchTimes = -1*np.ones([numMergers, NUM_BH_TYPES, NUM_BH_TIMES], dtype=DTYPE.SCALAR)


    ## Create Dictionary To Store Results
    mergDets = {}
    shape = [numMergers, len(BH_TYPE), len(BH_TIME)]
    # Initialize Dictionary to Invalid entries (-1)
    for KEY in DETAILS_PHYSICAL_KEYS:
        if( KEY == DETAILS.IDS ): mergDets[KEY] = np.zeros(shape, dtype=DTYPE.ID)
        else:                     mergDets[KEY] = -1*np.ones(shape, dtype=DTYPE.SCALAR)

    ## Iterate over all Snapshots
    #  --------------------------
    if( verbose ): pbar = zio.getProgressBar(numSnaps)
    for ii,snum in enumerate(snapNumbers):
        # Load appropriate Details File
        dets = BHDetails.loadBHDetails(run, snum, verbose=False)
        # If there are no details here, continue to next iter, but save BHs
        if( dets[DETAILS.NUM] == 0 ): continue
        # Track which matches are 'new' (+1 = new); reset at each snapshot
        matchNew  = -1*np.ones([shape], dtype=int)  # THIS MIGHT NEED TO BE A LONG???? FIX

        ## Load Details Information
        numMatches = getDetailIndicesForMergers(targets, mergers[MERGERS.IDS], 
                                                mergers[MERGERS.SCALES], matchInds,
                                                matchTimes, matchNew,
                                                dets[DETAILS.IDS], dets[DETAILS.SCALES])


        ## Store Matches Data

        # iterate over in/out BHs
        for TYPE in BH_TYPE:
            # Iterate over match times
            for TIME in BH_TIME:
                # Find valid matches
                inds = np.where( matchNew[:, TYPE, TIME] >= 0 )[0]
                if( len(inds) > 0 ):
                    # Store Each Parameter
                    for KEY in DETAILS_PHYSICAL_KEYS:
                        mergDets[KEY][inds, TYPE, TIME] = dets[KEY][ matchInds[inds, TYPE, TIME] ]

                    # KEY
            # FBA
        # BH
        
        if( verbose ): pbar.update(ii)

    # } ii

    if( verbose ): pbar.finish()

    return mergDets

# detailsForMergers()




def checkMatches(matches, mergers):
    """
    Perform basic diagnostic on merger-detail matches.

    Arguments
    ---------

    """

    RAND_NUMS = 4

    numMergers = mergers[MERGERS.NUM]
    bh_str = { BH_TYPE.IN : "In ", BH_TYPE.OUT : "Out" }
    time_str = { BH_TIME.FIRST : "First ", BH_TIME.BEFORE : "Before", BH_TIME.AFTER : "After " }

    good = np.zeros([numMergers, len(BH_TYPE), len(BH_TIME)], dtype=bool)
    
    ### Count Seemingly Successful Matches ###
    print " - - - Number of Matches (%6d Mergers) :" % (numMergers)
    print "             First  Before After"

    # Iterate over both BHs
    for TYPE in BH_TYPE:
        num_str = ""
        # Iterate over match times
        for TIME in BH_TIME:
            inds = np.where( matches[DETAILS.SCALES][:,TYPE,TIME] >= 0.0 )[0]
            good[inds, BH, FBA] = True
            num_str += "%5d  " % (len(inds))

        # TIME

        print "       %s : %s" % ( bh_str[BH], num_str )

    # TYPE

    ## Count Combinations of Matches
    #  -----------------------------
    print "\n - - - Number of Match Combinations :"

    # All Times
    inds = np.where( np.sum(good[:,BH_TYPE.OUT,:], axis=1) == len(BH_TIME) )[0]
    print " - - - - All            : %5d" % (len(inds))
    
    # Before and After
    inds = np.where( good[:,BH_TYPE.OUT,BH_TIME.BEFORE] & 
                     good[:,BH_TYPE.OUT,BH_TIME.AFTER] )[0]
    print " - - - - Before & After : %5d" % (len(inds))

    # First and Before
    inds = np.where( good[:,BH_TYPE.OUT,BH_TIME.FIRST] & 
                     good[:,BH_TYPE.OUT,BH_TIME.BEFORE] )[0]
    print " - - - - First & Before : %5d" % (len(inds))

    print ""
    inds = np.where( (good[:,BH_TYPE.OUT,BH_TIME.FIRST]  == True ) & 
                     (good[:,BH_TYPE.OUT,BH_TIME.BEFORE] == False) )[0]

    print " - - - Number of First without Before : %5d" % (len(inds))
    if( len(inds) > 0 ):
        print "\t\t         First      Before    ( Merger  )   After"
        if( len(inds) <= RAND_NUMS ): sel = np.arange(len(inds))
        else:                         sel = np.random.randint(0, len(inds), size=RAND_NUMS)  

        for ii in sel:
            sel_id = inds[ii]
            tmat = matches[DETAILS.SCALES][sel_id,BH_TYPE.OUT,:]
            tt = mergers[MERGERS.SCALES][sel_id]
            print "\t\t%5d : %+f  %+f  (%+f)  %+f" % \
                (sel_id, tmat[BH_TIME.FIRST], tmat[BH_TIME.BEFORE], tt, tmat[BH_TIME.AFTER])


    ### Check ID Matches
    print "\n - - - Number of BAD ID Matches:"
    print "             First  Before After"
    # Iterate over both BHs
    for BH in BH_TYPE:
        num_str = ""
        eg = None
        # Iterate over match times
        for FBA in BH_TIME:
            dids = matches[DETAILS.IDS][:,BH,FBA]
            mids = mergers[MERGERS.IDS][:,BH]
            inds = np.where( (good[:,BH,FBA]) & (dids != mids) )[0]
            
            num_str += "%5d  " % (len(inds))
            if( len(inds) > 0 ): eg = inds[0]


        print "       %s : %s" % ( bh_str[BH], num_str )
        if( eg is not None ):
            tmat = matches[DETAILS.SCALES][sel_id,BH_TYPE.OUT,:]
            tt = mergers[MERGERS.SCALES][sel_id]
            print "\t\t%5d : %+f  %+f  (%+f)  %+f" % \
                (eg, tmat[BH_TIME.FIRST], tmat[BH_TIME.BEFORE], tt, tmat[BH_TIME.AFTER])


    ## Check Time Matches
    print "\n - - - Number of BAD time Matches:"
    # Iterate over both BHs
    for BH in BH_TYPE:
        num_str = ""
        eg = None
        # Iterate over match times
        for FBA in BH_TIME:
            dt = matches[DETAILS.SCALES][:,BH,FBA]
            mt = mergers[MERGERS.SCALES]

            if( FBA == BH_TIME.AFTER): inds = np.where( (good[:,BH,FBA]) & (dt <  mt) )[0]
            else:                      inds = np.where( (good[:,BH,FBA]) & (dt >= mt) )[0]

            num_str += "%5d  " % (len(inds))
            if( len(inds) > 0 ): eg = inds[0]
            
        print "       %s : %s" % ( bh_str[BH], num_str )
        if( eg is not None ):
            tmat = matches[DETAILS.SCALES][eg,BH_TYPE.OUT,:]
            tt = mergers[MERGERS.SCALES][eg]
            print "\t\t%5d : %+f  %+f  (%+f)  %+f" % \
                (eg, tmat[BH_TIME.FIRST], tmat[BH_TIME.BEFORE], tt, tmat[BH_TIME.AFTER])

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


    numMergers = mergers[MERGERS.NUM]
    mass = mdets[DETAILS.MASSES]
    scal = mdets[DETAILS.SCALES]

    if( debug ):
        inds_inn_bef = np.where( mass[:,BH_TYPE.IN ,BH_TIME.BEFORE] <= 0.0 )[0]
        inds_out_bef = np.where( mass[:,BH_TYPE.OUT,BH_TIME.BEFORE] <= 0.0 )[0]
        inds_out_aft = np.where( mass[:,BH_TYPE.OUT,BH_TIME.AFTER ] <= 0.0 )[0]
        print "BHMatcher.inferOutMasses() : %d missing IN  BEFORE" % (len(inds_inn_bef))
        print "BHMatcher.inferOutMasses() : %d missing OUT BEFORE" % (len(inds_out_bef))
        print "BHMatcher.inferOutMasses() : %d missing OUT AFTER " % (len(inds_out_aft))


        
    ## Fix Mass Entries
    #  ----------------
    if( verbose ): print " - - - Inferring 'out' BH masses at merger"
        
    # Details entries just before merger are the best option, default to this
    massOut = np.array( mass[:,BH_TYPE.OUT,BH_TIME.BEFORE] )
    inds = np.where( massOut <= 0.0 )[0]
    if( verbose ): 
        print " - - - - %d/%d Entries missing details 'out' 'before'" % \
            (len(inds), numMergers)

    # If some are missing, use difference from out-after and merger-in
    inds = np.where( massOut <= 0.0 )[0]
    if( len(inds) > 0 ): 
        massOutAfter = mass[inds,BH_TYPE.OUT,BH_TIME.AFTER]
        massInDuring = mergers[MERGERS.MASSES][inds,BH_TYPE.IN]
        massOut[inds] = massOutAfter - massInDuring 
        inds = np.where( massOut[inds] > 0.0 )[0]
        print " - - - - %d Missing entries replaced with 'out' 'after' minus merger 'in'" % \
            (len(inds))

    # If some are still missing, use difference from out-after and in-before
    inds = np.where( massOut <= 0.0 )[0]
    if( len(inds) > 0 ): 
        massOutAfter = mass[inds,BH_TYPE.OUT,BH_TIME.AFTER]
        massInBefore = mass[inds,BH_TYPE.IN,BH_TIME.BEFORE]
        massOut[inds] = massOutAfter - massInBefore
        inds = np.where( massOut[inds] > 0.0 )[0]
        print " - - - - %d Missing entries replaced with 'out' 'after' minus 'in' 'before'" % \
            (len(inds))

    inds = np.where( massOut <= 0.0 )[0]
    if( verbose ): print " - - - - %d/%d Out masses still invalid" % (len(inds), numMergers)

    return massOut

# inferMergerOutMasses()


if( __name__ == "__main__" ): main()
