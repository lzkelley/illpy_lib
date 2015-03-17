# ==================================================================================================
# RepeatedMergers.py
# ------------------
#
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================


import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import os


import illpy
from illpy import AuxFuncs as aux
from illpy.Constants import *
from illpy.illbh.BHConstants import *

#from Settings import *
#from Constants import *

import Basics

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
import FindRepeats

import plotting as gwplot

RUN = 3                                                                                         # Which illustris simulation to target
VERBOSE = True
FILE_NAME = lambda xx: "ill-%d_repeated-mergers.npz" % (xx)

LOAD = False

REPEAT_LAST = 'last'
REPEAT_NEXT = 'next'
REPEAT_INTERVAL = 'interval'
REPEAT_CREATED = 'created'
REPEAT_RUN = 'run'


HEADER = " - RepeatedMergers :"

###  ==============================================================  ###
###  ===========================  MAIN  ===========================  ###
###  ==============================================================  ###



def main(run=RUN, load=LOAD, verbose=VERBOSE):

    ### Initialize Log File ###
    print "\nRepeatedMergers.py\n"

    start_time  = datetime.now()

    ### Set basic Parameters ###
    print " - Loading Basics"
    start = datetime.now()
    base = Basics.Basics(run)
    stop = datetime.now()
    print " - - Loaded after %s" % (str(stop-start))


    ### Load Repeated Mergers ###
    repeats = getRepeats(run, base, load=False, verbose=verbose)

    ### Process Repeat Data ###
    lowerInter, numPast, numFuture = analyzeRepeats(interval, next, last, base)

    return interval, lowerInter, numPast, numFuture

    
    ### Plot Repeat Data ###
    gwplot.plotFig4_RepeatedMergers(interval, lowerInter, numFuture)

    end_time    = datetime.now()
    durat       = end_time - start_time

    print "Done after %s\n\n" % (str(durat))

    return

# main()



def getRepeats(run, base, load=False, verbose=VERBOSE):
    """
    Load repeat data from save file if possible, or recalculate directly.

    Arguments
    ---------
    run : <int>
        Illlustris run number {1,3}
    base : <Basics>, Basics.py:Basics object
        Contains merger information
    load : <bool>, (optional=False)
        Reload repeat data directly from merger data
    verbose : <bool>, (optional=VERBOSE)
        Print verbose output

    Returns
    -------
    repeats : <dict>
        container for repeat data - see RepeatedMergers doc

    """

    if( verbose ): print " - - RepeatedMergers.getRepeats()"

    fname = FILE_NAME(run)
    # Try to load precalculated repeat data
    if( os.path.exists(fname) and not load ): 
        if( verbose ) : print " - - - Loading Repeated Merger Data from '%s'" % (fname)
        start = datetime.now()
        repeats = np.load(fname)
        stop = datetime.now()
        if( verbose ) : print " - - - - Loaded after %s" % (str(stop-start))

    # Reload repeat data from mergers
    else:
        print " - - - Finding Repeated Mergers from Merger Data"
        start = datetime.now()
        # Find Repeats
        repeats = calculateRepeatedMergers(run, base)
        # Save Repeats data
        aux.saveDictNPZ(repeats, fname, verbose=True)
        stop = datetime.now()
        if( verbose ): print " - - - - Done after %s" % (str(stop-start))


    return repeats


###  ==============================================================  ###
###  =====================  PRIMARY FUNCTIONS  ====================  ###
###  ==============================================================  ###



def calculateRepeatedMergers(run, base, verbose=VERBOSE):
    """
    Use merger data to find and connect BHs which merge multiple times.

    Arguments
    ---------
    run : <int>
        Illlustris run number {1,3}
    base : <Basics>, Basics.py:Basics object
        Contains merger information
    verbose : <bool>, (optional=VERBOSE)
        Print verbose output

    Returns
    -------
    repeats : <dict>
        container for repeat data - see RepeatedMergers doc

    """

    if( verbose ): print " - - RepeatedMergers.calculateRepeatedMergers()"

    numMergers = base.mergers[MERGERS_NUM]
    next  = -1*  np.ones([numMergers],              dtype=long)
    last  = -1*  np.ones([numMergers,NUM_BH_TYPES], dtype=long)
    inter = -1.0*np.ones([numMergers,NUM_BH_TYPES], dtype=np.float64)

    # Convert merger scale factors to ages
    if( verbose ): print " - - - Converting merger times"
    start = datetime.now()
    scales = base.mergers[MERGERS_TIMES]
    times = np.array([ base.cosmo.age(sc) for sc in scales ], dtype=np.float64)
    stop = datetime.now()
    if( verbose ): print " - - - - Done after %s" % (str(stop-start))


    # Get repeated merger information
    if( verbose ): print " - - - Getting repeat statistics"
    start = datetime.now()
    mids = base.mergers[MERGERS_IDS]
    FindRepeats.findRepeats(mids, times, last, next, inter)
    stop = datetime.now()
    if( verbose ): print " - - - - Retrieved after %s" % (str(stop-start))

    inds = np.where( last < 0 )[0]
    print "MISSING LAST = ", len(inds)

    inds = np.where( next < 0 )[0]
    print "MISSING NEXT = ", len(inds)


    # Create dictionary to store data
    repeats = { REPEAT_LAST     : last,
                REPEAT_NEXT     : next,
                REPEAT_INTERVAL : inter,
                REPEAT_CREATED  : datetime.now().ctime(),
                REPEAT_RUN      : run }

    return repeats




def analyzeRepeats(repeats, base, verbose=VERBOSE):
    """
    Analyze the data from calculation of repeated mergers to obtain typical number of repeats, etc.

    Arguments
    ---------
    repeats : <dict>
        container for repeat data - see RepeatedMergers doc
    base : <Basics>, Basics.py:Basics object
        Contains merger information
    verbose : <bool>, (optional=VERBOSE)
        Print verbose output

    Returns
    -------


    """

    if( verbose ): print " - - RepeatedMergers.analyzeRepeats()"
    
    numMergers = base.numMergers

    interval = repeats[REPEAT_INTERVAL]
    last     = repeats[REPEAT_LAST]
    next     = repeats[REPEAT_NEXT]

    aveFuture = 0.0
    aveFutureNum = 0
    avePast = 0.0
    avePastNum = 0

    lowerInter = -1.0*np.ones(base.numMergers, dtype=float)
    numPast = -1*np.ones(base.numMergers, dtype=int)
    numFuture = -1*np.ones(base.numMergers, dtype=int)

    # Get age of the universe now
    nowTime = base.cosmo.age(1.0)



    if( verbose ): print " - - - %d Mergers" % (numMergers)

    # Find number of unique merger BHs (i.e. no previous mergers)
    inds = np.where( (last[:,IN_BH] < 0) & (last[:,OUT_BH] < 0) & (next[:] < 0) )
    numTwoIsolated = len(inds)
    inds = np.where( ((last[:,IN_BH] < 0) ^ (last[:,OUT_BH] < 0)) & (next[:] < 0) )                 # 'xor' comparison
    numOneIsolated = len(inds)
    
    if( verbose ): 
        print " - - - Mergers with neither BH previously merged = %d" % (numTwoIsolated)
        print " - - - Mergers with one     BH previously merged = %d" % (numOneIsolated)


    inds = np.where( interval == 0.0 )
    print "%d Intervals = 0.0" % (len(inds[0]))
    if( len(inds[0]) > 0 ):
        print "\te.g."
        print "\t", inds[0][0], inds[0][1], " : ", interval[inds]


    inds = np.where( interval < 0.0 )
    print "%d Intervals < 0.0" % (len(inds[0]))
    if( len(inds[0]) > 0 ):
        print "\te.g."
        print "\t", inds[0][0], inds[0][1], " : ", interval[inds]


    ### Go back through All Mergers to Count Repeats ###
    
    for ii in xrange(base.numMergers):

        # Get lower limit intervals (i.e. no later merger)
        if( interval[ii] < 0.0 ):
            mtime = base.cosmo.age(base.mergers[MERGERS_TIMES][ii])
            # Store time from merger until now
            lowerInter[ii] = (nowTime - mtime)
           

        ## Count Forward from First Mergers ##
        # If this is a first merger
        if( last[ii] < 0 ):
            jj = next[ii]
            # Walk through future mergers
            while( jj >= 0 ):
                numFuture[ii] += 1
                jj = next[jj]

            # Accumulate for averaging
            aveFuture += numFuture[ii]
            aveFutureNum += 1

                
        ## Count Backward from Last Mergers ##
        # If this is a final merger
        if( next[ii] < 0 ):
            jj = last[ii]
            # Walk through past mergers
            while( jj >= 0 ):
                numPast[ii] += 1
                jj = last[jj]

            # Accumulate for averaging
            avePast += numPast[ii]
            avePastNum += 1


    # } ii

    # Calculate averages
    if( avePastNum   > 0 ): avePast   /= avePastNum
    if( aveFutureNum > 0 ): aveFuture /= aveFutureNum

    inds = np.where(next >= 0)[0]
    numRepeats = len(inds)
    fracRepeats = 1.0*numRepeats/base.numMergers
    print " - - Number of repeated mergers = %d/%d = %.4f" % (numRepeats, base.numMergers, fracRepeats)
    print " - - Average Number of Repeated mergers  past, future  =  %.3f, %.3f" % (avePast, aveFuture)

    return lowerInter, numPast, numFuture



if __name__ == "__main__": main()

