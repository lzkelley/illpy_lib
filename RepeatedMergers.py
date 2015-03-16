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
FILE_NAME = lambda xx: "ill-%d_repeated-mergers.npz" % (xx)

LOAD = False

REPEAT_LAST = 'last'
REPEAT_NEXT = 'next'
REPEAT_INTERVAL = 'interval'
REPEAT_CREATED = 'created'
REPEAT_RUN = 'run'


###  ==============================================================  ###
###  ===========================  MAIN  ===========================  ###
###  ==============================================================  ###



def main(run=RUN, load=LOAD):

    ### Initialize Log File ###
    print "\nRepeatedMergers.py\n"

    start_time  = datetime.now()

    ### Set basic Parameters ###
    print " - Loading Basics"
    start = datetime.now()
    base = Basics.Basics(run)
    stop = datetime.now()
    print " - - Loaded after %s" % (str(stop-start))


    ### Find Repeated Mergers ###

    fname = FILE_NAME(run)
    # Try to load precalculated repeat data
    if( os.path.exists(fname) and not load ): 
        print " - Loading Repeated Merger Data"
        start = datetime.now()
        dat = np.load(fname)
        interval = dat['inter']
        next = dat['next']
        last = dat['last']
        stop = datetime.now()
        print " - - Loaded after %s" % (str(stop-start))

    # Reload repeat data from mergers
    else:
        print " - Finding Repeated Mergers"
        start = datetime.now()
        repeats = repeatedMergerTimes(run, base)
        aux.saveDictNPZ(repeats, fname, verbose=True)
        stop = datetime.now()
        print " - - Done after %s" % (str(stop-start))


    ### Process Repeat Data ###
    lowerInter, numPast, numFuture = processRepeats(interval, next, last, base)

    return interval, lowerInter, numPast, numFuture

    
    ### Plot Repeat Data ###
    gwplot.plotFig4_RepeatedMergers(interval, lowerInter, numFuture)

    end_time    = datetime.now()
    durat       = end_time - start_time

    print "Done after %s\n\n" % (str(durat))

    return

# main()






###  ==============================================================  ###
###  =====================  PRIMARY FUNCTIONS  ====================  ###
###  ==============================================================  ###



def repeatedMergerTimes(run, base):

    print " - repeatedMergerTimes()"

    numMergers = base.mergers[MERGERS_NUM]
    next  = -1*  np.ones([numMergers],              dtype=long)
    last  = -1*  np.ones([numMergers,NUM_BH_TYPES], dtype=long)
    inter = -1.0*np.ones([numMergers,NUM_BH_TYPES], dtype=np.float64)

    # Convert merger scale factors to ages
    print " - - Converting merger times"
    start = datetime.now()
    scales = base.mergers[MERGERS_TIMES]
    times = np.array([ base.cosmo.age(sc) for sc in scales ], dtype=np.float64)
    stop = datetime.now()
    print " - - - Done after %s" % (str(stop-start))


    # Get repeated merger information
    print " - - Getting repeat statistics"
    start = datetime.now()
    mids = base.mergers[MERGERS_IDS]
    FindRepeats.findRepeats(mids, times, last, next, inter)
    stop = datetime.now()
    print " - - - Retrieved after %s" % (str(stop-start))

    # Create dictionary to store data
    repeatDict = { REPEAT_LAST     : last,
                   REPEAT_NEXT     : next,
                   REPEAT_INTERVAL : inter,
                   REPEAT_CREATED  : datetime.now().ctime(),
                   REPEAT_RUN      : run }

    return repeatDict





def processRepeats(repeats, base):

    print " - processRepeats()"
    
    print " - - %d Mergers" % (base.numMergers)

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
    nowTime = base.cosmo.age(1.0) #0.999999)

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

