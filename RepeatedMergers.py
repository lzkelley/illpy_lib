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


### Builtin Modules ###
import numpy as np
import scipy as sp
import traceback as tb
from glob import glob
from datetime import datetime
from matplotlib import pyplot as plt
import bisect
import random
import sys
import os
import h5py

import warnings
warnings.simplefilter('error')                                                                      # Throw Error on Warnings


### Custom Modules and Files ###
from Settings import *
sys.path.append(*LIB_PATHS)

from Constants import *

import Basics

#import pyximport #; pyximport.install()
#pyximport.install(setup_args={"include_dirs":np.get_include()})
import FindRepeats

import plotting as gwplot

RUN_NUM = 3                                                                                         # Which illustris simulation to target
FILE_NAME = "repeated_save.npz"



###  ==============================================================  ###
###  ===========================  MAIN  ===========================  ###
###  ==============================================================  ###



def main():

    ### Initialize Log File ###
    print "\nRepeatedMergers.py\n"

    start_time  = datetime.now()

    ### Set basic Parameters ###
    run = RUN_NUM

    print " - Loading Basics"
    start = datetime.now()
    base = Basics.Basics(run)
    stop = datetime.now()
    print " - - Loaded after %s" % (str(stop-start))


    ### Find Repeated Mergers ###

    fname = FILE_NAME
    # Try to load precalculated repeat data
    if( os.path.exists(fname) ): 
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
        interval, next, last = repeatedMergerTimes(base)
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



def processRepeats(inter, next, last, base):

    print " - processRepeats()"
    
    print " - - %d Mergers" % (base.numMergers)

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
        if( inter[ii] < 0.0 ):
            mtime = base.cosmo.age(base.mergers['time'][ii])
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




def repeatedMergerTimes(base):

    print " - repeatedMergerTimes()"

    # Convert merger scale factors to ages
    print " - - Converting merger times"
    start = datetime.now()
    scales = base.mergers['time']
    times = np.array([ base.cosmo.age(sc) for sc in scales ], dtype=np.float32)
    stop = datetime.now()
    print " - - - Done after %s" % (str(stop-start))


    # Get repeated merger information
    print " - - Getting repeat statistics"
    start = datetime.now()
    inter, next, last = FindRepeats.findRepeats(base.mergers['id'], times)
    stop = datetime.now()
    print " - - - Retrieved after %s" % (str(stop-start))
    fname = FILE_NAME
    np.savez(fname, inter=inter, next=next, last=last)
    print " - - Saved to '%s'" % (fname)


    inds = np.where( inter >= 0.0 )[0]
    print "Num Intervals = %d" % (len(inds))
    aveInter = np.average( inter[inds] )
    print "Ave Interval  =  %.4e [s]  =  %.4f [Myr]" % (aveInter, aveInter/(1.0e6*YEAR))

    return interv, next, last



if __name__ == "__main__": main()

