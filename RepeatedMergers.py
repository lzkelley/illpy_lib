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

#import ObjLog
#from ObjLog import Log

#import DetailsManager as DetMan
#import DetailsForMergers as DetMerg
import Basics

#from FlatLogDistribution import FlatLogDistribution as fldist
#import AuxFuncs as aux

#import plotting as gwplot


import pyximport #; pyximport.install()
pyximport.install(setup_args={"include_dirs":np.get_include()})
import FindRepeats



RUN_NUM = 3                                                                                         # Which illustris simulation to target








###  ==============================================================  ###
###  ===========================  MAIN  ===========================  ###
###  ==============================================================  ###



def main(load=True, save=False, detail=False):

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

    print " - Finding Repeated Mergers"
    start = datetime.now()
    #interval, numFuture, numPast = repeatedMergerTimes(base)
    repeatedMergerTimes(base)
    stop = datetime.now()
    print " - - Done after %s" % (str(stop-start))


    end_time    = datetime.now()
    durat       = end_time - start_time

    print "Done after %s\n\n" % (str(durat))

    return

# main()






###  ==============================================================  ###
###  =====================  PRIMARY FUNCTIONS  ====================  ###
###  ==============================================================  ###



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
    # inter, aveInter, numFuture, aveFuture, numPast, avePast = FindRepeats.findRepeats(base.mergers['id'], times)
    inter, next, last = FindRepeats.findRepeats(base.mergers['id'], times)
    stop = datetime.now()
    print " - - - Retrieved after %s" % (str(stop-start))
    fname = "repeated_save.npz"
    np.savez(fname, inter=inter, next=next, last=last)
    print " - - Saved to '%s'" % (fname)



    inds = np.where( inter >= 0.0 )[0]
    print "Num Intervals = %d" % (len(inds))
    aveInter = np.average( inter[inds] )
    print "Ave Interval  =  %.4e [s]  =  %.4f [Myr]" % (aveInter, aveInter/(1.0e6*YEAR))

    # Report averages
    #print " - - - Average number of mergers:  Future, Past = %.3f, %.3f" % (aveFuture, avePast)
    #print " - - - Average interval between mergers:  %.3e [s]  =  %.3f [Myr]" % (aveInterv, aveInterv/(1.0e6*YEAR))


    #return interv, numFuture, numPast
    return



if __name__ == "__main__": main()

