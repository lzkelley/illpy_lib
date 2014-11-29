# ==================================================================================================
# DetailsForMergers.py
# --------------------
#
#
#
#
#
#
#
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================


### Builtin Modules ###
import sys

import numpy as np

from glob import glob
from datetime import datetime


### Custom Modules and Files ###
# Import Global Settings
from Settings import *
sys.path.append(*LIB_PATHS)
from Constants import *

import Basics

# Import local project files and objects
import AuxFuncs as aux
import DetailsManager as DetMan


# Cython
import pyximport#; pyximport.install()
pyximport.install(setup_args={"include_dirs":np.get_include()})
import MatchDetails


ID_FILE = "/n/home00/lkelley/illustris/pta-mergers/ill-3_bh-merger_ids-scales.npz"
SCALES_FILE = "/n/home00/lkelley/illustris/pta-mergers/ill-3_snapshot-scales.npz"
NUM_SNAPS = 136
SAVE_DIR = lambda x: "/n/home00/lkelley/illustris/post-process/Illustris-%d/bh-details/mergers/" % (x)
SAVE_NAME = lambda x,y: "ill-%d_snap-%d_details_merger-products.npz" % (x,y)


RUN = 3

###  ===================================  ###
###  =============  MAIN  ==============  ###
###  ===================================  ###



def main(snap=None):

    print "DetailsForMergers.py"
    startTime = datetime.now()

    ### Set Basic Parameters ###

    arglen = len(sys.argv)

    ### Determine Target Snapshots from Function and Command-line Arguments ###

    # If snapshots have been provided as function arguments
    if( snap != None ):
        # Make sure result is iterable
        if( np.iterable(snap) ): useSnaps = snap
        else:                    useSnaps = [ snap ]

    # If command-line arguments are provided
    elif( arglen > 1 ):

        start = np.int( sys.argv[1] )
        stop  = start + 1
        interv = 1

        if( arglen > 2 ): stop   = np.int( sys.argv[2] )
        if( arglen > 3 ): interv = np.int( sys.argv[3] )

        useSnaps = np.arange(start, stop, interv)

    # No snapshots specified, default to all
    else:
        useSnaps = np.arange(NUM_SNAPS)




    print " - Loading Basics"
    tm1 = datetime.now()
    ids, scales, snapScales = loadBasics()
    tm2 = datetime.now()
    print " - - Loaded %d after %s" % (len(ids), str(tm2-tm1))


    # Iterate over target Snapshots #
    print " - Iterating over %d snapshots" % (len(useSnaps))
    sys.stdout.flush()
    for ii,snap in enumerate(useSnaps):
        print " - - %d - %d/%d" % (snap, ii+1, len(useSnaps))
        tm1 = datetime.now()
        good, bad, fixed = findBinaryDetails(RUN, snap, snapScales[snap], ids, scales)
        tm2 = datetime.now()
        print " - - - %d Good,  %d Bad,  %d Fixed" % (good, bad, fixed)
        print " - - - After %s / %s" % (str(tm2-tm1), str(tm2-startTime))
        sys.stdout.flush()


    stopTime = datetime.now()
    print "\nDone after %s\n\n" % (str(stopTime-startTime))

    return




###  ================================================  ###
###  ===============  OTHER FUNCTIONS  ==============  ###
###  ================================================  ###



def findBinaryDetails(run, snap, snapScale, ids, scales):
    """
    Find BH Details for merger blackholes in the target snapshot.
    """

    # Select binaries which have already formed, 'active' binaries
    #    i.e. their merger scalefactor is <= that of this snapshot
    actInds = np.where( scales <= snapScale )[0]

    numGood = 0
    numBad = 0
    numFixed = 0

    print " - - - %d Active in snapshot %d" % (len(actInds), snap)
    sys.stdout.flush()

    scale = -1.0*np.ones(len(ids), dtype=DBL)
    mass = -1.0*np.ones(len(ids), dtype=DBL)
    mdot = -1.0*np.ones(len(ids), dtype=DBL)
    rho = -1.0*np.ones(len(ids), dtype=DBL)
    cs = -1.0*np.ones(len(ids), dtype=DBL)

    searchIDs = -1*np.ones([len(ids),2], dtype=LONG)


    # If there are active Binaries
    if( len(actInds) > 0 ): 


        # Load Details for this snapshot
        print " - - - Loading BH Details"
        start = datetime.now()
        snapDets = DetMan.loadBHDetails_NPZ(run, snap)
        detIDs = snapDets['id']
        detScales = snapDets['scale']
        stop = datetime.now()
        print " - - - - Loaded After %s" % (str(stop-start))

        # Iterate over active binaries
        print " - - - Getting Details Indices"

        start = datetime.now()
        targetIDs, foundIDs, mergerDetInds = MatchDetails.getDetailIndicesForMergers(actInds, ids[:,0], ids[:,1], detIDs) # detScales)
        stop = datetime.now()
        print " - - - - Done after %s" % (str(stop-start))

        ### Store Matches and Save ###
        searchIDs[actInds,0] = targetIDs
        searchIDs[actInds,1] = foundIDs

        scale[actInds] = snapDets['scale'][mergerDetInds]
        mass[actInds] = snapDets['mass'][mergerDetInds]
        mdot[actInds] = snapDets['mdot'][mergerDetInds]
        rho[actInds] = snapDets['rho'][mergerDetInds]
        cs[actInds] = snapDets['cs'][mergerDetInds]

        inds = np.where( targetIDs > 0 )[0]
        numGood = len(inds)
        inds = np.where( targetIDs < 0 )[0]
        numBad = len(inds)
        inds = np.where( targetIDs != foundIDs )[0]
        numFixed = len(inds)


    filename = SAVE_DIR(run) + SAVE_NAME(run,snap)
    np.savez(filename, ids=searchIDs, scale=scale, mass=mass, mdot=mdot, rho=rho, cs=cs)
    print " - - - - Saved to '%s' " % (filename)
    
    return numGood, numBad, numFixed



def loadBasics():
    dat = np.load(ID_FILE)
    ids = dat['ids']
    scales = dat['scale']

    dat = np.load(SCALES_FILE)
    snapScales = dat['scales']

    return ids, scales, snapScales





if __name__ == "__main__": main()
