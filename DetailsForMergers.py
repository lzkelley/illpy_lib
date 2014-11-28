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
        good, bad, fix = findBinaryDetails(RUN, snap, snapScales[snap], ids, scales)
        tm2 = datetime.now()
        print " - - - %d Good,  %d Bad,  %d Fixed" % (good, bad, fix)
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
    numFix = 0

    COUNT_LIMIT = 100
    NUM_DET_PARAMS = 5

    details = -1.0*np.ones([len(ids),NUM_DET_PARAMS], dtype=DBL)
    detIDs = -1*np.ones([len(ids),2], dtype=LONG)

    # If there are no active Binaries, return
    if( len(actInds) == 0 ): return numGood, numBad, numFix

    print " - - - %d Active in snapshot %d" % (len(actInds), snap)
    sys.stdout.flush()

    # Load Details for this snapshot
    snapDets = DetMan.loadBHDetails_NPZ(run, snap)

    # Iterate over active binaries
    for act in actInds:
        target  = ids[act,1]
        found   = target
        binDets = DetMan.detailsForBH(target, snapDets, side='left')

        # If details are NOT found for 'out' BH
        count = 0
        while( binDets[0] == None ):

            # See if 'out' BH is the 'in' of another merger
            inds = np.where( ids[:,0] == found )[0]
            if( len(inds) > 1 or len(inds) == 0 ):
                # raise RuntimeError("Could not Fix missing ID %d  %d!" % (target, found))
                print "Could not Fix missing ID %d  %d!" % (target, found)
                numBad += 1
                break
                

            found = ids[inds[0],1]
            binDets = DetMan.detailsForBH(found, snapDets, side='left')
            count += 1
            if( count >= COUNT_LIMIT ):
                # raise RuntimeError("Too many corrections!  For Missing ID %d" % (target) )
                print "Too many corrections!  For Missing ID %d" % (target)
                numBad += 1
                break


        if( count > 0 ): numFix += 1
        
        numGood += 1

        detIDs[act,:] = [target, found]
        details[act,:] = binDets[:]


    filename = SAVE_DIR(run) + SAVE_NAME(run,snap)
    np.savez(filename, ids=detIDs, details=details)
    print " - - - - Saved to '%s' " % (filename)

    return numGood, numBad, numFix




def loadBasics():
    dat = np.load(ID_FILE)
    ids = dat['ids']
    scales = dat['scale']

    dat = np.load(SCALES_FILE)
    snapScales = dat['scales']

    return ids, scales, snapScales





if __name__ == "__main__": main()
