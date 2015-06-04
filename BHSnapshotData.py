"""


"""

import numpy as np
import scipy as sp
from datetime import datetime

import illpy
from illpy.Constants import NUM_SNAPS, DTYPE_ID, DTYPE_SCALAR, GET_ILLUSTRIS_OUTPUT_DIR
from illpy.illbh import BHMergers
from illpy.illbh.BHConstants import MERGERS_NUM, MERGERS_MAP_MTOS, MERGERS_IDS

from Settings import *

import plotting as gwplot

import illustris_python as ill


RUN_NUM = 3

PARTICLE_FIELDS = ['ParticleIDs', 'BH_Hsml', 'BH_Mass', 'Masses', 'SubfindHsml']

#SAVE_FILE = "ill-3_bh-mergers_snapshots_smoothing-lengths.npz"



###  ==============================================================  ###
###  ===========================  MAIN  ===========================  ###
###  ==============================================================  ###



def main(run=RUN_NUM, verbose=VERBOSE):

    print " - SmoothingLengths.py"

    start_time  = datetime.now()

    ### Load Basic Simulation and Merger Variables ###
    print " - Loading Basics"
    start = datetime.now()
    base = Basics.Basics(run)
    stop = datetime.now()
    print " - - Loaded after %s" % (str(stop-start))


    ### Load Smoothing Lengths ###

    print " - Loading smoothing lengths"

    bhHsml, sfHsml, bhMass = getSmoothingLengths(run, base)

    # Plot
    gwplot.plotFig5_SmoothingLengths(bhHsml, sfHsml, bhMass)


    ### Get Effective Smoothing Length for each Merger ###

    Hsml = -1.0*np.ones(len(bhHsml), dtype=DBL)

    # Find mergers where both BHs have smoothing lengths
    inds = np.where( (bhHsml[:,0] >= 0.0) & (bhHsml[:,1] >= 0.0) )[0]

    # Get max of each pair of smoothing lengths
    Hsml[inds] = np.max( bhHsml[inds], axis=1 )

    # Save
    fname = PP_MERGERS_SNAPSOT_SMOOTHING_LENGTHS(run)
    hdict = { 'run':run,
              'bh_Hsml':bhHsml,
              'sf_Hsml':sfHsml,
              'Hsml':Hsml }

    np.savez(fname, **hdict)
    print " - - Saved to '%s'" % (fname)

    end_time    = datetime.now()
    durat       = end_time - start_time
    
    print "\nDone after %s\n\n" % (str(durat))

    return

# main()






###  ==============================================================  ###
###  =====================  PRIMARY FUNCTIONS  ====================  ###
###  ==============================================================  ###


def importSmoothingLengths(run, verbose=VERBOSE):

    if( verbose ): print " - - SmoothingLengths.importSmoothingLengths()"

    dir_illustris = GET_ILLUSTRIS_OUTPUT_DIR(run)

    ## Load BH Mergers
    #  ===============
    if( verbose ): print " - - - Loading BH Mergers"
    mergers = BHMergers.loadFixedMergers(3, verbose=verbose, loadsave=True)
    numMergers = mergers[MERGERS_NUM]
    if( verbose ): print " - - - - Loaded %d mergers" % (numMergers)

    bhHsml = -1*np.ones([numMergers, 2], dtype=DTYPE_SCALAR)
    sfHsml = -1*np.ones([numMergers, 2], dtype=DTYPE_SCALAR)
    bhMass = -1*np.ones([numMergers, 2], dtype=DTYPE_SCALAR)
    bhID   = -1*np.ones([numMergers, 2], dtype=DTYPE_ID)

    count = 0

    for snap in xrange(NUM_SNAPS):
        
        # Get number of mergers in this snapshot
        nmerg = len(base.s2m[snap])

        # If there are any mergers, load snapshot
        if( nmerg > 0 ):

            snap = ill.snapshot.loadSubset(dir_illustris, snap, 'bh', fields=PARTICLE_FIELDS)

            dat = np.load(fname)

            # Get the merger numbers for these BHs
            minds = dat['mergers']
        
            count += len(minds)

            bhHsml[minds,:] = dat['BH_Hsml']
            sfHsml[minds,:] = dat['SubfindHsml']
            bhMass[minds,:] = dat['BH_Mass']


    print " - - Retrieved %d/%d mergers" % (count, numMergers)

    return bhHsml, sfHsml, bhMass
    
# importSmoothingLengths()


def analyzeSmoothingLengths(bhHsml, sfHsml, bhMass):
    
    print " - analyzeSmoothingLengths()"


    return




if __name__ == "__main__": main()

