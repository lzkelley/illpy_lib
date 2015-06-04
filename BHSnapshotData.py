"""


"""

import numpy as np
import scipy as sp
from datetime import datetime
from enum import Enum
import os

import illpy
from illpy.Constants import NUM_SNAPS, DTYPE_ID, DTYPE_SCALAR, GET_ILLUSTRIS_OUTPUT_DIR
from illpy.illbh import BHMergers
from illpy.illbh.BHConstants import MERGERS_NUM, MERGERS_MAP_STOM, MERGERS_IDS, BH_IN, BH_OUT

from Settings import *
import plotting as gwplot

import illustris_python as ill

import zcode
import zcode.InOut as zio


RUN_NUM = 3



_SAVE_FILE_NAME = "ill-%d_bh_snapshot_data_v-%.1f.npz"
def GET_SAVE_FILE_NAME(run, vers): return _SAVE_FILE_NAME % (run, vers)

def SNAP(Enum):
    VERSION = 'version'
    CREATED = 'created'
    RUN     = 'run'
    DIR_SRC = 'directory'


SNAPSHOT_FIELDS = ['ParticleIDs', 'BH_Hsml', 'BH_Mass', 'Masses', 'SubfindHsml']

VERSION = 0.1


    


'''
def main(run=RUN_NUM, verbose=VERBOSE):

    print " - BHSnapshotData.py"

    start_time  = datetime.now()

    ### Load Basic Simulation and Merger Variables ###
    print " - Loading Basics"
    start = datetime.now()
    base = Basics.Basics(run)
    stop = datetime.now()
    print " - - Loaded after %s" % (str(stop-start))


    ### Load Smoothing Lengths ###

    print " - Loading Blackhole Snapshot Particle Data"

    bhHsml, sfHsml, bhMass = importBHSnapshotData(run, base)

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
'''



def loadBHSnapshotData(run, loadsave=True, verbose=VERBOSE, debug=False):
    """
    Load Blackhole snapshot particle data.

    If ``loadsave``, first data is attempted to be loaded from existing savefile.  If this fails,
    of if ``loadsave`` is `False`, then the data is reimported directly from illustris snapshots.

    Arguments
    ---------
       run      <int>  : illustris simulation number {1,3}
       loadsave <bool> : optional, attempt to load previous save
       verbose  <bool> : optional, print verbose output
       debug    <bool> : optional, print extremely verbose diagnostic output

    Returns
    -------
       snapData <dict> : dictionary of snapshot particle data for all merger BHs

                         Each entry in the dictionary is shaped as [N,2] where ``N`` is the 
                         number of mergers

    """
    
    if( verbose ): print " - - BHSnapshotData.loadBHSnapshotData()"
    
    saveFileName = GET_SAVE_FILE_NAME(run, VERSION)

    ## Load Existing Data
    if( loadsave ):
        # If save file exists, load it
        if( os.path.exists(saveFileName) ):
            snapData = zio.npzToDict(saveFileName)
            # Make sure version matches
            if( snapData[SNAP.VERSION] != VERSION ):
                print " - - - Snapshot Data save file '%s' is out of date!" % (saveFileName)
                loadsave = False

        else:
            print " - - - Snapshot Data save file '%s' does not exist!" % (saveFileName)
            loadsave = False


    ## Import data directly from Illustris
    if( not loadsave ):
        # Import data
        snapData = importBHSnapshotData(run, verbose=verbose, debug=debug)
        # Save data to NPZ file
        zio.dictToNPZ(snapData, saveFileName, verbose=verbose)


    return snapData

# loadBHSnapshotData()


def importBHSnapshotData(run, verbose=VERBOSE, debug=False):
    """

    """

    if( verbose ): print " - - BHSnapshotData.importBHSnapshotData()"

    dir_illustris = GET_ILLUSTRIS_OUTPUT_DIR(run)
    if( verbose ): print " - - - Using illustirs data dir '%s'" % (dir_illustris)

    ## Load BH Mergers
    #  ===============
    if( verbose ): print " - - - Loading BH Mergers"
    mergers = BHMergers.loadFixedMergers(3, verbose=verbose, loadsave=True)
    numMergers = mergers[MERGERS_NUM]
    if( verbose ): print " - - - - Loaded %d mergers" % (numMergers)

    data = {}
    num_pos = 0
    num_neg = 0

    ## Iterate Over Each Snapshot, Loading Data
    #  ========================================
    first = True
    if( verbose ): print " - - - Iterating over snapshots"
    for snap in xrange(NUM_SNAPS):

        if( debug ): print "Snapshot %d/%d" % (snap, NUM_SNAPS)
        
        # Get Mergers for this Snapshot
        mrgs = mergers[MERGERS_MAP_STOM][snap]
        nums = len(mrgs)
        targetIDs = mergers[MERGERS_IDS][mrgs]
        if( debug ): print "Targeting %d mergers" % (nums)
        
        pos = 0
        neg = 0

        # If there are any mergers, load snapshot
        if( nums > 0 ):

            snap = ill.snapshot.loadSubset(dir_illustris, snap, 'bh', fields=SNAPSHOT_FIELDS)
            snap_keys = snap.keys()
            if( 'count' in snap_keys ): snap_keys.remove('count')
            if( debug ): print "Loaded %d particles" % (snap['count'])

            # First time through, initialize dictionary of results
            if( first ):
                if( debug ): print "Initializing output dictionary"
                for key in snap_keys: 
                    data[key] = -1*np.ones([numMergers, 2], dtype=np.dtype(snap[key][0]))
                    
                first = False


            # Match target BHs
            if( debug ): print "Matching BHs"
            for index,tid in zip(mrgs,targetIDs):
                for BH in [BH_IN, BH_OUT]:
                    ind = np.where( snap['ParticleIDs'] == tid[BH] )[0]
                    if( len(ind) == 1 ):
                        pos += 1
                        for key in snap_keys: data[key][index,BH] = snap[key][ind[0]]
                    else:
                        neg += 1

                # } for BH

            # } for index,tid
                
            if( debug ): print "%d Matching, %d Missing" % (pos, neg)
            num_pos += pos
            num_neg += neg

        # } if nums

    # } for snap

    if( verbose ): print " - - - - %d Matched, %d Missing" % (num_pos, num_neg)

    # Add Meta-Data
    data[SNAP.VERSION] = VERSION
    data[SNAP.VERSION] = run
    data[SNAP.VERSION] = str(datetime.now())
    data[SNAP.VERSION] = dir_illustris

    return data
    
# importBHSnapshotData()


