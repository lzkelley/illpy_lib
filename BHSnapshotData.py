"""


"""

import numpy as np
import scipy as sp
from datetime import datetime
from enum import Enum
import os, sys

from illpy.Constants import NUM_SNAPS, GET_ILLUSTRIS_OUTPUT_DIR
from illpy.illbh import BHMergers
from illpy.illbh.BHConstants import MERGERS, BH_TYPE

from Settings import *

import illustris_python as ill

import zcode
import zcode.InOut as zio


VERSION = 0.2

_SAVE_FILE_NAME = "./data/ill-%d_bh_snapshot_data_v-%.1f.npz"
def GET_SAVE_FILE_NAME(run, vers): return _SAVE_FILE_NAME % (run, vers)

class SNAP():
    VERSION = 'version'
    CREATED = 'created'
    RUN     = 'run'
    DIR_SRC = 'directory'

SNAPSHOT_FIELDS = ['ParticleIDs', 'BH_Hsml', 'BH_Mass', 'Masses', 'SubfindHsml']



def main():

    if( len(sys.argv) > 1 ): run = np.int(sys.argv[1])
    else:                    run = RUN_NUM

    print " - BHSnapshotData.main()"
    print " - - Loading BH Snapshot Data for run %d" % (run)
    beg = datetime.now()
    loadBHSnapshotData(run, verbose=True)
    end = datetime.now()
    print " - Done after %s" % (str(end-beg))

    return

# main()


def loadBHSnapshotData(run, loadsave=True, verbose=True, debug=False):
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
            if( verbose ): print " - - - Snapshot Data save file '%s' Exists" % (saveFileName)
            snapData = zio.npzToDict(saveFileName)
            # Make sure version matches
            if( snapData[SNAP.VERSION] == VERSION ):
                if( verbose ): print " - - - Snapshot Particle Data loaded"
            else:
                print " - - - Snapshot Data save file '%s' is out of date!" % (saveFileName)
                loadsave = False

        else:
            print " - - - Snapshot Data save file '%s' does not exist!" % (saveFileName)
            loadsave = False


    ## Import data directly from Illustris
    if( not loadsave ):
        # Import data
        snapData = _importBHSnapshotData(run, verbose=verbose, debug=debug)
        # Save data to NPZ file
        zio.dictToNPZ(snapData, saveFileName, verbose=verbose)


    return snapData

# loadBHSnapshotData()


def _importBHSnapshotData(run, verbose=True, debug=False):
    """
    Import BH particle data directly from illustris snapshots.

    Arguments
    ---------
       run     <int>  : illustris simulation number {1,3}
       verbose <bool> : optional, print verbose output
       debug   <bool> : optional, print extremely verbose output for diagnostics

    Returns
    -------
       data    <dict> : BH snapshot particle data for each merger
       
                        Each entry in the dictionary is shapes [N,2] for ``N`` mergers,
                        and each of ``BH_TYPE.IN`` and ``BH_TYPE.OUT``.

    """

    if( verbose ): print " - - BHSnapshotData._importBHSnapshotData()"

    dir_illustris = GET_ILLUSTRIS_OUTPUT_DIR(run)
    if( verbose ): print " - - - Using illustirs data dir '%s'" % (dir_illustris)

    ## Load BH Mergers
    #  ===============
    if( verbose ): print " - - - Loading BH Mergers"
    mergers = BHMergers.loadFixedMergers(run, verbose=verbose, loadsave=True)
    numMergers = mergers[MERGERS.NUM]
    if( verbose ): print " - - - - Loaded %d mergers" % (numMergers)

    data = {}
    num_pos = 0
    num_neg = 0

    ## Iterate Over Each Snapshot, Loading Data
    #  ========================================
    first = True
    if( verbose ): print " - - - Iterating over snapshots"
    beg = datetime.now()
    if( not debug ): pbar = zio.getProgressBar(NUM_SNAPS-1)
    # Go over snapshots in random order to get a better estimate of ETA/duration
    snapList = np.arange(NUM_SNAPS-1)
    np.random.shuffle(snapList)
    for snapNum in snapList:
        if( debug ): print "Snapshot %d/%d" % (snapNum, NUM_SNAPS)
        
        # Get Mergers occuring just after this Snapshot
        mrgs = mergers[MERGERS.MAP_STOM][snapNum+1]
        nums = len(mrgs)
        targetIDs = mergers[MERGERS.IDS][mrgs]
        if( debug ): print "Targeting %d mergers" % (nums)
        
        pos = 0
        neg = 0

        # If there are any mergers, load snapshot
        if( nums > 0 ):
            # catch non-fatal output
            with zio.StreamCapture() as output:
                snap = ill.snapshot.loadSubset(dir_illustris, snapNum, 'bh', fields=SNAPSHOT_FIELDS)

            snap_keys = snap.keys()
            if( 'count' in snap_keys ): snap_keys.remove('count')
            if( debug ): print "Loaded %d particles" % (snap['count'])

            # First time through, initialize dictionary of results
            if( first ):
                if( debug ): print "Initializing output dictionary"
                for key in snap_keys: 
                    data[key] = np.zeros([numMergers, 2], dtype=np.dtype(snap[key][0]))
                    
                first = False


            # Match target BHs
            if( debug ): print "Matching BHs"
            for index,tid in zip(mrgs,targetIDs):
                for BH in [BH_TYPE.IN, BH_TYPE.OUT]:
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

        if( not debug ): pbar.update(snapNum)

    # } for snap

    if( not debug ): pbar.finish()
    end = datetime.now()

    if( verbose ): print " - - - - %d Good, %d Bad - after %s" % (num_pos, num_neg, str(end-beg))

    # Add Meta-Data
    data[SNAP.VERSION] = VERSION
    data[SNAP.RUN]     = run
    data[SNAP.CREATED] = str(datetime.now())
    data[SNAP.DIR_SRC] = dir_illustris

    return data
    
# _importBHSnapshotData()



if( __name__ == "__main__" ): main()


