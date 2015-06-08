"""


"""

import numpy as np
import scipy as sp
from datetime import datetime
from enum import Enum
import os
from cStringIO import StringIO
import sys

import illpy
from illpy.Constants import NUM_SNAPS, DTYPE_ID, DTYPE_SCALAR, GET_ILLUSTRIS_OUTPUT_DIR
from illpy.illbh import BHMergers
from illpy.illbh.BHConstants import MERGERS_NUM, MERGERS_MAP_STOM, MERGERS_IDS, BH_IN, BH_OUT

from Settings import *

import illustris_python as ill

import zcode
import zcode.InOut as zio


RUN_NUM = 3
VERSION = 0.2


_SAVE_FILE_NAME = "./data/ill-%d_bh_snapshot_data_v-%.1f.npz"
def GET_SAVE_FILE_NAME(run, vers): return _SAVE_FILE_NAME % (run, vers)

class SNAP():
    VERSION = 'version'
    CREATED = 'created'
    RUN     = 'run'
    DIR_SRC = 'directory'

SNAPSHOT_FIELDS = ['ParticleIDs', 'BH_Hsml', 'BH_Mass', 'Masses', 'SubfindHsml']

class Capturing(list):

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self._stringio = StringIO()
        sys.stderr = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout
        sys.stderr = self._stderr
    



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


def _importBHSnapshotData(run, verbose=VERBOSE, debug=False):
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
                        and each of ``BH_IN`` and ``BH_OUT``.

    """

    if( verbose ): print " - - BHSnapshotData._importBHSnapshotData()"

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
    beg = datetime.now()
    if( not debug ): print "\t\t|%s|\n\t\t|" % (" "*(NUM_SNAPS-2)),
    for snap in xrange(NUM_SNAPS-1):

        # if( num_pos > 100 ): break

        if( debug ): print "Snapshot %d/%d" % (snap, NUM_SNAPS)
        
        # Get Mergers occuring just after this Snapshot
        mrgs = mergers[MERGERS_MAP_STOM][snap+1]
        nums = len(mrgs)
        targetIDs = mergers[MERGERS_IDS][mrgs]
        if( debug ): print "Targeting %d mergers" % (nums)
        
        pos = 0
        neg = 0

        # If there are any mergers, load snapshot
        if( nums > 0 ):

            # catch non-fatal output
            with Capturing() as output:
                snap = ill.snapshot.loadSubset(dir_illustris, snap, 'bh', fields=SNAPSHOT_FIELDS)

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

        sys.stdout.write('.')
        sys.stdout.flush()

    # } for snap

    if( not debug ): print "|"
    end = datetime.now()

    if( verbose ): print " - - - - %d Good, %d Bad - after %s" % (num_pos, num_neg, str(end-beg))

    # Add Meta-Data
    data[SNAP.VERSION] = VERSION
    data[SNAP.RUN]     = run
    data[SNAP.CREATED] = str(datetime.now())
    data[SNAP.DIR_SRC] = dir_illustris

    return data
    
# _importBHSnapshotData()





