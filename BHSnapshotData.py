"""


"""

import numpy as np
import scipy as sp
from datetime import datetime
from enum import Enum
import os, sys
import argparse

from mpi4py import MPI

from illpy.Constants import NUM_SNAPS, GET_ILLUSTRIS_OUTPUT_DIR, GET_PROCESSED_DIR
from illpy.illbh import BHMergers
from illpy.illbh.BHConstants import MERGERS, BH_TYPE

import Settings

import illustris_python as ill

import zcode
import zcode.InOut as zio



_VERSION = 0.2
DEBUG = False



def _GET_BH_SNAPSHOT_DIR(run):
    return GET_PROCESSED_DIR(run) + "blackhole_particles/"

_BH_SINGLE_SNAPSHOT_FILENAME = "snap{0:03d}/ill-{1:d}_snap{0:03d}_merger-bh_snapshot_{2:.2f}.npz"
def _GET_BH_SINGLE_SNAPSHOT_FILENAME(run, snap, version=_VERSION):
    return _GET_BH_SNAPSHOT_DIR(run) + _BH_SINGLE_SNAPSHOT_FILENAME.format(run,snap,version)

_BH_SNAPSHOT_FILENAME = "ill-{0:d}_merger-bh_snapshot_v{1:.2f}.npz"
def GET_BH_SNAPSHOT_FILENAME(run, version=_VERSION):
    return _GET_BH_SNAPSHOT_DIR(run) + _BH_SNAPSHOT_FILENAME.format(run,version)


_STATUS_FILENAME = 'stat_BHSnapshotData_ill%d_v%.2f.txt'
def _GET_STATUS_FILENAME(run):
    return _STATUS_FILENAME % (run, _VERSION)



class SNAP():
    VERSION = 'version'
    CREATED = 'created'
    RUN     = 'run'
    DIR_SRC = 'directory'

SNAPSHOT_FIELDS = ['ParticleIDs', 'BH_Hsml', 'BH_Mass', 'Masses', 'SubfindHsml']


class TAGS():
    READY = 0
    START = 1
    DONE  = 2
    EXIT  = 3
# } class TAGS


class ENVSTAT():
    FAIL = -1
    EXST =  0
    NEWF =  1
# } class ENVSTAT



def main():

    ## Initialize MPI Parameters
    #  -------------------------

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    name = MPI.Get_processor_name()
    stat = MPI.Status()

    if( rank == 0 ):
        NAME = sys.argv[0]
        print "\n%s\n%s\n%s" % (NAME, '='*len(NAME), str(datetime.now()))


    ## Parse Arguments
    #  ---------------
    args    = _parseArguments()
    run     = args.run
    verbose = args.verbose
    debug   = args.debug

    ## Master Process
    #  --------------
    if( rank == 0 ):
        print "run           = %d  " % (run)
        print "version       = %.2f" % (_VERSION)
        print "MPI comm size = %d  " % (size)
        print ""
        print "verbose       = %s  " % (str(verbose))
        print "debug         = %s  " % (str(verbose))
        print ""
        beg_all = datetime.now()

        try: 
            _runMaster(run, comm)
        except Exception as err:
            _mpiError(comm, err)

        end_all = datetime.now()
        print " - - Total Duration '%s'" % (str(end_all-beg_all))


    ## Slave Processes
    #  ---------------
    else:

        try:    
            _runSlave(run, comm, verbose=True)
        except Exception as err:
            _mpiError(comm, err)

            
    return 



    beg = datetime.now()
    loadBHSnapshotData(run, verbose=True)
    end = datetime.now()
    print " - Done after %s" % (str(end-beg))

    return

# main()



def _runMaster(run, comm, verbose=True):
    """
    Run master process which manages all of the secondary ``slave`` processes.

    Details
    -------

    """

    stat = MPI.Status()
    rank = comm.rank
    size = comm.size

    if( verbose ): print " - Initializing"

    # Create output directory
    #    don't let slave processes create it - makes conflicts
    fname = _GET_BH_SINGLE_SNAPSHOT_FILENAME(run, 0)
    zio.checkPath(fname)

    ## Load BH Mergers
    if( verbose ): print " - - - Loading BH Mergers"
    mergers = BHMergers.loadFixedMergers(run, verbose=verbose, loadsave=True)
    numMergers = mergers[MERGERS.NUM]
    if( verbose ): print " - - - - Loaded %d mergers" % (numMergers)

    ## Init status file
    statFileName = _GET_STATUS_FILENAME(run)
    statFile = open(statFileName, 'w')
    print " - - Opened status file '%s'" % (statFileName)
    statFile.write('%s\n' % (str(datetime.now())))
    beg = datetime.now()

    num_pos = 0
    num_neg = 0
    count = 0

    # Go over snapshots in random order to get a better estimate of ETA/duration
    snapList = np.arange(NUM_SNAPS-1)
    np.random.shuffle(snapList)
    for snapNum in snapList:
        
        # Get Mergers occuring just after Snapshot `snapNum`
        mrgs = mergers[MERGERS.MAP_STOM][snapNum+1]
        nums = len(mrgs)
        targetIDs = mergers[MERGERS.IDS][mrgs]

        # if( len(subs) <= 0 ): continue

        # Look for available slave process
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
        source = stat.Get_source()
        tag = stat.Get_tag()

        # Track number of completed profiles
        if( tag == TAGS.DONE ): 
            durat, pos, neg = data

            times[count] = durat
            num_pos += pos
            num_neg += neg
            count += 1


        # Distribute tasks
        comm.send([snapNum, mrgs, targetIDs, numMergers], dest=source, tag=TAGS.START)

        # Write status to file
        dur = (datetime.now()-beg)
        statStr = 'Snap %3d   %8d/%8d = %.4f   in %s   %8d pos   %8d neg\n' % \
            (snapNum, count, NUM_SNAPS-1, 1.0*count/(NUM_SNAPS-1), str(dur), num_pos, num_neg)
        statFile.write(statStr)
        statFile.flush()

    # snapNum

    statFile.write('\n\nDone after %s' % (str(datetime.now()-beg)))
    statFile.close()

    ## Close out all Processes
    #  =======================

    numActive = size-1
    print " - Exiting %d active processes" % (numActive)
    while( numActive > 0 ):
        
        # Find available slave process
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
        source = stat.Get_source()
        tag = stat.Get_tag()

        # If we're recieving exit confirmation, count it
        if( tag == TAGS.EXIT ): numActive -= 1
        else:
            # If a process just completed, count it
            if( tag == TAGS.DONE ): 
                durat, pos, neg = data
                times[count] = durat
                count += 1

            # Send exit command
            comm.send(None, dest=source, tag=TAGS.EXIT)

    # } while

    print " - - %d/%d = %.4f Completed tasks!" % (count, numSnaps-1, 1.0*count/(numSnaps-1))
    print " - - Totals: pos = %5d   neg = %5d" % (num_pos, num_neg)

    return
    
# _runMaster()




def _runSlave(run, comm, loadsave=True, verbose=False, debug=False):
    """

    Arguments
    ---------
       run      <int>       : illustris simulation run number {1,3}
       comm     <...>       : MPI intracommunicator object (e.g. ``MPI.COMM_WORLD``)

       loadsave <bool>      : optional, load data for this subhalo if it already exists

    Details
    -------
     - Waits for ``master`` process to send subhalo numbers

     - Returns status to ``master``

    """

    stat = MPI.Status()
    rank = comm.rank
    size = comm.size

    dir_illustris = GET_ILLUSTRIS_OUTPUT_DIR(run)
    data = {}
    first = True

    if( verbose ): print " - - BHSnapshotData._runSlave() : rank %d/%d" % (rank, size)

    # Keep looking for tasks until told to exit
    while True:
        # Tell Master this process is ready
        comm.send(None, dest=0, tag=TAGS.READY)
        # Receive ``task`` ([snap,idxs,bhids,numMergers])
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=stat)
        tag = stat.Get_tag()

        if( tag == TAGS.START ):
            # Extract parameters
            snap, idxs, bhids, numMergers = task
            beg = datetime.now()
            # Load and save Snapshot BHs
            data, first, pos, neg = _loadSnapshotBHs(run, dir_illustris, numMergers, snap, data,
                                                     first, idxs, bhids, debug=debug)
            end = datetime.now()
            durat = (end-beg).total_seconds()
            comm.send([durat,pos,neg], dest=0, tag=TAGS.DONE)
        elif( tag == TAGS.EXIT  ):
            break


    # Finish, return done
    comm.send(None, dest=0, tag=TAGS.EXIT)

    return
    
# _runSlave()



def _loadSnapshotBHs(run, illdir, numMergers, snapNum, data, first, idxs, bhids, debug=False):
    """
    Load the data for BHs in a single snapshot
    """

    ## Load Snapshot
    #  -------------
    with zio.StreamCapture() as output:
        snapshot = ill.snapshot.loadSubset(illdir, snapNum, 'bh', fields=SNAPSHOT_FIELDS)

    snap_keys = snapshot.keys()
    if( 'count' in snap_keys ): 
        snap_keys.remove('count')
        if( debug ): print "Loaded %d particles" % (snapshot['count'])

    # First time through, initialize dictionary of results
    if( first ):
        for key in snap_keys: 
            data[key] = np.zeros([numMergers, 2], dtype=np.dtype(snapshot[key][0]))

        first = False


    ## Match target BHs
    #  ----------------
    pos = 0
    neg = 0
    for index,tid in zip(mrgs,targetIDs):
        for BH in [BH_TYPE.IN, BH_TYPE.OUT]:
            ind = np.where( snapshot['ParticleIDs'] == tid[BH] )[0]
            if( len(ind) == 1 ):
                pos += 1
                for key in snap_keys: data[key][index,BH] = snapshot[key][ind[0]]
            else:
                neg += 1

        # BH
    # index, tid


    ## Save Data
    #  ---------
    fname = _GET_BH_SINGLE_SNAPSHOT_FILENAME(run, snapNum)
    zio.npzToDict(data, fname, verbose=debug)

    return data, first, pos, neg

# _loadSnapshotBHs()



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
    
    saveFileName = GET_SAVE_FILENAME(run, _VERSION)

    ## Load Existing Data
    if( loadsave ):
        # If save file exists, load it
        if( os.path.exists(saveFileName) ):
            if( verbose ): print " - - - Snapshot Data save file '%s' Exists" % (saveFileName)
            snapData = zio.npzToDict(saveFileName)
            # Make sure version matches
            if( snapData[SNAP.VERSION] == _VERSION ):
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
    data[SNAP.VERSION] = _VERSION
    data[SNAP.RUN]     = run
    data[SNAP.CREATED] = str(datetime.now())
    data[SNAP.DIR_SRC] = dir_illustris

    return data
    
# _importBHSnapshotData()



def _parseArguments():
    """
    Prepare argument parser and load command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version='%s %.2f' % (sys.argv[0], _VERSION))
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Verbose output', default=Settings.VERBOSE)
    parser.add_argument('--debug', action='store_true', 
                        help='Very verbose output', default=DEBUG)


    '''
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check",    dest='check',   action="store_true", default=CHECK_EXISTS)
    group.add_argument("--no-check", dest='nocheck', action="store_true", default=(not CHECK_EXISTS))
    '''
    parser.add_argument("run", type=int, nargs='?', choices=[1, 2, 3],
                        help="illustris simulation number", default=Settings.RUN_NUM)
    args = parser.parse_args()
    
    return args

# _parseArguments()



def _mpiError(comm, err="ERROR"):
    """
    Raise an error through MPI and exit all processes.

    Arguments
    ---------
       comm <...> : mpi intracommunicator object (e.g. ``MPI.COMM_WORLD``)
       err  <str> : optional, extra error-string to print

    """

    import traceback
    rank = comm.rank

    print "\nERROR: rank %d\n%s\n" % (rank, str(datetime.now()))
    print sys.exc_info()[0]
    print err.message
    print err.__doc__
    print "\n"
    print(traceback.format_exc())
    print "\n\n"

    comm.Abort(rank)
    return

# _mpiError()



if( __name__ == "__main__" ): main()


