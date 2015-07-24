"""
Collect snapshot/particle data for merger BHs.

Objects
-------
    SNAP
    TAGS
    
Functions
---------
    main
    loadBHSnapshotData

    _runMaster
    _runSlave
    _loadSingleSnapshotBHs
    _mergeBHSnapshotFiles
    _initStorage
    _parseArguments
    _mpiError

"""

import numpy as np
import scipy as sp
from datetime import datetime
from enum import Enum
import os, sys
import argparse

from mpi4py import MPI

from illpy.Constants import NUM_SNAPS, GET_ILLUSTRIS_OUTPUT_DIR, GET_PROCESSED_DIR, DTYPE, \
    GET_BAD_SNAPS
from illpy.illbh import BHMergers
from illpy.illbh.BHConstants import MERGERS, BH_TYPE

import Settings

import illustris_python as ill

import zcode
import zcode.InOut as zio


_VERSION = 0.4
DEBUG = False


def _GET_BH_SNAPSHOT_DIR(run):
    return GET_PROCESSED_DIR(run) + "blackhole_particles/"

_BH_SINGLE_SNAPSHOT_FILENAME = "ill-{0:d}_snap{1:03d}_merger-bh_snapshot_{2:.2f}.npz"
def _GET_BH_SINGLE_SNAPSHOT_FILENAME(run, snap, version=_VERSION):
    return _GET_BH_SNAPSHOT_DIR(run) + _BH_SINGLE_SNAPSHOT_FILENAME.format(run,snap,version)

_BH_SNAPSHOT_FILENAME = "ill-{0:d}_merger-bh_snapshot_v{1:.2f}.npz"
def GET_BH_SNAPSHOT_FILENAME(run, version=_VERSION):
    return _GET_BH_SNAPSHOT_DIR(run) + _BH_SNAPSHOT_FILENAME.format(run,version)


_STATUS_FILENAME = 'stat_BHSnapshotData_ill%d_v%.2f.txt'
def _GET_STATUS_FILENAME(run):
    return _STATUS_FILENAME % (run, _VERSION)



class SNAP():
    RUN     = 'run'
    SNAP    = 'snap'
    VERSION = 'version'
    CREATED = 'created'
    DIR_SRC = 'directory'
    VALID   = 'valid'
    TARGET  = 'target'

SNAPSHOT_FIELDS = ['ParticleIDs', 'BH_Hsml', 'BH_Mass', 'Masses', 'SubfindHsml']
SNAPSHOT_DTYPES = [DTYPE.ID, DTYPE.SCALAR, DTYPE.SCALAR, DTYPE.SCALAR, DTYPE.SCALAR]

class TAGS():
    READY = 0
    START = 1
    DONE  = 2
    EXIT  = 3




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
            _runMaster(run, comm, debug=debug)
        except Exception as err:
            _mpiError(comm, err)

        end_all = datetime.now()
        print " - - Total Duration '%s'" % (str(end_all-beg_all))

        if( verbose ): print " - - - Merging Files"
        # allData = _mergeBHSnapshotFiles(run, verbose=verbose)
        allData = loadBHSnapshotData(run, loadsave=False, verbose=verbose, debug=debug)

    ## Slave Processes
    #  ---------------
    else:

        try:    
            _runSlave(run, comm, debug=debug)
        except Exception as err:
            _mpiError(comm, err)

            
    return 

# main()


def loadBHSnapshotData(run, loadsave=True, verbose=True, debug=False):

    if( verbose ): print " - - BHSnapshotData.loadBHSnapshotData()"

    fname = GET_BH_SNAPSHOT_FILENAME(run)

    ## Load Existing File
    #  ------------------
    if( loadsave ): 
        if( verbose ): print " - - - Loading from '%s'" % (fname)
        if( os.path.exists(fname) ):
            data = zio.npzToDict(fname)
        else:
            print "WARNING: '%s' does not exist!  Recreating!" % ( fname )
            loadsave = False

    ## Recreate data (Merge individual snapshot files)
    #  -----------------------------------------------
    else:
        if( verbose ): print " - - - Recreating '%s'" % (fname)
        data = _mergeBHSnapshotFiles(run, verbose=verbose, debug=debug)
        
        # Add Metadata
        data[SNAP.RUN] = run
        data[SNAP.VERSION] = _VERSION
        data[SNAP.CREATED] = datetime.now().ctime()

        # Save
        zio.dictToNPZ(data, fname, verbose=verbose)


    return data

# loadBHSnapshotData()


def _runMaster(run, comm, verbose=True, debug=False):
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
    print " - - - Opened status file '%s'" % (statFileName)
    statFile.write('%s\n' % (str(datetime.now())))
    beg = datetime.now()

    num_pos = 0
    num_neg = 0
    num_new = 0
    countDone = 0
    count = 0
    times = np.zeros(NUM_SNAPS-1)

    # Go over snapshots in random order to get a better estimate of ETA/duration
    snapList = np.arange(NUM_SNAPS-1)
    np.random.shuffle(snapList)
    if( verbose ): print " - - - Iterating over snapshots"
    if( verbose and not debug ): pbar = zio.getProgressBar(NUM_SNAPS-1)
    for snapNum in snapList:
        
        # Get Mergers occuring just after Snapshot `snapNum`
        mrgs = mergers[MERGERS.MAP_STOM][snapNum+1]
        nums = len(mrgs)
        targetIDs = mergers[MERGERS.IDS][mrgs]

        # if( len(subs) <= 0 ): continue

        # Look for available slave process
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
        src = stat.Get_source()
        tag = stat.Get_tag()

        # Track number of completed profiles
        if( tag == TAGS.DONE ): 
            durat, pos, neg, new = data

            times[countDone] = durat
            num_pos += pos
            num_neg += neg
            num_new += new
            countDone += 1


        # Distribute tasks
        comm.send([snapNum, mrgs, targetIDs, numMergers], dest=src, tag=TAGS.START)

        # Write status to file
        dur = (datetime.now()-beg)
        fracDone = 1.0*countDone/(NUM_SNAPS-1)
        statStr = 'Snap %3d (rank %03d)   %8d/%8d = %.4f  in %s  %8d pos  %8d neg  %3d new\n' % \
            (snapNum, src, countDone, NUM_SNAPS-1, fracDone, str(dur), num_pos, num_neg, num_new)
        statFile.write(statStr)
        statFile.flush()
        count += 1
        if( verbose and not debug ): pbar.update(count)

    # snapNum

    statFile.write('\n\nDone after %s' % (str(datetime.now()-beg)))
    statFile.close()
    if( verbose and not debug ): pbar.finish()

    ## Close out all Processes
    #  =======================

    numActive = size-1
    print " - Exiting %d active processes" % (numActive)
    while( numActive > 0 ):
        
        # Find available slave process
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
        src = stat.Get_source()
        tag = stat.Get_tag()

        # If we're recieving exit confirmation, count it
        if( tag == TAGS.EXIT ): numActive -= 1
        else:
            # If a process just completed, count it
            if( tag == TAGS.DONE ): 
                durat, pos, neg, new = data
                times[countDone] = durat
                countDone += 1
                num_pos += pos
                num_neg += neg
                num_new += new

            # Send exit command
            comm.send(None, dest=src, tag=TAGS.EXIT)

    # } while

    fracDone = 1.0*countDone/(NUM_SNAPS-1)
    print " - - %d/%d = %.4f Completed tasks!" % (countDone, NUM_SNAPS-1, fracDone)
    print " - - Average time %.4f +- %.4f" % (np.average(times), np.std(times))
    print " - - Totals: pos = %5d   neg = %5d   new = %3d" % (num_pos, num_neg, num_new)

    return
    
# _runMaster()




def _runSlave(run, comm, loadsave=True, debug=False):
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

    data = {}
    first = True

    if( debug ): print "%d : BHSnapshotData._runSlave()" % (rank)

    # Keep looking for tasks until told to exit
    while True:
        # Tell Master this process is ready
        if( debug ): print "%d : sending ready" % (rank)
        comm.send(None, dest=0, tag=TAGS.READY)
        # Receive ``task`` ([snap,idxs,bhids,numMergers])
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=stat)
        tag = stat.Get_tag()
        if( debug ): print "%d : received tag %d" % (rank, tag)

        if( tag == TAGS.START ):
            # Extract parameters
            snap, idxs, bhids, numMergers = task
            if( debug ): print "%d : starting snapshot %d" % (rank, snap)
            beg = datetime.now()

            data, pos, neg, new = _loadSingleSnapshotBHs(run, snap, numMergers, idxs, bhids, 
                                                   rank=rank, loadsave=loadsave, debug=debug)



            end = datetime.now()
            durat = (end-beg).total_seconds()
            if( debug ): print "%d : sending done after %f" % (rank, durat)
            comm.send([durat,pos,neg,new], dest=0, tag=TAGS.DONE)
        elif( tag == TAGS.EXIT  ):
            if( debug ): print "%d : received exit" % (rank)
            break


    # Finish, return done
    if( debug ): print "%d : sending exit" % (rank)
    comm.send(None, dest=0, tag=TAGS.EXIT)

    return
    
# _runSlave()



def _loadSingleSnapshotBHs(run, snapNum, numMergers, idxs, bhids, 
                           rank=0, loadsave=True, debug=False):
    """
    Load the data for BHs in a single snapshot, save to npz file.

    If no indices (``idxs``) or BH IDs (``bhids``) are given, or this is a 'bad' snapshot,
    then it isn't actually loaded and processed.  An NPZ file with all zero entries is still
    produced.

    """
    illdir = GET_ILLUSTRIS_OUTPUT_DIR(run)
    fname = _GET_BH_SINGLE_SNAPSHOT_FILENAME(run, snapNum)
    if( debug ): print "%d : snap %d - filename '%s'" % (rank, snapNum, fname)

    pos = 0
    neg = 0
    new = 0

    ## Load and Return existing save if desired
    #  ----------------------------------------
    if( loadsave and os.path.exists(fname) ):
        if( debug ): print "%d : file exists for snap %d" % (rank, snapNum)
        data = zio.npzToDict(fname)
        return data, pos, neg, new


    ## Initialize dictionary of results
    #  --------------------------------
    if( debug ): print "%d : initializing storage" % (rank)
    data = _initStorage(numMergers)
    for index,tid in zip(idxs,bhids):
        for BH in [BH_TYPE.IN, BH_TYPE.OUT]:
            data[SNAP.TARGET][index,BH] = tid[BH]


    ## Decide if this is a valid Snapshot
    #  ----------------------------------
    process_snapshot = True
    # Some illustris-1 snapshots are bad
    if( snapNum in GET_BAD_SNAPS(run) ): 
        if( debug ): print "%d : skipping bad snapshot %d" % (rank, snapNum)
        process_snapshot = False

    # Make sure there are mergers in this snapshot
    if( len(idxs) <= 0 or len(bhids) <= 0 ):
        if( debug ): print "%d : skipping snap %d with no valid BHs" % (rank, snapNum)
        process_snapshot = False


    ## Load And Process Snapshot if its good
    #  =====================================
    if( process_snapshot ):

        ## Load Snapshot
        #  -------------
        if( debug ): print "%d : loading snapshot %d" % (rank, snapNum)
        with zio.StreamCapture() as output:
            snapshot = ill.snapshot.loadSubset(illdir, snapNum, 'bh', fields=SNAPSHOT_FIELDS)

        snap_keys = snapshot.keys()
        if( 'count' in snap_keys ): 
            snap_keys.remove('count')
            if( debug ): print "%d : Loaded %d particles" % (rank, snapshot['count'])

        # Make sure all target keys are present
        union = list(set(snap_keys) & set(SNAPSHOT_FIELDS))
        if( len(union) != len(SNAPSHOT_FIELDS) ):
            print ""
            print "%d: snap_keys       = " % (rank), snap_keys
            print "%d: SNAPSHOT_FIELDS = " % (rank), SNAPSHOT_FIELDS
            raise RuntimeError("Rank %d, Snap %d : fields mismatch!" % (rank, snapNum))

        ## Match target BHs
        #  ----------------
        if( debug ): print "%d : matching %d BH Mergers" % (rank, len(bhids))
        for index,tid in zip(idxs,bhids):
            for BH in [BH_TYPE.IN, BH_TYPE.OUT]:
                ind = np.where( snapshot['ParticleIDs'] == tid[BH] )[0]
                if( len(ind) == 1 ):
                    pos += 1
                    data[SNAP.VALID][index,BH] = True
                    for key in SNAPSHOT_FIELDS: data[key][index,BH] = snapshot[key][ind[0]]
                else:
                    neg += 1

            # BH
        # index, tid

        if( debug ): print "%d : pos %d, neg %d" % (rank, pos, neg)

    # } if( process_snapshot )


    ## Add Metadata and Save File
    #  ==========================
    data[SNAP.RUN]     = run
    data[SNAP.SNAP]    = snapNum
    #data[SNAP.VALID]   = valid
    #data[SNAP.TARGET]  = targets
    data[SNAP.VERSION] = _VERSION
    data[SNAP.CREATED] = datetime.now().ctime()
    data[SNAP.DIR_SRC] = illdir

    zio.dictToNPZ(data, fname, verbose=debug)
    new = 1

    return data, pos, neg, new

# _loadSingleSnapshotBHs()



def _mergeBHSnapshotFiles(run, verbose=True, debug=False):

    if( verbose ): print " - - BHSnapshotData._mergeBHSnapshotFiles()"

    snapList = np.arange(NUM_SNAPS-1)
    np.random.shuffle(snapList)
    count = 0
    newFiles = 0
    oldFiles = 0
    num_pos = 0
    num_neg = 0
    num_val = 0
    num_tar = 0


    ## Load BH Mergers
    if( verbose ): print " - - - Loading BH Mergers"
    mergers = BHMergers.loadFixedMergers(run, verbose=verbose, loadsave=True)
    numMergers = mergers[MERGERS.NUM]
    if( verbose ): print " - - - - Loaded %d mergers" % (numMergers)

    allData = _initStorage(numMergers)    

    ## Load each snapshot file
    #  -----------------------
    if( verbose ): print " - - - Iterating over snapshots"
    beg = datetime.now()
    if( verbose and not debug ): pbar = zio.getProgressBar(NUM_SNAPS-1)
    for snap in snapList:

        mrgs = mergers[MERGERS.MAP_STOM][snap+1]
        nums = len(mrgs)
        targetIDs = mergers[MERGERS.IDS][mrgs]

        ## Load Snapshot Data
        data, pos, neg, new = _loadSingleSnapshotBHs(run, snap, numMergers, mrgs, targetIDs,
                                                     loadsave=True, debug=debug)
        
        ## Store to global dictionary
        valids = data[SNAP.VALID]
        numValid   = np.count_nonzero(valids)
        numTargets = np.count_nonzero(data[SNAP.TARGET] > 0)
        
        # Copy valid elements
        allData[SNAP.TARGET][valids] = data[SNAP.TARGET][valids]
        allData[SNAP.VALID][valids] = data[SNAP.VALID][valids]
        for key in SNAPSHOT_FIELDS:
            allData[key][valids] = data[key][valids]


        if( new == 1 ): 
            newFiles += 1
            if( verbose ): 
                print " - - - Snap %d : new" % (snap)
                print " - - - - pos %d, neg %d, (sum %d) expected %d" % (pos, neg, nums)
                print " - - - - Targets %d, Valid %d" % (numTargets, numValid)
        else:
            oldFiles += 1
            if( verbose ): 
                pos = numValid
                neg = 2*nums - pos
                print " - - - Snap %d : new" % (snap)
                print " - - - - pos %d, expected %d, neg %d" % (pos, nums, neg)
                print " - - - - Targets %d, Valid %d" % (numTargets, numValid)
        
        num_pos += pos
        num_neg += neg
        num_val += numValid
        num_tar += numTargets
        count += 1
        if( verbose and not debug ): pbar.update(count)

    # snap

    if( verbose and not debug ): pbar.finish()
    end = datetime.now()
    if( verbose ): 
        print " - - - Done after %s" % (str(end-beg))
        print " - - - - %d new, %d old.  Pos %d, Neg %d" % (newFiles, oldFiles, num_pos, num_neg)
        print " - - - - Targets %d, Valid %d" % (num_tar, num_val)
        numValid = np.count_nonzero(allData[SNAP.VALID])
        print " - - - - %d/%d = %.4f valid" % (numValid, 2*numMergers, 0.5*numValid/numMergers)


    return allData

# _mergeBHSnapshotFiles()



def _initStorage(numMergers):
    data = {}
    for key,typ in zip(SNAPSHOT_FIELDS, SNAPSHOT_DTYPES):
        data[key] = np.zeros([numMergers, 2], dtype=np.dtype(typ))

    data[SNAP.VALID]  = np.zeros([numMergers,2], dtype=bool)
    data[SNAP.TARGET] = np.zeros([numMergers,2], dtype=DTYPE.ID)

    return data


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


