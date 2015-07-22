"""


"""

import numpy as np
import scipy as sp
from datetime import datetime
from enum import Enum
import os, sys

from mpi4py import MPI

from illpy.Constants import NUM_SNAPS, GET_ILLUSTRIS_OUTPUT_DIR
from illpy.illbh import BHMergers
from illpy.illbh.BHConstants import MERGERS, BH_TYPE

import Settings

import illustris_python as ill

import zcode
import zcode.InOut as zio

_VERSION = 0.2

_SAVE_FILE_NAME = "./data/ill-%d_bh_snapshot_data_v-%.1f.npz"
def GET_SAVE_FILE_NAME(run, vers): return _SAVE_FILE_NAME % (run, vers)

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

    ## Master Process
    #  --------------
    if( rank == 0 ):
        print "RUN           = %d  " % (run)
        print "VERSION       = %.2f" % (_VERSION)
        print "MPI COMM SIZE = %d  " % (size)
        print ""
        print "VERBOSE       = %s  " % (str(verbose))
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



def _runMaster(run, comm):
    """
    Run master process which manages all of the secondary ``slave`` processes.

    Details
    -------

    """

    stat = MPI.Status()
    rank = comm.rank
    size = comm.size

    print " - Initializing"

    mergSnap, snapMerg, mergSubh = getMergerAndSubhaloIndices(run, verbose=True)

    # Get all subhalos for each snapshot (including duplicates and missing)
    snapSubh     = [ mergSubh[smrg] for smrg in snapMerg ]
    # Get unique subhalos for each snapshot, discard duplicates
    snapSubh_uni = [ np.array(list(set(ssubh))) for ssubh in snapSubh ]
    # Discard missing matches ('-1')
    snapSubh_uni = [ ssubh[np.where(ssubh != -1)] for ssubh in snapSubh_uni ]

    numUni = [len(ssubh) for ssubh in snapSubh_uni]
    numUniTot = np.sum(numUni)
    numMSnaps = np.count_nonzero(numUni)

    print " - - %d Unique subhalos over %d Snapshots" % (numUniTot, numMSnaps)

    ## Iterate over Snapshots and Subhalos
    #  ===================================
    #     distribute tasks to slave processes
    
    count = 0
    new   = 0
    exist = 0
    fail  = 0
    times = np.zeros(numUniTot)

    statFileName = GET_ENVIRONMENTS_STATUS_FILENAME(run)
    statFile = open(statFileName, 'w')
    print " - - Opened status file '%s'" % (statFileName)
    statFile.write('%s\n' % (str(datetime.now())))
    beg = datetime.now()

    for snap,subs in zmath.renumerate(snapSubh_uni):

        if( len(subs) <= 0 ): continue

        # Create output directory (subhalo doesn't matter since only creating dir)
        #    don't let slave processes create it - makes conflicts
        fname = GET_MERGER_SUBHALO_FILENAME(run, snap, 0)
        zio.checkPath(fname)

        # Get most bound particles for each subhalo in this snapshot
        mostBound = Subhalo.importGroupCatalogData(run, snap, subhalos=subs, 
                                                   fields=[SUBHALO.MOST_BOUND], verbose=False)

        # Go over each subhalo
        for boundID, subhalo in zip(mostBound, subs):

            # Write status to file
            dur = (datetime.now()-beg)
            statStr = 'Snap %3d   %8d/%8d = %.4f   in %s   %8d new   %8d exist  %8d fail\n' % \
                (snap, count, numUniTot, 1.0*count/numUniTot, str(dur), new, exist, fail)
            statFile.write(statStr)
            statFile.flush()

            # Look for available slave process
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
            source = stat.Get_source()
            tag = stat.Get_tag()

            # Track number of completed profiles
            if( tag == TAGS.DONE ): 
                retStat, durat = data

                times[count] = durat
                count += 1
                if(   retStat == ENVSTAT.NEWF ): new   += 1
                elif( retStat == ENVSTAT.EXST ): exist += 1
                else:                            fail  += 1


            # Distribute tasks
            comm.send([snap, subhalo, boundID], dest=source, tag=TAGS.START)

        # } for boundID, subhalo 

    # } for snap, subs

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
                times[count] = data[1]
                count += 1
                if( data[0] ): new += 1

            # Send exit command
            comm.send(None, dest=source, tag=TAGS.EXIT)



    print " - - %d/%d = %.4f Completed tasks!" % (count, numUniTot, 1.0*count/numUniTot)
    print " - - %d New Files" % (new)

    return
    
# _runMaster()




def _runSlave(run, comm, radBins=None, loadsave=True, verbose=False):
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

    if( verbose ): print " - - Environments._runSlave() : rank %d/%d" % (rank, size)

    # Keep looking for tasks until told to exit
    while True:
        # Tell Master this process is ready
        comm.send(None, dest=0, tag=TAGS.READY)
        # Receive ``task`` ([snap,boundID,subhalo])
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=stat)
        tag = stat.Get_tag()

        if( tag == TAGS.START ):
            # Extract parameters of environment
            snap, subhalo, boundID = task
            beg = datetime.now()
            # Load and save Merger Environment
            retEnv, retStat = loadMergerEnv(run, snap, subhalo, boundID, radBins=radBins, 
                                            loadsave=True, verbose=verbose)
            end = datetime.now()
            durat = (end-beg).total_seconds()
            comm.send([retStat,durat], dest=0, tag=TAGS.DONE)
        elif( tag == TAGS.EXIT  ):
            break


    # Finish, return done
    comm.send(None, dest=0, tag=TAGS.EXIT)

    return
    
# _runSlave()





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
    
    saveFileName = GET_SAVE_FILE_NAME(run, _VERSION)

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

    dir_illustris = GET_ILLUSTRIS_OUTPUT_DIR(run)
    if( verbose ): print " - - - Using illustris data dir '%s'" % (dir_illustris)

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


