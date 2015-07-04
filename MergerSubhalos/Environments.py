"""
Load Subhalo and environmental data corresponding to Merger BHs.

Classes
-------
   ENVIRON : enumerator-like object for managing subhalo (environment) parameters dictionaries
   TAGS    : enumerator-like object for managing MPI communication


Functions
---------
   GET_MERGER_SUBHALO_FILENAME          : get filename for individual subhalo file
   GET_MISSING_MERGER_SUBHALO_FILENAME  : get filename for list of missing subhalos
   GET_MERGER_ENVIRONMENT_FILENAME      : get filename for dictionary of all subhalos

   getMergerAndSubhaloIndices           : get merger, snapshot and subhalo index numbers
   checkSubhaloFiles                    : check which subhalo files exist or are missing
   loadMergerEnvironments               : primary API - load all subhalo environments as dict

   _runMaster                           : process manages all secondary ``slave`` processes
   _runSlave                            : secondary process loads and saves data for each subhalo
   _collectMergerEnvironments           : merge each subhalo environment file into single dict
   _initStorage                         : initializes dict to store data for all subhalos
   _parseArguments                      : parse commant line arguments
   _mpiError                            : raise an error through MPI and exit all processes   

"""


import numpy as np
from datetime import datetime
import sys
import os
import argparse

# progressbar2 (https://pypi.python.org/pypi/progressbar2)
import progressbar

from mpi4py import MPI
from enum import Enum

import zcode.InOut as zio
import zcode.Math  as zmath
from zcode.Constants import PC

from illpy.illbh import BHMergers, BHMatcher
from illpy.illbh.BHConstants import MERGERS_NUM, MERGERS_MAP_MTOS, MERGERS_MAP_STOM
from illpy.Subhalos import Subhalo, Profiler
from illpy.Subhalos.Constants import SUBHALO
from illpy.Constants import DIST_CONV, PARTICLE_NUM

from GalaxyModels.MergerSubhaloCatalog import loadMergerSubhaloCatalog, MSUBH_SUBHALO_INDICES

from Settings import DIR_DATA, DIR_PLOTS

# Hard Settings
_VERSION      = 1.2

# Soft (Default) Settings (Can be changed from command line)
VERBOSE      = False
CHECK_EXISTS = True
RUN          = 3

RAD_BINS     = 100
RAD_EXTREMA  = [1.0, 1.0e7]     # PC (converted to simulation units)


PLOT_DIR = DIR_PLOTS + "merger-subhalos/"


class ENVIRON():
    """ Keys for dictionary of subhalo environmental parameters.  See source for documentation."""

    # Meta Data
    RUN   = "run"                                 # Illustris simulation number {1,3}
    SNAP  = "snap"                                # Illustris snapshot   number {1,135}
    VERS  = "version"                             # ``Environments`` version number
    DATE  = "created"                             # datetime of creation

    # For each Subhalo:
    SUBH  = "subhalo"                             # Subhalo (index) number corresponding to catalog
    BPID  = "boundid"                             # Mostbound Particle ID number for each subhalo
    CENT  = "center"                              # Center position of subhalo (most-bound particle)
    TYPE  = "types"                               # Illustris Particle type numbers present
    NAME  = "names"                               # Illustris Particlr type names

    # Radial profiles
    RADS  = "rads"                                # Positions of right-edges of radial bins 
    NUMS  = "nums"                                # Number of particles by type in each bin
    MASS  = "mass"                                # Mass   of particle  by type in each bin
    DENS  = "dens"                                # Dens (ave)          by type    each bin
    POTS  = "pots"                                # Grav potential for all types   each bin
    DISP  = "disp"                                # Vel dispersion     all types   each bin

    GCAT_KEYS = "cat_keys"                        # Parameters of group-catalog entries included

# } class ENVIRON


class TAGS():
    READY = 0
    START = 1
    DONE  = 2
    EXIT  = 3

# } class TAGS


_MERGER_SUBHALO_FILENAME_BASE = ( DIR_DATA + "merger_subhalos/Illustris-%d/snap%03d/" + 
                                  "ill%d_snap%03d_subhalo%06d_v%.2f.npz" )

def GET_MERGER_SUBHALO_FILENAME(run, snap, subhalo, version=_VERSION):
    return _MERGER_SUBHALO_FILENAME_BASE % (run, snap, run, snap, subhalo, version)


_MISSING_MERGER_SUBHALO_FILENAME = ( DIR_DATA + "merger_subhalos/" + 
                                     "ill%d_missing_merger-subhalos_v%.2f.txt" )

def GET_MISSING_MERGER_SUBHALO_FILENAME(run, version=_VERSION):
    return _MISSING_MERGER_SUBHALO_FILENAME % (run, version)


_MERGER_ENVIRONMENT_FILENAME = ( DIR_DATA + "ill%d_merger-environments_v%.2f.npz" )
def GET_MERGER_ENVIRONMENT_FILENAME(run, version=_VERSION):
    return _MERGER_ENVIRONMENT_FILENAME % (run, version)




def getMergerAndSubhaloIndices(run, verbose=True):
    """
    Get indices of mergers, snapshots and subhalos.

    Arguments
    ---------

    Returns
    -------
       mergSnap <int>[N]     : snapshot number for each merger
       snapMerg <int>[135][] : list of merger indices for each snapshot
       mergSubh <int>[N]     : subhalo index number for each merger

    """

    if( verbose ): print " - - Environments.getMergerAndSubhaloIndices()"

    if( verbose ): print " - - Loading Mergers"
    mergers = BHMergers.loadFixedMergers(run, verbose=VERBOSE)
    if( verbose ): print " - - - Loaded %d mergers" % (mergers[MERGERS_NUM])

    if( verbose ): print " - - Loading Merger Subhalos Catalog"
    mergerSubhalos = loadMergerSubhaloCatalog(run, verbose=VERBOSE)

    if( verbose ): print " - - Loading Merger Details"
    mergerDetails  = BHMatcher.loadMergerDetails(run, mergers=mergers, verbose=VERBOSE)


    # Snapshot for each merger
    mergSnap = mergers[MERGERS_MAP_MTOS]
    # Mergers for each snapshot
    snapMerg = mergers[MERGERS_MAP_STOM]
    # Subhalos for each merger
    mergSubh = mergerSubhalos[MSUBH_SUBHALO_INDICES]

    return mergSnap, snapMerg, mergSubh

# getMergerAndSubhaloIndices()



def _runMaster(run, comm):
    """
    Run master process which manages all of the secondary ``slave`` processes.

    Details
    -------
     - Retrieves merger, snapshot and subhalo indices
     - Iterates over snapshots and merger-subhalo pairs, distributing them to ``slave`` processes
       which load each subhalo profile and writes them to individual-subhalo files.
       - Loads most-bound particle ID numbers from group caalog for each snapshot and distributes
         this to each slave-process as-well.
       - Tracks how-many and which process (and subhalos) finish successfully

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
    new = 0
    times = np.zeros(numUniTot)
    statFileName = 'stat_MergerSubhalos_ill%d_v%.2f.txt' % (run, _VERSION)
    statFile = open(statFileName, 'w')
    print " - - Opened status file '%s'" % (statFileName)
    statFile.write('%s\n' % (str(datetime.now())))
    beg = datetime.now()

    for snap,subs in zmath.renumerate(snapSubh_uni):
        # for snap,subs in zip([135,134], [range(200,210), [0,2]]):

        if( len(subs) <= 0 ): continue

        # Create output directory (subhalo doesn't matter since only creating dir)
        #    don't let slave processes create it - makes conflicts
        fname = GET_MERGER_SUBHALO_FILENAME(run, snap, 0)
        zio.checkPath(fname)

        # Get most bound particles for each subhalo in this snapshot
        mostBound = Subhalo.importGroupCatalogData(run, snap, subhalos=subs, 
                                                   fields=[SUBHALO.MOST_BOUND], verbose=VERBOSE)

        # Go over each subhalo
        for boundID, subhalo in zip(mostBound, subs):

            dur = (datetime.now()-beg)
            statFile.write('Snap %3d   %8d/%8d = %.4f   in %s   %8d new\n' % 
                           (snap, count, numUniTot, 1.0*count/numUniTot,str(dur), new) )
            statFile.flush()

            # Look for available slave process
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
            source = stat.Get_source()
            tag = stat.Get_tag()
            # Track number of completed profiles
            if( tag == TAGS.DONE ): 
                times[count] = data[1]
                count += 1
                if( data[0] ): new += 1

            # Distribute tasks
            comm.send([snap, boundID, subhalo], dest=source, tag=TAGS.START)

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




def _runSlave(run, comm, radBins=None, loadsave=True):
    """
    Secondary process which continually receives subhalo numbers from ``master`` to load and save.

    Arguments
    ---------
       run      <int>       : illustris simulation run number {1,3}
       comm     <...>       : MPI intracommunicator object (e.g. ``MPI.COMM_WORLD``)
       radBins  <scalar>[N] : optional, positions of right-edges of radial bins
       loadsave <bool>      : optional, load data for this subhalo if it already exists

    Details
    -------
     - Waits for ``master`` process to send subhalo numbers
     - Loads existing save of subhalo data if possible (and ``loadsave``), otherwise re-imports it
     - Returns status to ``master``

    """

    stat = MPI.Status()
    rank = comm.rank
    size = comm.size

    # Keep looking for tasks until told to exit
    while True:
        # Tell Master this process is ready
        comm.send(None, dest=0, tag=TAGS.READY)
        # Receive ``task`` ([snap,boundID,subhalo])
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=stat)
        tag = stat.Get_tag()

        if( tag == TAGS.START ):
            snap, boundID, subhalo = task
            beg = datetime.now()

            '''
            retval = loadSubhaloRadialProfiles(run, snap, boundID, subhalo, radBins=radBins,
                                               check=CHECK_EXISTS, verbose=VERBOSE)
            '''
            
            fname = GET_MERGER_SUBHALO_FILENAME(run, snap, subhalo)
            retval = False

            # If we shouldnt or cant load existing save, reload profiles
            if( not loadsave or not os.path.exists(fname) ): 

                # Load Radial Profiles
                outRadBins, posRef, partTypes, partNames, numsBins, \
                    massBins, densBins, potsBins, dispBins = \
                    Profiler.loadSubhaloRadialProfiles(run, snap, subhalo, radBins=radBins, 
                                                       mostBound=boundID, verbose=False)

                # Build dict of data
                env = { 
                    ENVIRON.RUN  : run,
                    ENVIRON.SNAP : snap,
                    ENVIRON.BID  : boundID,
                    ENVIRON.SUBH : subhalo,
                    ENVIRON.VERS : _VERSION,
                    ENVIRON.DATE : str(datetime.now()),

                    ENVIRON.COM  : posRef,
                    ENVIRON.RADS : outRadBins,
                    ENVIRON.TYPE : partTypes,
                    ENVIRON.NAME : partNames,
                    ENVIRON.NUMS : numsBins,
                    ENVIRON.MASS : massBins,
                    ENVIRON.DENS : densBins,
                    ENVIRON.POTS : potsBins,
                    ENVIRON.DISP : dispBins
                    }

                # Save Data as NPZ file
                zio.dictToNPZ(env, fname, verbose=verbose)

            # } if 

            end = datetime.now()
            durat = (end-beg).total_seconds()
            comm.send([retval,durat], dest=0, tag=TAGS.DONE)
        elif( tag == TAGS.EXIT  ):
            break


    # Finish, return done
    comm.send(None, dest=0, tag=TAGS.EXIT)

    return
    
# _runSlave()




def checkSubhaloFiles(run, verbose=True, version=_VERSION):
    """
    Check each Merger to make sure its subhalo profile has been found and saved.

    Writes missing merger subhalos to file ``GET_MISSING_MERGER_SUBHALO_FILENAME``.
    """
    
    if( verbose ): print " - - Environments.checkSubhaloFiles()"

    if( verbose ): print " - - - Initializing parameters"
    # Load indices for mergers, snapshots and subhalos
    mergSnap, snapMerg, mergSubh = getMergerAndSubhaloIndices(run, verbose=verbose)
    numMergers = len(mergSnap)

    missing_fname = GET_MISSING_MERGER_SUBHALO_FILENAME(run, version=version)

    widgets = [
        progressbar.Percentage(),
        ' ', progressbar.Bar(),
        ' ', progressbar.AdaptiveETA(),
    ]

    beg = datetime.now()
    count = 0
    numYa = 0
    numNo = 0

    # Open file to store missing entries
    with open(missing_fname, 'w') as missing:
        if( verbose ): print " - - - Opening file '%s'" % (os.path.split(missing_fname)[1])
        missing.write('# Merger Snap Subhalo\n')
        
        if( verbose ): print " - - - Checking files"
        # Start progress-bar
        with progressbar.ProgressBar(widgets=widgets, maxval=numMergers, term_width=100) as pbar:

            ## Iterate over all mergers
            #  ------------------------
            for ii, (snap, subh) in enumerate(zip(mergSnap, mergSubh)):
                fname = GET_MERGER_SUBHALO_FILENAME(run, snap, subh, version=version)
                count += 1

                # Count success
                if( os.path.exists(fname) ):
                    numYa += 1
                # Count failure, save to output file
                else:
                    numNo += 1
                    missing.write('%6d  %3d  %6d\n' % (ii, snap, subh))
                    missing.flush()

                # Update progress bar
                pbar.update(count)


    end = datetime.now()
    if( verbose ): print " - - - - Checked %d files after '%s'" % (count, end-beg)
    print " - - - - %d Files Found  " % (numYa)
    print " - - - - %d Files Missing" % (numNo)

    return

# checkSubhaloFiles()


def loadMergerEnvironments(run, loadsave=True, verbose=True, version=_VERSION):
    """
    Load all subhalo environment data as a dictionary with keys from ``ENVIRON``.
    
    Arguments
    ---------
       run      <int>  : illustris simulation run number, {1,3}
       loadsave <bool> : optional, load existing save if it exists, otherwise create new
       verbose  <bool> : optional, print verbose output
       version  <flt>  : optional, version number to load (can only create current version!)
       
    Returns
    -------
       env <dict> : all environment data for all subhalos, keys given by ``ENVIRON`` class

    """

    if( verbose ): print " - - Environments.loadMergerEnvironments()"

    fname = GET_MERGER_ENVIRONMENT_FILENAME(run, version=version)

    ## Try to Load Existing Save File
    #  ------------------------------
    if( loadsave ):
        if( verbose ): print " - - - Attempting to load saved file from '%s'" % (fname)
        if( os.path.exists(fname) ):
            env = zio.npzToDict(fname)
            if( verbose ): print " - - - Loaded.  Creation date: %s" % (env[ENVIRON.DATE])
        else:
            print " - - - File '%s' does not exist!" % (fname)
            loadsave = False


    ## Import environment data directly, and save
    #  ------------------------------------------
    if( not loadsave ):
        if( verbose ): print " - - - Importing Merger Environments, version %s" % (str(VERSION))
        env = _collectMergerEnvironments(run, verbose=verbose)
        zio.dictToNPZ(env, fname, verbose=True)


    return env

# loadMergerEnvironments()


def _collectMergerEnvironments(run, verbose=True):
    """
    Load each subhalo environment file and merge into single dictionary object.

    Parameters for dictionary are given by ``ENVIRON`` class.

    Notes
    -----
     - Require uptodate version (``_VERSION``)

    """
    
    if( verbose ): print " - - Environments._collectMergerEnvironments()"

    # Load indices for mergers, snapshots and subhalos
    mergSnap, snapMerg, mergSubh = getMergerAndSubhaloIndices(run, verbose=verbose)
    numMergers = len(mergSnap)

    # Get all subhalos for each snapshot (including duplicates and missing)
    snapSubh = [ mergSubh[smrg] for smrg in snapMerg ]

    # Initialize Storage
    sampleSnap = 135
    env = _initStorage(run, sampleSnap, snapSubh[sampleSnap], numMergers, verbose=verbose)


    count = 0
    widgets = [
        progressbar.Percentage(),
        ' ', progressbar.Bar(),
        ' ', progressbar.ETA(),
        '   Adaptive ', progressbar.AdaptiveETA(),
    ]

    # Initialize progressbar
    beg = datetime.now()
    with progressbar.ProgressBar(widgets=widgets, maxval=numMergers, term_width=100) as pbar:

        ### Iterate over each Snapshot
        #   ==========================
        for snap, (merg, subh) in zmath.renumerate(zip(snapMerg, snapSubh)):

            # Get indices of valid subhalos
            indsSubh = np.where( subh >= 0 )[0]
            # Skip this snapshot if no valid subhalos
            if( len(indsSubh) == 0 ): continue
            # Select corresponding merger indices
            indsMerg = np.array(merg)[indsSubh]


            ## Get Data from Group Catalog
            #  ---------------------------
            gcat = Subhalo.importGroupCatalogData(run, snap, subhalos=subh[indsSubh], verbose=False)

            # Extract desired data
            for key in env[ENVIRON.GCAT_KEYS]:
                env[key][indsMerg,...] = gcat[key][...]


            ## Iterate Over Each Merger and Subhalo
            #  ------------------------------------
            for ind_subh, ind_merg in zip(indsSubh,indsMerg):

                count += 1
                                        #
                fname = GET_MERGER_SUBHALO_FILENAME(run, snap, subh[ind_subh], version=_VERSION)
                # Skip if file doesn't exist (shouldn't happen)
                if( not os.path.exists(fname) ): 
                    if( verbose ): 
                        print "WARNING: run %d, snap %d, merger %d, subhalo file '%s' missing!" % \
                            (run, snap, merg[ind_subh], fname)

                    raise RuntimeError("THIS SHOULDNT HAPPEN!!!!!!!")
                    continue

                ## Load Group Catalog Data
                dat = np.load(fname)

                ## Make sure particle counts match
                #   Number of each particle type (ALL 6) from group catalog
                gcatLenType = np.array(env[SUBHALO.NUM_PARTS_TYPE][ind_merg])
                #   Indices of target particles (only 4) from subhalo profile files
                subhTypes   = env[ENVIRON.TYPE]
                #   Compare counts between group-cat and subhalo save file
                assert all(dat[ENVIRON.NUMS] == gcatLenType[subhTypes]), \
                    "Particle numbers do not match!"

                # Store Subhalo numbers for each merger
                env[ENVIRON.SUBH][ind_merg] = subh[ind_subh]

                ## Extract data
                env[ENVIRON.DENS][ind_merg,...] = dat[ENVIRON.DENS]
                env[ENVIRON.MASS][ind_merg,...] = dat[ENVIRON.MASS]
                env[ENVIRON.POTS][ind_merg,...] = dat[ENVIRON.POTS]
                env[ENVIRON.DISP][ind_merg,...] = dat[ENVIRON.DISP]

                env[ENVIRON.NUMS][ind_merg,...] = dat[ENVIRON.NUMS]

                # Update progessbar
                pbar.update(count)

            # } for ind_subh, ind_merg

        # } for snap, (merg, subh)

    # } with pbar

    end = datetime.now()
    if( verbose ): print " - - - Completed %d/%d Mergers after %s" % (count, numMergers, end-beg)

    return env

# _collectMergerEnvironments()


def _initStorage(run, snap, subhalos, numMergers, verbose=True):
    """
    Use data from a sample subhalo to shape and initialize a dictionary for storage.

    Arguments
    ---------
       run        <int>    : Illustis simulation number {1,3}
       snap       <int>    : Illustris snapshot number {1,135}
       subhalos   <int>[N] : List of merger subhalos for this snapshot
       numMergers <int>    : Total Number of mergers
       verbose    <bool>   : print verbose output

    Returns
    -------
       env <dict> : Dictionary to store environmental data with space for radial profiles
                    and subhalo catalog data

    Notes
    -----
     - Requires that version is current (i.e. ``_VERSION``)
     - Subhalo profiles only store some particle types (used ones), while some subhalo catalog
       entries store all of them ``PARTICLE_NUM`` (6)

    """

    if( verbose ): print " - - Environments._initStorage()"

    env = {}

    if( verbose ): print " - - - Finding sample subhalo"

    # Find Sample Halo
    inds = np.where( subhalos >= 0 )[0]
    sample = np.min(inds)

    ## Radial Profiles for Sample Halo
    #  ------------------------------------
    if( verbose ): print " - - - Loading Profiles for Sample: Snap %d, Subhalo %d" % (snap, sample)
    fname = GET_MERGER_SUBHALO_FILENAME(run, snap, subhalos[sample], version=_VERSION)
    subh = np.load(fname)

    # Find shape of arrays for each Subhalo
    #    [numParticles, numRadialBins]
    shape_type = np.shape(subh[ENVIRON.DENS])
    #    [numRadialBins]
    shape_all  = np.shape(subh[ENVIRON.DISP])

    # Double check number of particles and radial bins
    numTypes = len(subh[ENVIRON.NAME])
    numRBins = len(subh[ENVIRON.RADS])

    # Make sure lengths are consistent
    assert shape_type[0] == numTypes, "Number of particle types doesnt match!!"
    assert shape_type[1] == numRBins, "Number of radial bins    doesnt match!!"

    # Report particle types (numbers and names)
    if( verbose ): 
        print " - - - Particle Types %s" % (str(['%6s' % name for name in subh[ENVIRON.NAME]]))
        print " - - - Particle Names %s" % (str(['%6d' % nums for nums in subh[ENVIRON.TYPE]]))

    # Construct shape for all subhalos
    shape_type = np.concatenate([[numMergers],shape_type])
    shape_all  = np.concatenate([[numMergers],shape_all])
    if( verbose ): print " - - - Shape of Profile Arrays = %s" % (str(shape_type))

    # Initialize meta data
    env[ENVIRON.RADS] = subh[ENVIRON.RADS]
    env[ENVIRON.RUN]  = subh[ENVIRON.RUN]
    env[ENVIRON.TYPE] = subh[ENVIRON.TYPE]
    env[ENVIRON.NAME] = subh[ENVIRON.NAME]
    env[ENVIRON.VERS] = _VERSION
    env[ENVIRON.DATE] = str(datetime.now())
    env[ENVIRON.SUBH] = np.zeros(numMergers, dtype=int)

    # Initialize Profiles Storage Manually
    #    [ mergers, part-types, rad-bins ]
    env[ENVIRON.DENS] = np.zeros(shape_type)
    env[ENVIRON.MASS] = np.zeros(shape_type)
    
    #    [ mergers, rad-bins ]
    env[ENVIRON.DISP] = np.zeros(shape_all)
    env[ENVIRON.POTS] = np.zeros(shape_all)
    #    [ mergers, part-types ]
    env[ENVIRON.NUMS] = np.zeros([numMergers, numTypes])

    ## Catalog for Sample Halo
    #  ------------------------------------
    if( verbose ): print " - - - Loading Catalog for Sample: Snap %d, Subhalo %d" % (snap, sample)
    gcat = Subhalo.importGroupCatalogData(run, snap, subhalos=sample, verbose=True)

    # Initialize catalog properties automatically
    env[ENVIRON.GCAT_KEYS] = gcat.keys()
    for key in gcat.keys():
        dat = gcat[key]
        shape = np.concatenate([[numMergers], np.shape(dat)])
        env[key] = np.zeros(shape)

    return env

# _initStorage()


def _parseArguments():
    """
    Prepare argument parser and load command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version='%s %.2f' % (sys.argv[0], _VERSION))
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Verbose output', default=VERBOSE)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check",    dest='check',   action="store_true", default=CHECK_EXISTS)
    group.add_argument("--no-check", dest='nocheck', action="store_true", default=(not CHECK_EXISTS))

    parser.add_argument("RUN", type=int, nargs='?', choices=[1, 2, 3],
                        help="illustris simulation number", default=RUN)
    args = parser.parse_args()
    
    return args

# _parseArguments()



### ====================================================================
#
#                           EXECUTABLE SCRIPT
#
### ====================================================================



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
    RUN     = args.RUN
    VERBOSE = args.verbose
    if(   args.check   ): CHECK_EXISTS = True
    elif( args.nocheck ): CHECK_EXISTS = False

    # Create Radial Bins
    radExtrema = np.array(RAD_EXTREMA)*PC/DIST_CONV
    radBins = zmath.spacing(radExtrema, num=RAD_BINS)

    ## Master Process
    #  --------------
    if( rank == 0 ):
        print "RUN           = %d  " % (RUN)
        print "VERSION       = %.2f" % (_VERSION)
        print "MPI COMM SIZE = %d  " % (size)
        print ""
        print "VERBOSE       = %s  " % (str(VERBOSE))
        print "CHECK_EXISTS  = %s  " % (str(CHECK_EXISTS))
        print ""
        print "RAD_BINS      = %d  " % (RAD_BINS)
        print "RAD_EXTREMA   = [%.2e, %.2e] [pc]" % (RAD_EXTREMA[0], RAD_EXTREMA[1])
        print "              = [%.2e, %.2e] [sim]" % (radExtrema[0], radExtrema[1])
        beg_all = datetime.now()

        try: 
            _runMaster(RUN, comm)
        except Exception as err:
            _mpiError(comm, err)


        # Check subhalo files to see if/what is missing
        checkSubhaloFiles(RUN, verbose=VERBOSE, version=_VERSION)

        end_all = datetime.now()
        print " - - Total Duration '%s'" % (str(end_all-beg_all))


    ## Slave Processes
    #  ---------------
    else:

        try:    
            _runSlave(RUN, comm, radBins)
        except Exception as err:
            _mpiError(comm, err)

            
    return 

# main()







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




if __name__ == "__main__": main()
