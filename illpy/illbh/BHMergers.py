"""
Module to handle Illustris BH Merger Files.

This module is an interface to the 'blackhole_mergers_<#>.txt' files produced
by Illustris.  Raw Illustris files are only used to initially load data, then
an intermediate numpy npz-file is produced to store a dictionary of merger data
for easier access in all future calls.  Executing the `main()` routine will
prepare the intermediate file, as will calls to the `loadMergers()` function -
if the intermediate file hasn't already been loaded.

The `mergers` are represented as a dictionary object with keys given by the
variables `MERGERS_*`, e.g. `MERGERS_NUM` is the key for the number of mergers.


Internal Parameters
-------------------
RUN : int, the illustris run number {1,3} to load by default
VERBOSE : bool, whether or not to print verbose output by default


Functions
---------
main : initialize the intermediate npz file
loadMergers : load merger data as a dictionary


Examples
--------

> # Load mergers from Illustris-2
> mergers = BHMergers.loadMergers(run=2, verbose=True)
> # Print the number of mergers
> print mergers[BHMergers.MERGERS_NUM]
> # Print the first 10 merger times
> print mergers[BHMergers.MERGERS_TIMES][:10]


Raises
------


Notes
-----
   The underlying data is in the illustris bh merger files, 'blackhole_mergers_<#>.txt', which are
   processed by `_loadMergersFromIllustris()`.  Each line of the input files is processed by
   `_parseMergerLine()` which returns the redshift ('time') of the merger, and the IDs and masses
   of each BH.  Each `merger` is sorted by time (redshift) in `_importMergers()` and placed in a
   `dict` of all results.  This merger dictionary is saved to a 'raw' savefile whose name is given 
   by `savedMergers_rawFilename()`.
   The method `processMergers()` not only loads the merger objects, but also creates mappings of
   mergers to the snapshots nearest where they occur (``mapM2S`) and visa-versa (``mapS2M``); as
   well as mergers which take place exactly during a snapshot iteration (``ontop``).  These three
   maps are included in the merger dictionary.

"""

import os, sys
from glob import glob
from datetime import datetime
import numpy as np

import BHDetails
import BHConstants
from BHConstants import *

from .. import illcosmo
from .. import AuxFuncs as aux



### Default Runtime Parameters ###
RUN = 3                                                                                             # Default Illustris run to load {1,3}
VERBOSE = True                                                                                      # Print verbose output during execution

### Internal Parameters ###
_VERSION = 0.1                                                                                      # This doesn't actually do anything currently

# Where to find the 'raw' Illustris Merger files
_MERGERS_FILE_DIRS = { 3:'/n/ghernquist/Illustris/Runs/L75n455FP/output/blackhole_mergers/' }
_MERGERS_FILE_NAMES = "blackhole_mergers_*.txt"

# Where to save intermediate files
_MERGERS_RAW_SAVE_FILENAME = "ill-%d_raw-mergers_v%.1f.npz"
_MERGERS_FIXED_SAVE_FILENAME = "ill-%d_fixed-mergers_v%.1f.npz"


### Internal / Operational Parameters ###
# Should not be changed during norma usage

savedMergers_rawFilename = lambda x: DATA_PATH + (_MERGERS_RAW_SAVE_FILENAME % (x, _VERSION))
savedMergers_fixedFilename = lambda x: DATA_PATH + (_MERGERS_FIXED_SAVE_FILENAME % (x, _VERSION))

_PRINT_INTERVAL_1 = 200                                                                             # When loading raw files, status interval
_PRINT_INTERVAL_2 = 20                                                                              # When fixing mergers, status interval


def main(run=RUN, verbose=VERBOSE):
    """
    Load mergers from raw illustris files, save to intermediate npz files.

    Arguments
    ---------
    run : int, Illustris simulation number {1,3}; default: `RUN`
    verbose : bool, verbose output during execution; default: `VERBOSE`

    """

    if( verbose ): print "\nBHMergers.py\n"
    if( verbose ): print " - Run '%d'" % (run)

    mergers = processMergers(run, verbose)

    return


def processMergers(run=RUN, verbose=VERBOSE, loadRaw=True):

    ### Load Mergers from Illustris Files ###
    if( verbose ): print " - Importing Mergers"
    savefile = savedMergers_rawFilename(run)
    # Load mergers directly from Illustris Files
    if( not loadRaw or not os.path.exists(savefile) ):
        if( verbose ): print " - - Directly from Illustris files..."
        mergers = _importMergers(run, verbose)

    # Load from 'raw' save file
    else:
        if( verbose ): print " - - From previous 'raw' save"
        merg = np.load(savefile)
        mergers = { key: merg[key] for key in merg.keys() }


    ### Create Mapping Between Mergers and Snapshots ###
    mapM2S, mapS2M, ontop = _mapToSnapshots(mergers)
    # Store mappings
    mergers[MERGERS_MAP_MTOS]  = mapM2S
    mergers[MERGERS_MAP_STOM]  = mapS2M
    mergers[MERGERS_MAP_ONTOP] = ontop

    ### Save 'Raw' Mergers ###
    if( verbose ): print " - Saving Raw Mergers"
    _saveRawMergers(mergers, run, verbose)

    return mergers
    # processMergers()




def loadMergers(run=RUN, verbose=VERBOSE):
    """
    Load Illustris BH Mergers data into a dictionary object.

    First try to load mergers from an existing save file (npz file), if that
    doesnt exist, load mergers directly from raw illustris files and save the
    data to an intermediate file for easier access in the future.

    Arguments
    ---------
    run : int, Illustris simulation number {1,3}; default: `RUN`
    verbose : bool, verbose output during execution; default: `VERBOSE`

    Returns
    -------
    mergers : dictionary, all BH merger data
        keys are given by the global `MERGERS_*` parameters

    """

    if( verbose ): print " - Loading mergers from save file"
    mergers = _loadMergersFromSave(run, verbose)

    # If mergers don't already exist, create them from raw illustris files
    if( mergers == None ): mergers = processMergers(run, verbose)

    return mergers



def _loadMergersFromSave(run=RUN, verbose=VERBOSE):

    savefile = savedMergers_fixedFilename(run)

    # Return None if no save-file exists
    if( not os.path.exists(savefile) ):
        if( verbose ): print " - - No savefile '%s' exists!" % (savefile)
        return None

    # Load mergers and basic properties

    mergers = np.load(savefile)

    if( verbose ):
        nums    = mergers[MERGERS_NUM]
        mergRun = mergers[MERGERS_RUN]
        created = mergers[MERGERS_CREATED]
        print " - - Loaded %d Mergers from '%s'" % (nums, savefile)
        print " - - - Run %d, saved at '%s'" % (mergRun, created)
        print " - - - File size '%s'" % ( aux.getFileSize(savefile) )

    return mergers



def _importMergers(run=RUN, verbose=VERBOSE):

    ### Get Illustris Merger Filenames ###
    if( verbose ): print " - - Searching for merger Files"
    mergerDir = _MERGERS_FILE_DIRS[run]
    mergerFilenames = mergerDir + _MERGERS_FILE_NAMES
    mergerFiles = sorted(glob(mergerFilenames))
    numFiles = len(mergerFiles)
    if( verbose ): print " - - - Found %d merger Files" % (numFiles)


    ### Count Mergers and Prepare Storage for Data ###
    numLines = aux.countLines(mergerFiles)
    times = np.zeros(numLines, dtype=_DOUBLE)
    ids = np.zeros([numLines,2], dtype=_LONG)
    masses = np.zeros([numLines,2], dtype=_DOUBLE)


    ### Load Raw Data from Merger Files ###
    if( verbose ): print " - - Loading Merger Data"
    _loadMergersFromIllustris(times, ids, masses, mergerFiles, verbose=verbose)
    if( verbose ): print " - - - Loaded %d entries" % (numLines)


    ### Sort Data by Time ###
    if( verbose ): print " - - Sorting Data"
    # Find indices which sort by time
    inds = np.argsort(times)
    # Use indices to reorder arrays
    times[:] = times[inds]
    ids[:] = ids[inds]
    masses[:] = masses[inds]


    ### Store Sorted Data to Dictionary and Save ###
    if( verbose ): print " - - Storing Data to Dictionary"
    # Store to dictionary
    mergers = { MERGERS_DIR     : mergerDir,
                MERGERS_RUN     : run,
                MERGERS_NUM     : numLines,
                MERGERS_TIMES   : times,
                MERGERS_IDS     : ids,
                MERGERS_MASSES  : masses,
                MERGERS_CREATED : datetime.now().ctime()
                }

    return mergers



def _saveRawMergers(mergers, run, verbose=VERBOSE):
    savefile = savedMergers_rawFilename(run)
    aux.saveDictNPZ(mergers, savefile, verbose)
    return

def _saveFixedMergers(mergers, run, verbose=VERBOSE):
    savefile = savedMergers_fixedFilename(run)
    aux.saveDictNPZ(mergers, savefile, verbose)
    return


def _loadMergersFromIllustris(times, ids, masses, files, verbose=VERBOSE):
    """
    Fill the given arrays with merger data from the given target files.

    Arrays ``ids`` and ``masses`` are shaped [N, 2], for ``N`` total mergers.
    By convention the
        'in'  (accreted) BH is index ``IN_BH`` (0?)
        'out' (accretor) BH is index ``OUT_BH`` (1?)


    Arguments
    ---------
    times : array[N], int
        Array to be filled with times (scale-factor) of each merger
    ids : array[N,2], long
        Array to be filled with BH ID numbers
    masses : array[N,2], double
        Array to be filled with BH masses
    files : array_like, list of files to load from.
        The total number of lines in all files must equal the array lengths `N`
    verbose : bool
        whether to print verbose output during execution

    """


    nums = len(times)

    count = 0
    for fil in files:
        for line in open(fil):
            # Get target elements from each line of file
            time, out_id, out_mass, in_id, in_mass = _parseMergerLine(line)
            # Store values
            times[count] = time
            ids[count,IN_BH] = in_id
            ids[count,OUT_BH] = out_id
            masses[count,IN_BH] = in_mass
            masses[count,OUT_BH] = out_mass
            # Increment Counter
            count += 1

            # Print Progress
            if( verbose ):
                if( count % _PRINT_INTERVAL_1 == 0 or count == nums):
                    sys.stdout.write('\r - - %.2f%% Complete' % (100.0*count/nums))

                if( count == nums ):
                    sys.stdout.write('\n')

                sys.stdout.flush()

    return



def _parseMergerLine(line):
    """
    Get target quantities from each line of the merger files.

    See 'http://www.illustris-project.org/w/index.php/Blackhole_Files' for
    details regarding the illustris BH file structure.

    The format of each line is:
        "PROC-NUM  TIME  ID1  MASS1  ID2  MASS2"
        see: http://www.illustris-project.org/w/index.php/Blackhole_Files
        where
            '1' corresponds to the 'out'/'accretor'/surviving BH
            '2' corresponds to the 'in' /'accreted'/eliminated BH
        NOTE: that MASS1 is INCORRECT (dynamical mass, instead of BH)

    Returns
    -------
    time     : scalar, redshift of merger
    out_id   : long, id number of `out` BH
    out_mass : scalar, mass of `out` BH in simulation units (INCORRECT VALUE)
    in_id    : long, id number of `in` BH
    in_mass  : scalar, mass of `in` BH in simulation units

    """

    strs     = line.split()
    # Convert to proper types
    time     = _DOUBLE(strs[1])
    out_id   = _LONG(strs[2])
    out_mass = _DOUBLE(strs[3])
    in_id    = _LONG(strs[4])
    in_mass  = _DOUBLE(strs[5])

    #return _DOUBLE(strs[1]), _LONG(strs[2]), _DOUBLE(strs[3]), _LONG(strs[4]), _DOUBLE(strs[5])
    return time, out_id, out_mass, in_id, in_mass



def _mapToSnapshots(mergers, verbose=VERBOSE):
    """
    Find the snapshot during which, or preceding each merger.

    mergers   : IN <ObjMergers.Mergers> all 'N' mergers
    snTimes   : IN < [float] > times of each snapshot
    (verbose) : IN <int> verbose output flag, >= 0 for true

    return

    mapS2M    : < [[int]] > length M, list of mergers for each snapshot
    ontop     : < [bool] > length N, true if merger happened ontop of snapshot
    """

    if( verbose ): print " - - _mapToSnapshots()"


    ### Initialize Variables ###
    if( verbose ): print " - - - Initializing parameters and Cosmology"
    nums = mergers[MERGERS_NUM]                                                                     # Total number of mergers

    # Load Cosmology
    cosmo = illcosmo.Cosmology()
    snapTimes = cosmo.snapshotTimes()                                                               # Scale-factor of each snapshot  

    # Map Mergers-2-Snapshots: snapshot before (or ontop) of each merger
    mapM2S = np.zeros(nums, dtype=_LONG)
    # Map Snapshots-2-Mergers: list of mergers just-after (or ontop) of each snapshot
    mapS2M = [ [] for ii in range(cosmo.num) ]
    # Flags if merger happens exactly on a snapshot (init to False=0)
    ontop  = np.zeros(nums, dtype=bool)


    ### Find snapshots on each side of merger time ###

    # Find the snapshot just below and above each merger.
    #     each entry (returned) is [ low, high, dist-low, dist-high ]
    #     low==high if the times match (within function's default uncertainty)
    snapBins = [ aux.findBins(mtime, snapTimes) for mtime in mergers[MERGERS_TIMES] ]


    ### Create Mappings ###

    if( verbose ): print " - - - Creating mappings"
    for ii, bins in enumerate(snapBins):
        tsnap = bins[0]                                                                             # Set snapshot to lower bin
        mapM2S[ii] = tsnap                                                                          # Set snapshot for this merger
        mapS2M[tsnap].append(ii)                                                                    # Add merger to this snapshot
        # If this merger takes place ontop of snapshot, set flag
        if( bins[0] == bins[1] ): ontop[ii] = True


    # Find the most mergers in a snapshot
    numPerSnap = np.array([len(s2m) for s2m in mapS2M ])
    mostMergers = np.max( numPerSnap )
    mostIndex = np.where( mostMergers == numPerSnap )[0]
    # Find the number of ontop mergers
    numOntop = np.count_nonzero(ontop)
    if( verbose ): print " - - - Most is %d mergers at snapshot %d" % (mostMergers, mostIndex)
    if( verbose ): print " - - - %d (%.2f) ontop mergers" % (numOntop, 1.0*numOntop/nums)

    return mapM2S, mapS2M, ontop



if __name__ == "__main__": main()
