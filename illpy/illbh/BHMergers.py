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

"""

import os, sys
from glob import glob
from datetime import datetime
import numpy as np

import BHDetails
from .. import illcosmo
from .. import AuxFuncs as aux

__all__ = [ 'MERGERS_TIMES', 'MERGERS_IDS', 'MERGERS_MASSES', 'MERGERS_DIR',
            'MERGERS_RUN', 'MERGERS_CREATED', 'MERGERS_NUM',
            'loadMergers', 'main', 'RUN', 'VERBOSE' ]



### Default Runtime Parameters ###
RUN = 3                                                                                             # Default Illustris run to load {1,3}
VERBOSE = True                                                                                      # Print verbose output during execution

### Internal Parameters ###
_VERSION = 0.1                                                                                      # This doesn't actually do anything currently

# Where to find the 'raw' Illustris Merger files
_MERGERS_FILE_DIRS = { 3:'/n/ghernquist/Illustris/Runs/L75n455FP/output/blackhole_mergers/' }
_MERGERS_FILE_NAMES = "blackhole_mergers_*.txt"

# Where to save intermediate files
_SAVE_PATH = "%s/post-process/" % os.path.dirname(os.path.abspath(__file__))
_MERGERS_RAW_SAVE_FILENAME = "ill-%d_raw-mergers_v%.1f.npz"
_MERGERS_FIXED_SAVE_FILENAME = "ill-%d_fixed-mergers_v%.1f.npz"



### Internal / Operational Parameters ###
# Should not be changed during norma usage

savedMergers_rawFilename = lambda x: _SAVE_PATH + (_MERGERS_RAW_SAVE_FILENAME % (x, _VERSION))
savedMergers_fixedFilename = lambda x: _SAVE_PATH + (_MERGERS_FIXED_SAVE_FILENAME % (x, _VERSION))

_PRINT_INTERVAL_1 = 200                                                                             # When loading raw files, status interval
_PRINT_INTERVAL_2 = 20                                                                              # When fixing mergers, status interval

# Key Names for Mergers Dictionary
MERGERS_TIMES     = 'times'
MERGERS_IDS       = 'ids'
MERGERS_MASSES    = 'masses'
MERGERS_DIR       = 'dir'
MERGERS_RUN       = 'run'
MERGERS_CREATED   = 'created'
MERGERS_NUM       = 'num'
MERGERS_MAP_STOM  = 's2m'
MERGERS_MAP_MTOS  = 'm2s'
MERGERS_MAP_ONTOP = 'ontop'

# Index of [N,2] arrays corresponding to each BH
IN_BH  = 0
OUT_BH = 1


# Data Types
_DOUBLE = np.float64
_LONG = np.int64



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



def processMergers(run=RUN, verbose=VERBOSE, loadRaw=False, onlyRaw=False):

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

    # If only the 'raw' mergers are targeted, return them here
    if( onlyRaw ): return mergers

    ### Fix 'out' BH Mass in Mergers ###
    if( verbose ): print " - Fixing Mergers (using 'details')"
    _fixMergers(mergers, run, verbose)

    ### Save Fixed Mergers ###
    if( verbose ): print " - Saving Fixed Mergers"
    _saveFixedMergers(mergers, run, verbose)

    return mergers




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



def _fixMergers(mergers, run, verbose=VERBOSE):
    """
    Fix the accretor/'out' BH Mass using the blackhole details files.

    Merger files have an error in their output: the accretor BH (the 'out' BH
    which survives the merger process) mass is the 'dynamical' mass instead of
    the BH mass itself, see:
    http://www.illustris-project.org/w/index.php/Blackhole_Files

    This method finds the last entry in the Details files for the 'out' BH
    before the merger event, to 'fix' the recorded mass (i.e. to get the value
    from a different source).

    Details
    -------
    There are numerous complicating factors.  First: the details often aren't
    written at the same time as the mergers occur --- so there is a (small)
    temporal offset in the entries.  Second, and harder to deal with, is that
    some mergers happen soon enough after the next snapshot so that their BH
    didn't have a detail entry yet.  One solution to this would be to search
    the previous snapshot for the last valid entry for the BH that couldn't be
    found in the current snapshot... that's annoying.
    Instead
    


    """


    if( verbose ): print " - - _fixMergers()"

    # Import MatchDetails cython file
    import pyximport
    pyximport.install(setup_args={"include_dirs":np.get_include()})
    import MatchDetails

    numMergers = mergers[MERGERS_NUM]
    numSnaps = len(mergers[MERGERS_MAP_STOM])
    fixed = np.zeros(numMergers, dtype=bool)
    fracDiff = -1.0*np.ones(numMergers, dtype=float)
    fracDiff2 = -1.0*np.ones(numMergers, dtype=float)

    # Iterate over each snapshot, with list of mergers in each `s2m`
    count = 0
    start = datetime.now()
    for snap,s2m in enumerate(mergers[MERGERS_MAP_STOM]):

        # If there are no mergers in this snapshot, continue to next iteration
        if( len(s2m) <= 0 ): continue

        search = np.array(s2m)

        # Remove 'ontop' mergers (they merge before details are printed)
        #     in the previous snapshot, these mergers were added to the search list
        inds = np.where( mergers[MERGERS_MAP_ONTOP][search] )[0]
        search = np.delete(search, inds)

        # Add 'ontop' mergers from the next snapshot to search list
        if( snap < numSnaps-1 ):
            # Get the mergers from the next snapshot
            next = np.array(mergers[MERGERS_MAP_STOM][snap+1])
            if( len(next) > 0 ):
                # Filter to 'ontop' mergers
                inds = np.where( mergers[MERGERS_MAP_ONTOP][next] == True )[0]
                next = next[inds]
                # Add ontop mergers to list
                search = np.concatenate((search, next))


        # Get the details for this snapshot
        dets = BHDetails.loadBHDetails_NPZ(run, snap)
        detIDs = dets[BHDetails.DETAIL_IDS]
        detTimes = dets[BHDetails.DETAIL_TIMES]

        # If there are no details in this snapshot (should only happen at end), continue
        if( len(detIDs) <= 0 ): continue

        # Get the BH info for this snapshot
        bhids = mergers[MERGERS_IDS][search,OUT_BH]
        bhmasses = mergers[MERGERS_MASSES][search,OUT_BH]
        bhtimes = mergers[MERGERS_TIMES][search]

        # Find Details indices to match these BHs
        detInds, remInds = MatchDetails.detailsForBlackholes(bhids, bhtimes, detIDs, detTimes)

        # Find valid, normal matches
        inds = np.where( detInds >= 0 )[0]
        if( len(inds) > 0 ):
            fixMasses = dets[BHDetails.DETAIL_MASSES][detInds[inds]]
            fracDiff[search[inds]] = fixMasses/bhmasses[inds]
            mergers[MERGERS_MASSES][search[inds],OUT_BH] = fixMasses
            fixed[search[inds]] = True

        # Compensate for cases where only the combined remnant was found
        inds = np.where( remInds >= 0 )[0]
        if( len(inds) > 0 ):
            # The 'correct' mass is roughly the remnant mass, minus the 'accreted' BH
            fixMasses  = dets[BHDetails.DETAIL_MASSES][remInds[inds]] 
            fixMasses -= mergers[MERGERS_MASSES][search[inds],IN_BH]                                # Subtract off 'accreted' BH mass

            fracDiff2[search[inds]] = fixMasses/bhmasses[inds]
            mergers[MERGERS_MASSES][search[inds],OUT_BH] = fixMasses
            fixed[search[inds]] = True

        # Print progress
        count += len(bhids)
        if( verbose ):
            now = datetime.now()        
            statStr = aux.statusString(count, numMergers, now-start)
            sys.stdout.write('\r - - - %s' % (statStr))
            sys.stdout.flush()

    # } snap



    numFixed = np.count_nonzero(fixed == True)
    missing = np.where( fixed == False )[0]
    numMissing = len(missing)

    inds = np.where(fracDiff >= 0.0)[0]
    aveDiff = np.average(fracDiff[inds])
    stdDiff = np.std(fracDiff[inds])

    # Adjust down any missing entries
    if( numMissing > 0 ):
        fixMasses = aveDiff*mergers[MERGERS_MASSES][missing,OUT_BH]
        mergers[MERGERS_MASSES][missing,OUT_BH] = fixMasses

    if( verbose ):
        sys.stdout.write('\n')
        print " - - - %d Fixed, %d Missing -- (NOTE: Fixed manually!)" % (numFixed, numMissing)
        print " - - - Average fractional new mass = %.3e +- %.3e" % (aveDiff, stdDiff)

        inds = np.where(fracDiff2 >= 0.0)[0]
        print " - - - %d were remnant corrected" % (len(inds))
        if( len(inds) > 0 ):
            aveDiff = np.average(fracDiff2[inds])
            stdDiff = np.std(fracDiff2[inds])
            print " - - - - Average fractional new mass = %.3e +- %.3e" % (aveDiff, stdDiff)


    return




def _saveRawMergers(mergers, run, verbose=VERBOSE):
    savefile = savedMergers_rawFilename(run)
    _saveMergers(mergers, savefile, verbose)
    return

def _saveFixedMergers(mergers, run, verbose=VERBOSE):
    savefile = savedMergers_fixedFilename(run)
    _saveMergers(mergers, savefile, verbose)
    return

def _saveMergers(mergers, savefile, verbose=VERBOSE):
    """
    Save the given mergers dictionary to the given file.

    If the path to the given filename doesn't already exist, it is created.
    If ``verbose`` is True, the saved file size is printed out.
    """

    # Make sure path to file exists
    aux.checkPath(savefile)

    # Save and confirm
    np.savez(savefile, **mergers)
    if( not os.path.exists(savefile) ):
        raise RuntimeError("Could not save to file '%s'!!" % (savefile) )

    if( verbose ): print " - - Saved Mergers dictionary to '%s'" % (savefile)
    if( verbose ): print " - - - Size '%s'" % ( aux.getFileSize(savefile) )
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

    """

    strs = line.split()
    return _DOUBLE(strs[1]), _LONG(strs[2]), _DOUBLE(strs[3]), _LONG(strs[4]), _DOUBLE(strs[5])




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

#if __name__ == "__main__" and __package__ is None:
 #   __package__ = "ill_lib.illbh.BHMergers"

