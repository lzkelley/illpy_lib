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


Mergers Dictionary
------------------
{ MERGERS_NUM   : <int>, total number of mergers `N` ,
  MERGERS_TIMES : array(`N`, <int>), the time of each merger [scale-factor] , 
  

Examples
--------

>>> # Load mergers from Illustris-2
>>> mergers = BHMergers.loadMergers(run=2, verbose=True)
>>> # Print the number of mergers
>>> print mergers[BHMergers.MERGERS_NUM]
>>> # Print the first 10 merger times
>>> print mergers[BHMergers.MERGERS_TIMES][:10]


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



'''
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

    run = 3
    
    mergers = processMergers(run, verbose)

    return
'''



def processMergers(run, verbose=VERBOSE):

    if( verbose ): print " - - BHMergers.processMergers()"

    ### Load Mapped Mergers ###
    #   re-creates them if needed
    mergersMapped = loadMappedMergers(run, verbose=verbose)

    ### Load Fixed Mergers ###
    #mergersFixed = loadFixedMergers(run, verbose=verbose)

# processMergers()






def loadRawMergers(run, verbose=VERBOSE, recombine=False):
    """
    Load raw merger events into dictionary.

    Raw mergers are the data directly from illustris without modification.

    """

    if( verbose ): print " - - BHMergers.loadRawMergers()"


    ### Concatenate Raw Illustris Files into a Single Combined File ###

    combinedFilename = BHConstants.GET_MERGERS_RAW_COMBINED_FILENAME(run)
    if( recombine or not os.path.exists(combinedFilename) ):
        if( verbose ): print " - - Combining Illustris Merger files into '%s'" % (combinedFilename)
        mergerFilenames = BHConstants.GET_ILLUSTRIS_BH_MERGERS_FILENAMES(run)
        if( verbose ): print " - - - Found %d merger Files" % (len(mergerFilenames))
        aux.combineFiles(mergerFilenames, combinedFilename, verbose=verbose)


    if( verbose ): print " - - - Merger file '%s'" % (combinedFilename)


    ### Count Mergers and Prepare Storage for Data ###
    numLines = aux.countLines(combinedFilename)
    if( verbose ): print " - - - Merger Lines : %d" % (numLines)

    
    # Initialize Storage
    scales = np.zeros( numLines,               dtype=_DOUBLE)
    ids    = np.zeros([numLines,NUM_BH_TYPES], dtype=_LONG  )
    masses = np.zeros([numLines,NUM_BH_TYPES], dtype=_DOUBLE)


    ### Load Raw Data from Merger Files ###
    if( verbose ): print " - - Importing Merger Data"
    _importRawMergers(scales, ids, masses, mergerFilenames, verbose=verbose)


    ### Sort Data by Time ###
    if( verbose ): print " - - Sorting Data"

    # Find indices which sort by time
    inds = np.argsort(scales)
    # Use indices to reorder arrays
    scales[:] = scales[inds]
    ids[:] = ids[inds]
    masses[:] = masses[inds]


    return scales, ids, masses, combinedFilename




def loadMappedMergers(run, verbose=VERBOSE, remap=False ):
    """
    Load or create Mapped Mergers Dictionary as needed.
    """

    if( verbose ): print " - - BHMergers.loadMappedMergers()"

    mappedFilename = BHConstants.GET_MERGERS_RAW_MAPPED_FILENAME(run)

    if( not os.path.exists(mappedFilename) ):
        if( verbose ): print " - - - Mapped file '%s' does not exist" % (mappedFilename)
        remap = True

        
    ### Try to Load Existing Mapped Mergers ###
    if( not remap ):
        mergersMapped = np.load(mappedFilename)
        mergersMapped = aux.npzToDict(mergersMapped)
        if( verbose ): print " - - - Loaded from '%s'" % (mappedFilename)
        loadVers = mergersMapped[MERGERS_VERSION]
        # Make sure version matches, otherwise re-create mappings
        if( loadVers != VERSION ):
            loadTime = mergersMapped[MERGERS_CREATED]
            print "BHMergers.loadMappedMergers() : loaded version %f, from %s" % (loadVers, loadTime)
            print "BHMergers.loadMappedMergers() : VERSION %f, remapping!" % (VERSION)
            remap = True


    ### Recreate Mappings ###
    if( remap ):

        # Load Raw Mergers
        scales, ids, masses, filename = loadRawMergers(run, verbose=verbose)

        ### Create Mapping Between Mergers and Snapshots ###
        mapM2S, mapS2M, ontop = _mapToSnapshots(scales)

        # Store in dictionary
        mergersMapped = { MERGERS_FILE      : mappedFilename,
                          MERGERS_RUN       : run,
                          MERGERS_NUM       : len(scales),
                          MERGERS_CREATED   : datetime.now().ctime(),
                          MERGERS_VERSION   : VERSION,
                          
                          MERGERS_SCALES    : scales,
                          MERGERS_IDS       : ids,
                          MERGERS_MASSES    : masses,

                          MERGERS_MAP_MTOS  : mapM2S,
                          MERGERS_MAP_STOM  : mapS2M,
                          MERGERS_MAP_ONTOP : ontop,
                          }

        aux.saveDictNPZ(mergers, mappedFilename, verbose)


    return mergersMapped



def _importRawMergers(times, ids, masses, files, verbose=VERBOSE):
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

    if( verbose ): print " - - BHMergers._importRawMergers()"

    if( not np.iterable(files) ): files = [ files ]

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

# _importRawMergers()



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



def _mapToSnapshots(scales, verbose=VERBOSE):
    """
    Find the snapshot during which, or following each merger

    """

    if( verbose ): print " - - BHMergers._mapToSnapshots()"

    nums = len(scales)

    # Load Cosmology
    cosmo      = Cosmology()
    snapScales = cosmo.snapshotTimes()                                                               # Scale-factor of each snapshot  

    # Map Mergers-2-Snapshots: snapshot before (or ontop) of each merger
    mapM2S = np.zeros(nums, dtype=INT)
    # Map Snapshots-2-Mergers: list of mergers just-after (or ontop) of each snapshot
    mapS2M = [ [] for ii in range(cosmo.num) ]
    # Flags if merger happens exactly on a snapshot (init to False=0)
    ontop  = np.zeros(nums, dtype=bool)

    ### Find snapshots on each side of merger time ###

    # Find the snapshot just below and above each merger.
    #     each entry (returned) is [ low, high, dist-low, dist-high ]
    #     low==high if the times match (within function's default uncertainty)
    snapBins = [ aux.findBins(sc, snapScales) for sc in scales ]

    ### Create Mappings ###

    if( verbose ): print " - - - Creating mappings"
    for ii, bins in enumerate(snapBins):
        tsnap = bins[1]                                                                             # Set snapshot to upper bin
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
    if( verbose ): print " - - - Snapshot %d with the most (%d) mergers" % (mostIndex, mostMergers)
    if( verbose ): print " - - - %d (%.2f) ontop mergers" % (numOntop, 1.0*numOntop/nums)

    return mapM2S, mapS2M, ontop



if __name__ == "__main__": main()
