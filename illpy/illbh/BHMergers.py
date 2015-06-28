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
VERSION_MAP <flt> : version number for 'mapped' mergers, and associated save files
VERSION_FIX <flt> : version number for 'fixed'  mergers, and associated save files


Functions
---------
processMergers                : perform all processing methods, assures that save files are all 
                                constructed.
loadRawMergers                : load all merger entries, sorted by scalefactor, directly from 
                                illustris merger files - without any processing/filtering.
loadMappedMergers             : load dictionary of merger events with associated mappings to and
                                from snapshots.
loadFixedMergers              : load dictionary of merger events with mappings, which have been
                                processed and filtered.  These mergers also have the 'out' mass
                                entry corrected (based on inference from ``BHDetails`` entries).
                                

Mergers Dictionary
------------------
   { MERGERS_RUN       : <int>, illustris simulation number in {1,3}
     MERGERS_NUM       : <int>, total number of mergers `N`
     MERGERS_FILE      : <str>, name of save file from which mergers were loaded/saved
     MERGERS_CREATED   : <str>, 
     MERGERS_VERSION   : <float>,
   
     MERGERS_SCALES    : <double>[N], the time of each merger [scale-factor]
     MERGERS_IDS       : <ulong>[N,2],
     MERGERS_MASSES    : <double>[N,2], 

     MERGERS_MAP_MTOS  : <int>[N], 
     MERGERS_MAP_STOM  : <int>[136,list], 
     MERGERS_MAP_ONTOP : <int>[136,list], 
   }


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
 - 'Raw Mergers' : these are mergers directly from the illustris files with NO modifications or
                   filtering of any kind.



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
from datetime import datetime
import numpy as np

import BHDetails
import BHConstants
from BHConstants import *
import BHMatcher

from .. import Cosmology
from .. import AuxFuncs as aux


# VERSION = 0.21

VERSION_MAP = 0.21
VERSION_FIX = 0.31


def processMergers(run, verbose=VERBOSE):

    if( verbose ): print " - - BHMergers.processMergers()"

    ### Load Mapped Mergers ###
    #   re-creates them if needed
    mergersMapped = loadMappedMergers(run, verbose=verbose)

    ### Load Fixed Mergers ###
    mergersFixed = loadFixedMergers(run, verbose=verbose)

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


    ### Load Raw Data from Merger Files ###
    if( verbose ): print " - - - Importing Merger Data"
    scales, ids, masses = _importRawMergers(combinedFilename, verbose=verbose)


    ### Sort Data by Time ###
    if( verbose ): print " - - - Sorting Data"

    # Find indices which sort by time
    inds = np.argsort(scales)
    # Use indices to reorder arrays
    scales[:] = scales[inds]
    ids[:]    = ids[inds]
    masses[:] = masses[inds]

    return scales, ids, masses, combinedFilename

# loadRawMergers()



def loadMappedMergers(run, verbose=VERBOSE, loadsave=True ):
    """
    Load or create Mapped Mergers Dictionary as needed.
    """

    if( verbose ): print " - - BHMergers.loadMappedMergers()"

    mappedFilename = BHConstants.GET_MERGERS_RAW_MAPPED_FILENAME(run, VERSION_MAP)

    ### Try to Load Existing Mapped Mergers ###
    if( loadsave ):
        if( verbose ): print " - - - Loading saved data from '%s'" % (mappedFilename)

        # If file exists, load data
        if( os.path.exists(mappedFilename) ):
            mergersMapped = aux.npzToDict(mappedFilename)
            if( verbose ): print " - - - Dictionary Loaded"
            loadVers = mergersMapped[MERGERS_VERSION]
            # Make sure version matches, otherwise re-create mappings
            if( loadVers != VERSION_MAP ):
                loadTime = mergersMapped[MERGERS_CREATED]
                print "BHMergers.loadMappedMergers() : loaded version %f, from %s" % (loadVers, loadTime)
                print "BHMergers.loadMappedMergers() : VERSION %f, recreating new version!" % (VERSION_MAP)
                loadsave = False

        else:
            print " - - - Mapped file '%s' does not exist" % (mappedFilename)
            loadsave = False



    ### Recreate Mappings ###
    if( not loadsave ):

        # Load Raw Mergers
        scales, ids, masses, filename = loadRawMergers(run, verbose=verbose)

        ### Create Mapping Between Mergers and Snapshots ###
        mapM2S, mapS2M, ontop = _mapToSnapshots(scales)

        # Store in dictionary
        mergersMapped = { MERGERS_FILE      : mappedFilename,
                          MERGERS_RUN       : run,
                          MERGERS_NUM       : len(scales),
                          MERGERS_CREATED   : datetime.now().ctime(),
                          MERGERS_VERSION   : VERSION_MAP,

                          MERGERS_SCALES    : scales,
                          MERGERS_IDS       : ids,
                          MERGERS_MASSES    : masses,

                          MERGERS_MAP_MTOS  : mapM2S,
                          MERGERS_MAP_STOM  : mapS2M,
                          MERGERS_MAP_ONTOP : ontop,
                          }

        aux.dictToNPZ(mergersMapped, mappedFilename, verbose)


    return mergersMapped

# loadMappedMergers()



def loadFixedMergers(run, verbose=VERBOSE, loadsave=True ):
    """
    Load BH Merger data with duplicats removes, and masses corrected.

    Arguments
    ---------
       run      <int>  : illustris simulation run number {1,3}
       verbose  <bool> : optional, print verbose output
       loadsave <bool> : optional, load existing save file (recreate if `False`)

    Returns
    -------
       mergersFixed <dict> : dictionary of 'fixed' mergers, most entries shaped [N,2] for `N`
                             mergers, and an entry for each {``BH_IN``, ``BH_OUT``}

    """
    
    if( verbose ): print " - - BHMergers.loadFixedMergers()"

    fixedFilename = BHConstants.GET_MERGERS_FIXED_FILENAME(run, VERSION_FIX)

    ## Try to Load Existing Mapped Mergers
    if( loadsave ):
        if( verbose ): print " - - - Loading from save '%s'" % (fixedFilename)
        if( os.path.exists(fixedFilename) ):
            mergersFixed = aux.npzToDict(fixedFilename)
            if( verbose ): print " - - - Loaded from '%s'" % (fixedFilename)
            loadVers = mergersFixed[MERGERS_VERSION]
            # Make sure version matches, otherwise re-create mappings
            if( loadVers != VERSION_FIX ):
                loadTime = mergersFixed[MERGERS_CREATED]
                print "BHMergers.loadFixedMergers() : loaded version %f, from %s" % (loadVers, loadTime)
                print "BHMergers.loadFixedMergers() : VERSION %f, recreating!" % (VERSION_FIX)
                loadsave = False
                
        else:
            print " - - - '%s' does not exist.  Recreating." % (fixedFilename)
            loadsave = False


    ## Recreate Fixed Mergers
    if( not loadsave ):
        # Load Mapped Mergers
        mergersMapped = loadMappedMergers(run, verbose=verbose)
        # Fix Mergers
        mergersFixed = _fixMergers(run, mergersMapped, verbose=verbose)
        # Save
        aux.dictToNPZ(mergersFixed, fixedFilename, verbose)


    return mergersFixed

# loadFixedMergers()



def _fixMergers(run, mergers, verbose=VERBOSE):
    """
    Filter and 'fix' input merger catalog.

    This includes:
     - Remove duplicate entries (Note-1)
     - Load 'fixed' out-BH masses from ``BHMatcher`` (which uses ``BHDetails`` entries)

    Arguments
    ---------
       run     <int>  : illustris simulation number {1,3}
       mergers <dict> : input dictionary of unfiltered merger events
       verbose <bool> : optional, print verbose output

    Returns
    -------
       fixedMergers <dict> : filtered merger dictionary

    Notes
    -----
       1 : There are 'duplicate' entries which have different occurence times (scale-factors)
           suggesting that there is a problem with the actual merger, not just the logging.
           This is not confirmed.  Currently, whether the times match or not, the *later*
           merger entry is the only one that is preserved in ``fixedMergers``

    """


    if( verbose ): print " - - BHMergers._fixMergers()"

    # Make copy to modify
    fixedMergers = dict(mergers)


    ##  Remove Repeated Entries
    #   =======================
    #      Remove entries where IDs match a second time (IS THIS ENOUGH?!)

    ids    = fixedMergers[MERGERS_IDS]
    scales = fixedMergers[MERGERS_SCALES]

    # First sort by ``BH_IN`` then ``BH_OUT`` (reverse of given order)
    sort = np.lexsort( (ids[:,BH_OUT], ids[:,BH_IN]) )

    badInds = []
    numMismatch = 0

    if( verbose ): print " - - - Examining %d merger entries" % (len(sort))

    ## Iterate over all entries

    for ii in xrange(len(sort)-1):
        
        this = ids[sort[ii]]
        jj = ii+1

        # Look through all examples of same BH_IN
        while( ids[sort[jj],BH_IN] == this[BH_IN] ):
            # If BH_OUT also matches, this is a duplicate -- store first entry as bad |NOTE-1|
            if( ids[sort[jj],BH_OUT] == this[BH_OUT] ):

                # Double check that time also matches
                if( scales[sort[ii]] != scales[sort[jj]] ): numMismatch += 1
                badInds.append(sort[ii])
                break

            jj += 1

        # } while

    # } ii

            
    if( verbose ): print " - - - Total number of duplicates = %d" % (len(badInds))
    if( verbose ): print " - - - Number with mismatched times = %d" % (numMismatch)

    ## Remove Duplicate Entries
    for key in MERGERS_PHYSICAL_KEYS:
        fixedMergers[key] = np.delete(fixedMergers[key], badInds, axis=0)


    # Recalculate maps
    mapM2S, mapS2M, ontop = _mapToSnapshots(fixedMergers[MERGERS_SCALES])
    fixedMergers[MERGERS_MAP_MTOS ] = mapM2S
    fixedMergers[MERGERS_MAP_STOM ] = mapS2M
    fixedMergers[MERGERS_MAP_ONTOP] = ontop

    # Change number, creation date, and version
    oldNum = len(mergers[MERGERS_SCALES])
    newNum = len(fixedMergers[MERGERS_SCALES])
    fixedMergers[MERGERS_NUM] = newNum
    fixedMergers[MERGERS_CREATED] = datetime.now().ctime()
    fixedMergers[MERGERS_VERSION] = VERSION_FIX

    if( verbose ): print " - - - Number of Mergers %d ==> %d" % (oldNum, newNum)


    ### Fix Merger 'Out' Masses
    #   =======================
    if( verbose ): print " - - - Loading reconstructed 'out' BH masses"
    masses = fixedMergers[MERGERS_MASSES]
    aveBef = np.average( masses[:,BH_OUT] )
    massOut = BHMatcher.inferMergerOutMasses(run, mergers=fixedMergers, verbose=verbose)
    masses[:,BH_OUT] = massOut
    aveAft = np.average( masses[:,BH_OUT] )
    if( verbose ): print " - - - - Ave mass:  %.4e ===> %.4e" % (aveBef, aveAft)

    return fixedMergers

# _fixMergers()


def _importRawMergers(files, verbose=VERBOSE):
    """
    Fill the given arrays with merger data from the given target files.

    Arrays ``ids`` and ``masses`` are shaped [N, 2], for ``N`` total mergers.
    By convention the
        'in'  (accreted) BH is index ``BH_IN`` (0?)
        'out' (accretor) BH is index ``BH_OUT`` (1?)


    Arguments
    ---------

    """

    if( verbose ): print " - - BHMergers._importRawMergers()"

    # Make sure argument is a list
    if( not aux.iterableNotString(files) ): files = [ files ]

    ### Count Mergers and Prepare Storage for Data ###
    numLines = aux.countLines(files)
    if( verbose ): print " - - - Lines : %d" % (numLines)

    # Initialize Storage
    scales = np.zeros( numLines,               dtype=DBL)
    ids    = np.zeros([numLines,NUM_BH_TYPES], dtype=TYPE_ID)
    masses = np.zeros([numLines,NUM_BH_TYPES], dtype=DBL)

    nums = numLines-1
    if( nums <= 100 ): interval = nums
    else:              interval = np.int( np.floor(nums/100.0) )

    count = 0
    for fil in files:
        for line in open(fil):
            # Get target elements from each line of file
            time, out_id, out_mass, in_id, in_mass = _parseMergerLine(line)
            # Store values
            scales[count] = time
            ids[count,BH_IN ] = in_id
            ids[count,BH_OUT] = out_id
            masses[count,BH_IN ] = in_mass
            masses[count,BH_OUT] = out_mass
            # Increment Counter
            count += 1

            # Print Progress
            if( verbose ):
                if( count % interval == 0 or count == nums):
                    sys.stdout.write('\r - - - %.2f%% Complete' % (100.0*count/nums))

                if( count == nums ): sys.stdout.write('\n')
                sys.stdout.flush()


    return scales, ids, masses

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
    time     = DBL(strs[1])
    out_id   = TYPE_ID(strs[2])
    out_mass = DBL(strs[3])
    in_id    = TYPE_ID(strs[4])
    in_mass  = DBL(strs[5])

    return time, out_id, out_mass, in_id, in_mass

# _parseMergerLine()


def _mapToSnapshots(scales, verbose=VERBOSE):
    """
    Find the snapshot during which, or following each merger

    """

    if( verbose ): print " - - BHMergers._mapToSnapshots()"

    numMergers = len(scales)

    # Load Cosmology
    cosmo      = Cosmology()
    snapScales = cosmo.snapshotTimes()

    # Map Mergers-2-Snapshots: snapshot before (or ontop) of each merger
    mapM2S = np.zeros(numMergers, dtype=INT)
    # Map Snapshots-2-Mergers: list of mergers just-after (or ontop) of each snapshot
    mapS2M = [ [] for ii in range(cosmo.num) ]
    # Flags if merger happens exactly on a snapshot (init to False=0)
    ontop  = np.zeros(numMergers, dtype=bool)

    ### Find snapshots on each side of merger time ###

    # Find the snapshot just below and above each merger.
    #     each entry (returned) is [ low, high, dist-low, dist-high ]
    #     low==high if the times match (within function's default uncertainty)
    snapBins = [ aux.findBins(sc, snapScales) for sc in scales ]

    nums = numMergers-1
    if( nums < 100 ): interval = nums
    else:             interval = np.int( np.floor(nums/100.0) )

    ### Create Mappings ###

    if( verbose ): print " - - - Creating mappings"
    for ii, bins in enumerate(snapBins):
        tsnap = bins[1]                                                                             # Set snapshot to upper bin
        mapM2S[ii] = tsnap                                                                          # Set snapshot for this merger
        mapS2M[tsnap].append(ii)                                                                    # Add merger to this snapshot
        # If this merger takes place ontop of snapshot, set flag
        if( bins[0] == bins[1] ): ontop[ii] = True

        # Print Progress
        if( verbose ):
            if( ii % interval == 0 or ii == nums):
                sys.stdout.write('\r - - - %.2f%% Complete' % (100.0*ii/nums))

            if( ii == nums ): sys.stdout.write('\n')
            sys.stdout.flush()



    # Find the most mergers in a snapshot
    numPerSnap = np.array([len(s2m) for s2m in mapS2M ])
    mostMergers = np.max( numPerSnap )
    mostIndex = np.where( mostMergers == numPerSnap )[0]
    # Find the number of ontop mergers
    numOntop = np.count_nonzero(ontop)
    if( verbose ): print " - - - Snapshot %d with the most (%d) mergers" % (mostIndex, mostMergers)
    if( verbose ): print " - - - %d (%.2f) ontop mergers" % (numOntop, 1.0*numOntop/nums)

    return mapM2S, mapS2M, ontop

# _mapToSnapshots()



if __name__ == "__main__": main()
