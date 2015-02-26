"""
Module to handle Illustris blackhole details files.

Details are accessed via 'intermediate' files which are reorganized versions of the 'raw' illustris
files 'blackhole_details_<#>.txt'.  The `main()` function assures that details entries are properly
converted from raw to processed form, organized by time of entry instead of processor.  Those
details can then be accessed by snapshot and blackhole ID number.

Functions
---------
main() : returns None, assures that intermediate files exist -- creating them if necessary.
detailsForBH() : returns dict, dict; retrieves the details entries for a given BH ID number.


Notes
-----
  - The BH Details files from illustris, 'blackhole_details_<#>.txt' are organized by the processor
    on which each BH existed in the simulation.  The method `_reorganizeBHDetails()` sorts each
    detail entry instead by the time (scalefactor) of the entry --- organizing them into files
    grouped by which snapshot interval the detail entry corresponds to.  The reorganization is
    first done into 'temporary' ASCII files before being converted into numpy `npz` files by the
    method `_convertDetailsASCIItoNPZ()`.  The `npz` files are effectively dictionaries storing
    the select details parameters (i.e. mass, BH ID, mdot, rho, cs), along with some meta data
    about the `run` number, and creation time, etc.  Execution of the BHDetails ``main`` routine
    checks to see if the npz files exist, and if they do not, they are created.

  - There are also routines to obtain the details entries for a specific BH ID.  In particular,
    the method `detailsForBH()` will return the details entry/entries for a target BH ID and
    run/snapshot.

  - Illustris Blackhole Details Files 'blackhole_details_<#>.txt'
    - Each entry is given as
      0   1            2     3     4    5
      ID  scalefactor  mass  mdot  rho  cs


"""

### Builtin Modules ###
import os, sys
from glob import glob

import numpy as np

from BHConstants import DATA_PATH, _DOUBLE, _LONG
from BHMergers import MERGERS_IDS, MERGERS_MASSES, MERGERS_TIMES, MERGERS_NUM, MERGERS_MAP_STOM, \
                      MERGERS_MAP_MTOS, MERGERS_MAP_ONTOP, IN_BH, OUT_BH

from .. import AuxFuncs as aux
from .. import Constants as const
from .. import illcosmo

from datetime import datetime


### Default Runtime Parameters ###
RUN = 3                                                                                             # Default Illustris run to load {1,3}
VERBOSE = True                                                                                      # Print verbose output during execution

REDO_TEMP = False                                                                                   # Re-create temporary files
REDO_SAVE = False                                                                                   # Re-create NPZ save files
CLEAN_TEMP = False                                                                                  # Delete temporary files after NPZ are created

# Where to find the 'raw' Illustris Merger files
_DETAILS_FILE_DIRS = { 3:'/n/ghernquist/Illustris/Runs/L75n455FP/output/blackhole_details/' }
_DETAILS_FILE_NAMES = "blackhole_details_*.txt"                                                     # Glob regex to match details files

_PRINT_INTERVAL = 2e4                                                                               # Interval at which to print status


# Where to save intermediate files
_DETAILS_DIR = "ill-%d_bh-details/"
_DET_TEMP_NAME = "ill-%d_details_snap-%d_temp.txt"
_DET_SAVE_NAME = "ill-%d_details_snap-%d.npz"

details_temp_filename = lambda x,y: DATA_PATH + (_DETAILS_DIR % (x)) + (_DET_TEMP_NAME % (x,y))
details_save_filename = lambda x,y: DATA_PATH + (_DETAILS_DIR % (x)) + (_DET_SAVE_NAME % (x,y))


### Dictionary Keys for Details Parameters ###
DETAIL_IDS     = 'id'
DETAIL_TIMES   = 'times'
DETAIL_MASSES  = 'masses'
DETAIL_MDOTS   = 'mdots'
DETAIL_RHOS    = 'rhos'
DETAIL_CS      = 'cs'
DETAIL_RUN     = 'run'
DETAIL_SNAP    = 'snap'
DETAIL_NUM     = 'num'
DETAIL_CREATED = 'created'

DETAIL_BEFORE  = 0                                                                                  # Before merger time (MUST = 0!)
DETAIL_AFTER   = 1                                                                                  # After (or equal) merger time (MUST = 1!)

_DETAIL_PHYSICAL_KEYS = [ DETAIL_IDS,   DETAIL_TIMES, DETAIL_MASSES, 
                          DETAIL_MDOTS, DETAIL_RHOS,  DETAIL_CS     ]





###  ===================================  ###
###  =============  MAIN  ==============  ###
###  ===================================  ###



def main(run=RUN, verbose=VERBOSE, redo_temp=REDO_TEMP, redo_save=REDO_SAVE):

    print "HELLO!"
    raise RuntimeError("GOODBYE!")

    if( verbose ): print "\nBHDetails.py\n"

    ### Load Variables and Parameters ###

    # Load cosmology
    cosmo = illcosmo.Cosmology()
    snapTimes = cosmo.snapshotTimes()                                                               # Scale-factor of each snapshot

    ### Organize Details by Snapshot Time; create new, temporary ASCII Files ###

    # See if all temp files already exist
    tempFiles = [ details_temp_filename(run, snap) for snap in xrange(cosmo.num) ]
    tempExist = aux.filesExist(tempFiles)

    # If temp files dont exist, or we WANT to redo them, then create temp files
    if( not tempExist or redo_temp ):

        # Get Illustris BH Details Filenames
        if( verbose ): print " - Finding Illustris BH Details files"
        detailsFiles = _getIllustrisDetailsFilenames(run, verbose)
        if( len(detailsFiles) < 1 ): raise RuntimeError("Error no details files found!!")

        # Reorganize into temp files
        if( verbose ): print " - Reorganizing details into temporary files"
        _reorganizeBHDetails(detailsFiles, snapTimes, run, verbose=verbose)

    else:
        if( verbose ):
            print " - All temporary files already exist."
            note = ("   NOTE: if you would like to RE-create the temprary files, using the raw\n"
                    "         Illustris `blackhole_details_#` files, then rerun BHDetails.py\n"
                    "         with the ``REDO_TEMP`` flag set to ``True``.")
            print note


    # Confirm all temp files exist
    tempExist = aux.filesExist(tempFiles)

    # If files are missing, raise error
    if( not tempExist ):
        print "Temporary Files missing!  First file = '%s'" % (tempFiles[0])
        raise RuntimeError("Temporary Files missing!")


    # See if all npz files already exist
    saveFiles = [ details_save_filename(run, snap) for snap in xrange(cosmo.num) ]
    saveExist = aux.filesExist(saveFiles)

    ### Convert temp ASCII Files, to new Details object files ###

    # If NPZ files don't exist, or we WANT to redo them, create NPZ files
    if( not saveExist or redo_save ):

        if( verbose ): print " - Converting temporary files to NPZ"
        _convertDetailsASCIItoNPZ(cosmo.num, run, verbose=verbose)

    else:
        if( verbose ):
            print " - All NPZ save files already exist."
            note = ("   NOTE: if you would like to RE-create the npz save files, using the\n"
                    "         existing temp files, then rerun BHDetails.py with the\n"
                    "         ``REDO_SAVE`` flag set to ``True``.\n"
                    "         To also re-create the temp files, use ``REDO_TEMP``.\n")
            print note


    # Confirm save files exist
    saveExist = aux.filesExist(saveFiles)

    # If files are missing, raise error
    if( not saveExist ):
        print "Save (NPZ) Files missing!  First file = '%s'" % (saveFiles[0])
        raise RuntimeError("Save (NPZ) Files missing!")

    return

# main()






###  =====================================================  ###
###  ===========  PREPARE INTERMEDIATE FILES  ============  ###
###  =====================================================  ###



def _reorganizeBHDetails(detFiles, times, run, verbose=VERBOSE):

    numOldFiles = len(detFiles)

    if( verbose ):
        print " - - Counting lines in Illustris Details files"
        numLines = aux.estimateLines(detFiles)
        print " - - - Estimate %d (%.1e) lines in %d files" % (numLines, numLines, numOldFiles)


    ### Open new ASCII, Temp details files ###
    if( verbose ): print " - - Opening new files"
    tempFiles = [ details_temp_filename(run,snap) for snap in xrange(len(times)) ]
    # Make sure path is okay
    aux.checkPath(tempFiles[0])
    tempFiles = [ open(tfil, 'w') for tfil in tempFiles ]

    ### Iterate over all Illustris Details Files ###
    if( verbose ): print " - - Sorting details into times of snapshots"
    start = datetime.datetime.now()
    count = 0
    for ii,oldname in enumerate(detFiles):

        # Iterate over each entry in file
        for dline in open(oldname):
            detTime = _DOUBLE( dline.split()[1] )
            # Find the time bin given left edges (hence '-1'); include right-edges ('right=True')
            snapNum = np.digitize([detTime], times, right=True) - 1

            # Write line to apropriate file
            tempFiles[snapNum].write( dline )

            count += 1

            # Print Progress
            if( verbose ):

                if( count % _PRINT_INTERVAL == 0 or count == numLines):
                    # Find out current duration
                    now = datetime.datetime.now()
                    dur = now-start                                                                 # 'timedelta' object

                    # Print status and time to completion
                    statStr = aux.statusString(count, numLines, dur)
                    sys.stdout.write('\r - - - %s' % (statStr))
                    sys.stdout.flush()

            # } verbose
        # } dline
    # } ii

    if( verbose ): sys.stdout.write('\n')

    # Close out details files.
    fileSizes = 0.0
    for ii, newdf in enumerate(tempFiles):
        newdf.close()
        fileSizes += os.path.getsize(newdf.name)

    aveSize = fileSizes/(1.0*len(tempFiles))

    sizeStr = aux.bytesString(fileSizes)
    aveSizeStr = aux.bytesString(aveSize)
    if( verbose ): print " - - Total temp file size = '%s', average = '%s'" % (sizeStr, aveSizeStr)

    return



def _convertDetailsASCIItoNPZ(numSnaps, run, verbose=VERBOSE):

    start = datetime.datetime.now()
    filesSize = 0.0
    sav = None

    for snap in xrange(numSnaps):

        tmp = details_temp_filename(run, snap)
        sav = details_save_filename(run, snap)

        # Load details from ASCII File
        ids, times, masses, mdots, rhos, cs = _loadBHDetails_ASCII(tmp)

        # Store details in dictionary
        details = { DETAIL_NUM : len(ids),
                    DETAIL_RUN : run,
                    DETAIL_SNAP : snap,
                    DETAIL_CREATED : datetime.datetime.now().ctime(),
                    DETAIL_IDS    : ids,
                    DETAIL_TIMES  : times,
                    DETAIL_MASSES : masses,
                    DETAIL_MDOTS  : mdots,
                    DETAIL_RHOS   : rhos,
                    DETAIL_CS     : cs }

        np.savez(sav, **details)
        if( not os.path.exists(sav) ):
            raise RuntimeError("Could not save to file '%s'!!" % (sav) )


        # Find and report progress
        if( verbose ):
            filesSize += os.path.getsize(sav)

            now = datetime.datetime.now()
            dur = now-start

            statStr = aux.statusString(snap+1, numSnaps, dur)
            sys.stdout.write('\r - - - %s' % (statStr))
            sys.stdout.flush()
            if( snap+1 == numSnaps ): sys.stdout.write('\n')

    # } snap

    if( verbose ):
        aveFileSize = filesSize / numSnaps
        totSize = aux.bytesString(filesSize)
        aveSize = aux.bytesString(aveFileSize)
        print " - - - Saved Details NPZ files.  Total size = %s, Ave Size = %s" % (totSize, aveSize)
        print " - - - - Last : '%s'" % (sav)

    if( CLEAN_TEMP ):
        if( verbose ): print " - - - Removing temporary files"
        for snap in range(numSnaps):
            os.remove( details_temp_filename(run, snap) )
    else:
        if( verbose ): print " - - - Temporary files are NOT being removed."


    return



def _loadBHDetails_ASCII(asciiFile, verbose=VERBOSE):

    ### Files have some blank lines in them... Clean ###
    lines = open(asciiFile).readlines()                                                             # Read all lines at once
    nums = len(lines)

    # Allocate storage
    ids    = np.zeros(nums, dtype=_LONG)
    times  = np.zeros(nums, dtype=_DOUBLE)
    masses = np.zeros(nums, dtype=_DOUBLE)
    mdots  = np.zeros(nums, dtype=_DOUBLE)
    rhos   = np.zeros(nums, dtype=_DOUBLE)
    cs     = np.zeros(nums, dtype=_DOUBLE)

    count = 0
    # Iterate over lines, storing only those with content (should be all)
    for lin in lines:
        lin = lin.strip()
        if( len(lin) > 0 ):
            tid,tim,mas,dot,rho,tcs = _parseIllustrisBHDetailsLine(lin)
            ids[count] = tid
            times[count] = tim
            masses[count] = mas
            mdots[count] = dot
            rhos[count] = rho
            cs[count] = tcs
            count += 1

    # Trim excess (shouldn't be needed)
    if( count != nums ):
        trim = np.s_[count:]
        ids    = np.delete(ids, trim)
        times  = np.delete(times, trim)
        masses = np.delete(masses, trim)
        mdots  = np.delete(mdots, trim)
        rhos   = np.delete(rhos, trim)
        cs     = np.delete(cs, trim)


    return ids, times, masses, mdots, rhos, cs



def _parseIllustrisBHDetailsLine(instr):
    """
    Parse a line from an Illustris blachole_details_#.txt file

    The line is formatted (in C) as:
        "BH=%llu %g %g %g %g %g\n",
        (long long) P[n].ID, All.Time, BPP(n).BH_Mass, mdot, rho, soundspeed

    return ID, time, mass, mdot, rho, cs
    """
    args = instr.split()

    # First element is 'BH=########', trim to just the id number
    args[0] = args[0].split("BH=")[-1]

    return _LONG(args[0]), _DOUBLE(args[1]), _DOUBLE(args[2]), _DOUBLE(args[3]), _DOUBLE(args[4]), _DOUBLE(args[5])




def _getIllustrisDetailsFilenames(run=RUN, verbose=VERBOSE):
    detailsDir = _DETAILS_FILE_DIRS[run]
    if( verbose ): print " - - Searching for run '%d' in '%s'" % (run, detailsDir)
    detailsFilenames = detailsDir + _DETAILS_FILE_NAMES
    detailsFiles = sorted(glob(detailsFilenames))
    if( verbose ): print " - - - Found %d files" % (len(detailsFiles))
    return detailsFiles











###  ==============================================================  ###
###  =============  BH / MERGER - DETAILS MATCHING  ===============  ###
###  ==============================================================  ###



def loadBHDetails_NPZ(run, snap):
    detsName = details_save_filename(run,snap)
    dat = np.load(detsName)
    return dat


def detailsForBH(bhid, run, snap, details=None, side=None, verbose=VERBOSE):
    """
    Retrieve the details entry for a particular BH at a target snapshot.

    Parameters
    ----------
    bhid : int
        ID of the target BH
    run : int
        Number of this Illustris run {1,3}
    snap : int
        Number of snapshot in which top find details
    details : dictionary (npz file)
        BH Details in a given snapshot (optional, default: None)
        If `None` these are reloaded
    side : {'left', 'right', None}, (optional, default: None)
        Which matching elements to return.
        None : return all matches
        'left' : return the earliest match
        'right' : return the latest match


    Returns
    -------
    scale, mass, mdot, rho, cs : 5 scalars or scalar arrays with the matching
                                 entries, determined by the `side` argument
        scale : the scale factor of the entry
        mass : the mass of the BH in each entry
        mdot : the accretion rate of the BH in each entry
        rho : the ambient density
        cs : the ambient sound speed

        All elements are set to `None` if no matches are found.

    Raises
    ------
    RuntimeError
        when the `side` argument is invalid

    """

    if( verbose ): print " - - detailsForBH()"

    # Details keys which should be returned
    returnKeys = [ DETAIL_TIMES, DETAIL_MASSES, DETAIL_MDOTS, DETAIL_RHOS, DETAIL_CS ]
    # If no match is found, return all values as this:
    missingValue = -1.0

    ### Make sure Details are Loaded and Appropriate ###

    # If details not provided for this snapshot, load them
    if( details == None ):
        if( verbose ): print " - - - No details provided, loading for snapshot %d" % (snap)
        details = loadBHDetails_NPZ(run, snap)
        if( verbose ): print " - - - - Loaded %d details" % (details[DETAIL_NUM])

    # Make sure these details match target snapshot and run
    assert details[DETAIL_RUN] == run, \
        "Details run %d does not match target %d!" % (details[DETAIL_RUN], run)

    assert details[DETAIL_SNAP] == snap, \
        "Details snap %d does not match target %d!" % (details[DETAIL_SNAP], snap)

    ### Find the Details index which matched ID and Nearest in Time ###

    # Find details indices with BH ID match
    inds = np.where( bhid == details[DETAIL_IDS] )[0]

    # If there are no matches, return None array
    if( len(inds) == 0 ):
        if( verbose ): print "No matches in snap %d for ID %d" % (snap, bhid)
        return { key : missingValue for key in returnKeys }

    # Get times for matching details
    detTimes = details[DETAIL_TIMES][inds]
    # Get indices to sort times
    sortInds = np.argsort(detTimes)


    ### Determine Which Matching Entries to Return ###

    # Return all matching entries
    if( side == None ):
        retInds = sortInds
    # Return entry for earliest matching time
    elif( side == 'left' ):
        retInds = sortInds[0]
    # Return entry for latest matching time
    elif( side == 'right' ):
        retInds = sortInds[-1]
    # Error
    else:
        raise RuntimeError("Unrecognized side='%s'!" % (side) )


    ### Return Matching Details ###

    # Convert indices to global array
    retInds = inds[retInds]
    # Create output dictionary with same keys as `details`
    bhDets = { key : details[key][retInds] for key in returnKeys }

    return details, bhDets




def detailsForMergers(mergers, run, verbose=VERBOSE):
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


    if( verbose ): print " - - detailsForMergers()"

    # Import MatchDetails cython file
    import pyximport
    pyximport.install(setup_args={"include_dirs":np.get_include()})
    import MatchDetails

    numMergers = mergers[MERGERS_NUM]
    numSnaps = len(mergers[MERGERS_MAP_STOM])

    ### Initialize Output Arrays ###
    # array[merger, in/out, bef/aft]
    detID   = -1*np.ones([numMergers, 2, 2], dtype=_LONG)
    detTime = -1*np.ones([numMergers, 2, 2], dtype=_DOUBLE) 
    detMass = -1*np.ones([numMergers, 2, 2], dtype=_DOUBLE)
    detMDot = -1*np.ones([numMergers, 2, 2], dtype=_DOUBLE)
    detRho  = -1*np.ones([numMergers, 2, 2], dtype=_DOUBLE)
    detCS   = -1*np.ones([numMergers, 2, 2], dtype=_DOUBLE)

    # Create dictionary of details for mergers
    mergDets = { DETAIL_IDS    : detID,
                 DETAIL_TIMES  : detTime,
                 DETAIL_MASSES : detMass,
                 DETAIL_MDOTS  : detMDot,
                 DETAIL_RHOS   : detRho,
                 DETAIL_CS     : detCS }


    # Iterate over each snapshot, with list of mergers in each `s2m`
    count = 0
    start = datetime.now()
    for snap,s2m in enumerate(mergers[MERGERS_MAP_STOM]):

        # If there are no mergers in this snapshot, continue to next iteration
        if( len(s2m) <= 0 ): continue

        # Convert from list to array
        search = np.array(s2m)


        ### Form List(s) of Target BH IDs ###

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


        ### Prepare Detail and Merger Information for Matching ###
        searchNum = len(search)

        # Get all details (IDs and times) for this snapshot
        dets     = loadBHDetails_NPZ(run, snap)
        detIDs   = dets[DETAIL_IDS]
        detTimes = dets[DETAIL_TIMES]

        # If there are no details in this snapshot (should only happen at end), continue
        if( len(detIDs) <= 0 ): continue

        # Get the BH merger info for this snapshot
        bhids    = mergers[MERGERS_IDS][search]
        bhmasses = mergers[MERGERS_MASSES][search]
        bhtimes  = mergers[MERGERS_TIMES][search]
        # Duplicate `times` to match shape of `ids` and `masses`
        bhtimes = np.array([bhtimes, bhtimes]).T

        # Reshape 2D to 1D arrays for matching
        searchShape = np.shape(bhids)                                                               # Store initial shape [searchNum,2]
        bhids = bhids.reshape(2*searchNum)
        bhmasses = bhmasses.reshape(2*searchNum)
        bhtimes = bhtimes.reshape(2*searchNum)

        ### Match Details to Mergers and Store ###

        # Find Details indices to match these BHs (just before and just after merger)
        indsBef, indsAft = MatchDetails.detailsForBlackholes(bhids, bhtimes, detIDs, detTimes)

        # Reshape indices to match mergers (in and out BHs)
        indsBef = np.array(indsBef).reshape(searchShape)
        indsAft = np.array(indsAft).reshape(searchShape)

        # Reshape det indices to match ``mergDets``
        # First makesure numbering is consistent!
        assert DETAIL_BEFORE == 0 and DETAIL_AFTER == 1, "`BEFORE` and `AFTER` broken!!"

        detInds = np.dstack([indsBef,indsAft])


        ### Store matches ###

        # Both 'before' and 'after' matches
        for BEF_AFT in [DETAIL_BEFORE, DETAIL_AFTER]:
            # Both 'in' and 'out' BHs
            for IN_OUT in [IN_BH, OUT_BH]:
                # Select only successful matches
                inds = np.where( detInds[:,IN_OUT,BEF_AFT] >= 0 )
                useInds = np.squeeze(detInds[inds,IN_OUT,BEF_AFT])

                # All target parameters
                for KEY in _DETAIL_PHYSICAL_KEYS:
                    mergDets[KEY][search[inds],IN_OUT,BEF_AFT] = dets[KEY][useInds]


        # Print progress
        count += len(bhids)
        if( verbose ):
            now = datetime.now()        
            statStr = aux.statusString(count, 2*numMergers, now-start)
            sys.stdout.write('\r - - - %s' % (statStr))
            sys.stdout.flush()

    # } snap

    return mergDets






if __name__ == "__main__": main()
