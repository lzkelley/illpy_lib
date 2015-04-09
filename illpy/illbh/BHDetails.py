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

from BHConstants import *

from .. import AuxFuncs as aux

from datetime import datetime


VERSION = 0.22                                                                                      # Version of BHDetails

_DEF_PRECISION = -8                                                                                 # Default precision



###  ===================================  ###
###  =============  MAIN  ==============  ###
###  ===================================  ###



def processDetails(run, loadsave=True, verbose=VERBOSE):

    if( verbose ): print " - - BHDetails.processDetails()"

    # Organize Details by Snapshot Time; create new, temporary ASCII Files
    tempFiles = organizeDetails(run, loadsave=loadsave, verbose=verbose)
    
    # Create Dictionary Details Files
    saveFiles = formatDetails(run, loadsave=loadsave, verbose=verbose)

    return

# processDetails()




def organizeDetails(run, loadsave=True, verbose=VERBOSE):

    if( verbose ): print " - - BHDetails.organizeDetails()"

    tempFiles = [ GET_DETAILS_TEMP_FILENAME(run, snap) for snap in xrange(NUM_SNAPS) ]

    # Check if all temp files already exist
    if( loadsave ):
        tempExist = aux.filesExist(tempFiles)
        if( not tempExist ):
            if( verbose ): print " - - - Temp files do not exist '%s'" % (tempFiles[0])
            loadsave = False


    # If temp files dont exist, or we WANT to redo them, then create temp files
    if( not loadsave ):

        # Get Illustris BH Details Filenames
        if( verbose ): print " - - - Finding Illustris BH Details files"
        rawFiles = GET_ILLUSTRIS_BH_DETAILS_FILENAMES(run, verbose)
        if( len(rawFiles) < 1 ): raise RuntimeError("Error no details files found!!")

        # Reorganize into temp files
        if( verbose ): print " - - - Reorganizing details into temporary files"
        _reorganizeBHDetailsFiles(run, rawFiles, tempFiles, verbose=verbose)


    # Confirm all temp files exist
    tempExist = aux.filesExist(tempFiles)

    # If files are missing, raise error
    if( tempExist ):
        if( verbose ): print " - - - Temp files exist"
    else:
        print "Temporary Files still missing!  '%s'" % (tempFiles[0])
        raise RuntimeError("Temporary Files missing!")


    return tempFiles

# organizeDetails()



def formatDetails(run, loadsave=True, verbose=VERBOSE):
    
    if( verbose ): print " - - BHDetails.formatDetails()"

    # See if all npz files already exist
    saveFilenames = [ GET_DETAILS_SAVE_FILENAME(run, snap, VERSION) for snap in xrange(NUM_SNAPS) ]

    # Check if all save files already exist, and correct versions
    if( loadsave ):
        saveExist = aux.filesExist(saveFilenames)
        if( saveExist ):
            dets = loadBHDetails(run, 0)
            loadVers = dets[DETAILS_VERSION]
            if( loadVers != VERSION ):
                print "BHDetails.formatDetails() : loaded version %s from '%s'" % (str(loadVers), dets[DETAILS_FILE])
                print "BHDetails.formatDetails() : current version %s" % (str(VERSION))
                print "BHDetails.formatDetails() : re-converting Details files !!!"
                loadsave = False

        else:
            print "BHDetails.formatDetails() : Save files do not exist e.g. '%s'" % (saveFilenames[0])
            print "BHDetails.formatDetails() : converting raw Details files !!!"
            loadsave = False


    if( not loadsave ):

        if( verbose ): print " - - - Converting temporary files to NPZ"
        _convertDetailsASCIItoNPZ(run, verbose=verbose)


    # Confirm save files exist
    saveExist = aux.filesExist(saveFilenames)

    # If files are missing, raise error
    if( saveExist ):
        if( verbose ): print " - - - Save files exist."
    if( not saveExist ):
        print "Save Files missing!  e.g. '%s'" % (saveFilenames[0])
        raise RuntimeError("Save Files missing!")


    return saveFilenames

# formatDetails()



def _reorganizeBHDetailsFiles(run, rawFilenames, tempFilenames, verbose=VERBOSE):

    if( verbose ): print " - - BHDetails._reorganizeBHDetailsFiles()"

    # Load cosmology
    from illpy import illcosmo
    cosmo = illcosmo.Cosmology()
    snapScales = cosmo.snapshotTimes()                                                              # Scale-factor of each snapshot

    numRaw = len(rawFilenames)


    ### Open new ASCII, Temp details files ###
    # Make sure path is okay
    aux.checkPath(tempFilenames[0])
    # Open each temp file
    tempFiles = [ open(tfil, 'w') for tfil in tempFilenames ]
    numTemp = len(tempFiles)


    if( verbose ): print " - - - Organizing %d raw files into %d temp files" % (numRaw, numTemp)

    ### Iterate over all Illustris Details Files ###
    if( verbose ): print " - - - Sorting details into times of snapshots"
    start = datetime.now()
    for ii,rawName in enumerate(rawFilenames):

        detLines = []                                                                               # Store each line of input details files
        detScales = []                                                                              # Store each scale factor of entries
        # Load all lines and entry scale-factors from raw details file
        for dline in open(rawName):
            detLines.append(dline)
            # Extract scale-factor from line
            detScale = DBL( dline.split()[1] )
            detScales.append(detScale)


        # Convert to array
        detLines  = np.array(detLines)
        detScales = np.array(detScales)

        # Get required precision in matching entry times (scales)
        try:
            prec = _getPrecision(detScales)
        # Set to a default value on error (not sure what's causing it)
        except ValueError, err:
            print "BHDetails._reorganizeBHDetailsFiles() : caught error '%s'" % (str(err))
            print "\tii = %d; file = '%s'" % (ii, rawName)
            print "\tlen(detScales) = ", len(detScales)
            prec = _DEF_PRECISION


        # Round snapshot scales to desired precision
        roundScales = np.around(snapScales, -prec)

        # Find snapshots following each entry (right-edge) or equal (include right: 'right=True')
        snapBins = np.digitize(detScales, roundScales, right=True)

        # For each Snapshot, write appropriate lines
        for jj in xrange(len(tempFiles)):

            inds = np.where( snapBins == jj )[0]
            if( len(inds) > 0 ):
                tempFiles[jj].writelines( detLines[inds] )

        # } jj



        # Print Progress
        if( verbose ):
            # Find out current duration
            now = datetime.now()
            dur = now-start

            # Print status and time to completion
            statStr = aux.statusString(ii+1, numRaw, dur)
            sys.stdout.write('\r - - - - %s' % (statStr))
            sys.stdout.flush()

        # } verbose

    # } ii

    if( verbose ): sys.stdout.write('\n')

    # Close out details files.
    fileSizes = 0.0
    for ii, newdf in enumerate(tempFiles):
        newdf.close()
        fileSizes += os.path.getsize(newdf.name)

    if( verbose ): 
        aveSize = fileSizes/(1.0*len(tempFiles))
        sizeStr = aux.bytesString(fileSizes)
        aveSizeStr = aux.bytesString(aveSize)
        print " - - - Total temp size = '%s', average = '%s'" % (sizeStr, aveSizeStr)


    inLines = aux.countLines(rawFilenames, progress=True)
    outLines = aux.countLines(tempFilenames, progress=True)
    if( verbose ): print " - - - Input lines = %d, Output lines = %d" % (inLines, outLines)
    if( inLines != outLines ): 
        print "in  file: ", rawFilenames[0]
        print "out file: ", tempFilenames[0]
        raise RuntimeError("WARNING: input lines = %d, output lines = %d!" % (inLines, outLines))


    return



def _convertDetailsASCIItoNPZ(run, verbose=VERBOSE):

    start = datetime.now()
    filesSize = 0.0
    sav = None

    ### Iterate over all Snapshots, convert from ASCII to NPZ ###

    # Go through snapshots in random order to make better estimate of duration
    allSnaps = np.arange(NUM_SNAPS)
    np.random.shuffle(allSnaps)

    for ii,snap in enumerate(allSnaps):

        tmp = GET_DETAILS_TEMP_FILENAME(run, snap)
        sav = GET_DETAILS_SAVE_FILENAME(run, snap, VERSION)

        # Load details from ASCII File
        ids, scales, masses, mdots, rhos, cs = _loadBHDetails_ASCII(tmp)

        # Store details in dictionary
        details = { DETAILS_NUM : len(ids),
                    DETAILS_RUN : run,
                    DETAILS_SNAP : snap,
                    DETAILS_CREATED : datetime.now().ctime(),
                    DETAILS_VERSION : VERSION,
                    DETAILS_FILE    : sav,
                    
                    DETAILS_IDS     : ids,
                    DETAILS_SCALES  : scales,
                    DETAILS_MASSES  : masses,
                    DETAILS_MDOTS   : mdots,
                    DETAILS_RHOS    : rhos,
                    DETAILS_CS      : cs }

        # Save Dictionary
        aux.dictToNPZ(details, sav)

        # Find and report progress
        if( verbose ):
            filesSize += os.path.getsize(sav)

            now = datetime.now()
            dur = now-start

            statStr = aux.statusString(ii+1, NUM_SNAPS, dur)
            sys.stdout.write('\r - - - %s' % (statStr))
            sys.stdout.flush()
            if( ii+1 == NUM_SNAPS ): sys.stdout.write('\n')

    # } snap

    if( verbose ):
        aveFileSize = filesSize / NUM_SNAPS
        totSize = aux.bytesString(filesSize)
        aveSize = aux.bytesString(aveFileSize)
        print " - - - Saved Details NPZ files.  Total size = %s, Ave Size = %s" % (totSize, aveSize)


    return



def _loadBHDetails_ASCII(asciiFile, verbose=VERBOSE):

    ### Files have some blank lines in them... Clean ###
    lines = open(asciiFile).readlines()                                                             # Read all lines at once
    nums = len(lines)

    # Allocate storage
    ids    = np.zeros(nums, dtype=LNG)
    times  = np.zeros(nums, dtype=DBL)
    masses = np.zeros(nums, dtype=DBL)
    mdots  = np.zeros(nums, dtype=DBL)
    rhos   = np.zeros(nums, dtype=DBL)
    cs     = np.zeros(nums, dtype=DBL)

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

    return LNG(args[0]), DBL(args[1]), DBL(args[2]), DBL(args[3]), DBL(args[4]), DBL(args[5])






###  ==============================================================  ###
###  =============  BH / MERGER - DETAILS MATCHING  ===============  ###
###  ==============================================================  ###



def loadBHDetails(run, snap):
    detsName = GET_DETAILS_SAVE_FILENAME(run, snap, VERSION)
    dets = aux.npzToDict(detsName)
    return dets


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
    # returnKeys = [ DETAILS_IDS, DETAILS_TIMES, DETAILS_MASSES, DETAILS_MDOTS, DETAILS_RHOS, DETAILS_CS ]
    # If no match is found, return all values as this:
    missingValue = -1.0

    ### Make sure Details are Loaded and Appropriate ###

    # If details not provided for this snapshot, load them
    if( details == None ):
        if( verbose ): print " - - - No details provided, loading for snapshot %d" % (snap)
        details = loadBHDetails(run, snap)
        if( verbose ): print " - - - - Loaded %d details" % (details[DETAILS_NUM])

    # Make sure these details match target snapshot and run
    assert details[DETAILS_RUN] == run, \
        "Details run %d does not match target %d!" % (details[DETAILS_RUN], run)

    assert details[DETAILS_SNAP] == snap, \
        "Details snap %d does not match target %d!" % (details[DETAILS_SNAP], snap)

    ### Find the Details index which matched ID and Nearest in Time ###

    # Find details indices with BH ID match
    inds = np.where( bhid == details[DETAILS_IDS] )[0]

    # If there are no matches, return None array
    if( len(inds) == 0 ):
        if( verbose ): print "No matches in snap %d for ID %d" % (snap, bhid)
        return { key : missingValue for key in DETAILS_PHYSICAL_KEYS }

    # Get times for matching details
    detTimes = details[DETAILS_TIMES][inds]
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
    bhDets = { key : details[key][retInds] for key in DETAILS_PHYSICAL_KEYS }

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
    detID   = -1*np.ones([numMergers, 2, 2], dtype=LNG)
    detTime = -1*np.ones([numMergers, 2, 2], dtype=DBL)
    detMass = -1*np.ones([numMergers, 2, 2], dtype=DBL)
    detMDot = -1*np.ones([numMergers, 2, 2], dtype=DBL)
    detRho  = -1*np.ones([numMergers, 2, 2], dtype=DBL)
    detCS   = -1*np.ones([numMergers, 2, 2], dtype=DBL)

    # Create dictionary of details for mergers
    mergDets = { DETAILS_IDS    : detID,
                 DETAILS_TIMES  : detTime,
                 DETAILS_MASSES : detMass,
                 DETAILS_MDOTS  : detMDot,
                 DETAILS_RHOS   : detRho,
                 DETAILS_CS     : detCS }


    # Iterate over each snapshot, with list of mergers in each `s2m`
    count = 0
    start = datetime.now()
    for snap,s2m in enumerate(mergers[MERGERS_MAP_STOM]):

        # If there are no mergers in this snapshot, continue to next iteration
        if( len(s2m) <= 0 ): continue

        # Convert from list to array
        search = np.array(s2m)


        ### Form List(s) of Target BH IDs ###

        # Add mergers from next snapshot
        if( snap < NUM_SNAPS-1 ):
            next = np.array(mergers[MERGERS_MAP_STOM][snap+1])
            if( len(next) > 0 ): search = np.concatenate( (search, next) )

        # Add mergers from previous snapshot
        if( snap > 0 ):
            prev = np.array(mergers[MERGERS_MAP_STOM][snap-1])
            if( len(prev) > 0 ): search = np.concatenate( (prev, search) )


        ### Prepare Detail and Merger Information for Matching ###
        searchNum = len(search)

        # Get all details (IDs and times) for this snapshot
        dets     = loadBHDetails(run, snap)
        detIDs   = dets[DETAILS_IDS]
        detTimes = dets[DETAILS_TIMES]

        # If there are no details in this snapshot (should only happen at end), continue
        if( len(detIDs) <= 0 ): continue

        # Get the BH merger info for this snapshot
        bhids    = mergers[MERGERS_IDS][search]
        #bhmasses = mergers[MERGERS_MASSES][search]
        bhtimes  = mergers[MERGERS_TIMES][search]
        # Duplicate `times` to match shape of `ids` and `masses`
        bhtimes = np.array([bhtimes, bhtimes]).T

        # Reshape 2D to 1D arrays for matching
        searchShape = np.shape(bhids)                                                               # Store initial shape [searchNum,2]
        bhids = bhids.reshape(2*searchNum)
        #bhmasses = bhmasses.reshape(2*searchNum)
        bhtimes = bhtimes.reshape(2*searchNum)

        ### Match Details to Mergers and Store ###

        # Find Details indices to match these BHs (just before and just after merger)
        indsBef, indsAft = MatchDetails.detailsForBlackholes(bhids, bhtimes, detIDs, detTimes)

        # Reshape indices to match mergers (in and out BHs)
        indsBef = np.array(indsBef).reshape(searchShape)
        indsAft = np.array(indsAft).reshape(searchShape)

        # Reshape det indices to match ``mergDets``
        # First makesure numbering is consistent!
        assert DETAILS_BEFORE == 0 and DETAILS_AFTER == 1, "`BEFORE` and `AFTER` broken!!"

        detInds = np.dstack([indsBef,indsAft])


        ### Store matches ###

        # Both 'before' and 'after'
        for BEF_AFT in [DETAILS_BEFORE, DETAILS_AFTER]:

            # Both 'in' and 'out' BHs
            for IN_OUT in [BH_IN, BH_OUT]:
                # Select only successful matches
                inds = np.where( detInds[:,IN_OUT,BEF_AFT] >= 0 )[0]
                #useInds = np.squeeze(detInds[inds,IN_OUT,BEF_AFT])
                useInds = detInds[inds,IN_OUT,BEF_AFT]

                # Select subset that hasn't been matched before
                newInds = np.where( mergDets[DETAILS_TIMES][search[inds],IN_OUT,BEF_AFT] < 0.0 )[0]

                if( len(newInds) > 0 ):
                    for KEY in DETAILS_PHYSICAL_KEYS:
                        mergDets[KEY][search[inds[newInds]],IN_OUT,BEF_AFT] = dets[KEY][useInds[newInds]]


                # Select subset with better matches

                # If we're looking for 'before', look for latest 'before'
                if( BEF_AFT == DETAILS_BEFORE ):
                    oldInds = np.where( mergDets[DETAILS_TIMES][search[inds],IN_OUT,BEF_AFT] < dets[DETAILS_TIMES][useInds] )[0]
                # If we're looking for 'adter', look for earliest 'after'
                else:
                    oldInds = np.where( mergDets[DETAILS_TIMES][search[inds],IN_OUT,BEF_AFT] > dets[DETAILS_TIMES][useInds] )[0]

                if( len(oldInds) > 0 ):
                    for KEY in DETAILS_PHYSICAL_KEYS:
                        mergDets[KEY][search[inds[oldInds]],IN_OUT,BEF_AFT] = dets[KEY][useInds[oldInds]]


        # Print progress
        count += len(bhids)
        if( verbose ):
            now = datetime.now()
            statStr = aux.statusString(count, 3*2*numMergers, now-start)
            sys.stdout.write('\r - - - %s' % (statStr))
            sys.stdout.flush()

    # } snap

    return mergDets




def _getPrecision(args):
    """

    """

    diffs = np.fabs(np.diff(sorted(args)))
    inds  = np.nonzero(diffs)
    if( len(inds) > 0 ): minDiff = np.min( diffs[inds] )
    else:                minDiff = np.power(10.0, _DEF_PRECISION)
    order = int(np.log10(0.49*minDiff))
    return order

# _getPrecision()


