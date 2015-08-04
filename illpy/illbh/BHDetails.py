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


Details Dictionary
------------------
   { DETAILS_RUN       : <int>, illustris simulation number in {1,3}
     DETAILS_NUM       : <int>, total number of mergers `N`
     DETAILS_FILE      : <str>, name of save file from which mergers were loaded/saved
     DETAILS_CREATED   : <str>, date and time this file was created
     DETAILS_VERSION   : <flt>, version of BHDetails used to create file
   
     DETAILS_IDS       : <uint64>[N], BH particle ID numbers for each entry
     DETAILS_SCALES    : <flt64> [N], scale factor at which each entry was written
     DETAILS_MASSES    : <flt64> [N], BH mass
     DETAILS_MDOTS     : <flt64> [N], BH Mdot
     DETAILS_RHOS      : <flt64> [N], ambient mass-density
     DETAILS_CS        : <flt64> [N], ambient sound-speed
   }


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

import os, sys, warnings

import numpy as np
from datetime import datetime

from illpy.Constants import DTYPE

import BHConstants
from BHConstants import DETAILS

import zcode.InOut as zio

VERSION = 0.23                                    # Version of BHDetails

_DEF_PRECISION = -8                               # Default precision




def processDetails(run, loadsave=True, verbose=True):

    if( verbose ): print " - - BHDetails.processDetails()"

    # Organize Details by Snapshot Time; create new, temporary ASCII Files
    tempFiles = organizeDetails(run, loadsave=loadsave, verbose=verbose)

    # Create Dictionary Details Files
    saveFiles = formatDetails(run, loadsave=loadsave, verbose=verbose)

    return

# processDetails()




def organizeDetails(run, loadsave=True, verbose=True):

    if( verbose ): print " - - BHDetails.organizeDetails()"

    tempFiles = [ BHConstants.GET_DETAILS_TEMP_FILENAME(run, snap) for snap in xrange(NUM_SNAPS) ]

    # Check if all temp files already exist
    if( loadsave ):
        tempExist = all([ os.path.exists(tfil) for tfil in tempFiles ])
        if( not tempExist ):
            if( verbose ): print " - - - Temp files do not exist '%s'" % (tempFiles[0])
            loadsave = False


    # If temp files dont exist, or we WANT to redo them, then create temp files
    if( not loadsave ):
        # Get Illustris BH Details Filenames
        if( verbose ): print " - - - Finding Illustris BH Details files"
        rawFiles = BHConstants.GET_ILLUSTRIS_BH_DETAILS_FILENAMES(run, verbose)
        if( len(rawFiles) < 1 ): raise RuntimeError("Error no details files found!!")

        # Reorganize into temp files
        if( verbose ): print " - - - Reorganizing details into temporary files"
        _reorganizeBHDetailsFiles(run, rawFiles, tempFiles, verbose=verbose)


    # Confirm all temp files exist
    tempExist = all([ os.path.exists(tfil) for tfil in tempFiles ])

    # If files are missing, raise error
    if( not tempExist ):
        print "Temporary Files still missing!  '%s'" % (tempFiles[0])
        raise RuntimeError("Temporary Files missing!")


    return tempFiles

# organizeDetails()



def formatDetails(run, loadsave=True, verbose=True):

    if( verbose ): print " - - BHDetails.formatDetails()"

    # See if all npz files already exist
    saveFilenames = [ BHConstants.GET_DETAILS_SAVE_FILENAME(run, snap, VERSION) 
                      for snap in xrange(NUM_SNAPS) ]

    # Check if all save files already exist, and correct versions
    if( loadsave ):
        saveExist = all([os.path.exists(sfil) for sfil in saveFilenames])
        if( not saveExist ):
            print "BHDetails.formatDetails() : Save files do not exist e.g. '%s'" % \
                (saveFilenames[0])
            print "BHDetails.formatDetails() : converting raw Details files !!!"
            loadsave = False

    # Re-convert files
    if( not loadsave ):
        if( verbose ): print " - - - Converting temporary files to NPZ"
        _convertDetailsASCIItoNPZ(run, verbose=verbose)


    # Confirm save files exist
    saveExist = all([os.path.exists(sfil) for sfil in saveFilenames])

    # If files are missing, raise error
    if( not saveExist ):
        print "Save Files missing!  e.g. '%s'" % (saveFilenames[0])
        raise RuntimeError("Save Files missing!")

    return saveFilenames

# formatDetails()



def _reorganizeBHDetailsFiles(run, rawFilenames, tempFilenames, verbose=True):

    if( verbose ): print " - - BHDetails._reorganizeBHDetailsFiles()"

    # Load cosmology
    from illpy import illcosmo
    cosmo = illcosmo.Cosmology()
    snapScales = cosmo.scales()

    # Open new ASCII, Temp details files
    #    Make sure path is okay
    zio.checkPath(tempFilenames[0])
    # Open each temp file
    tempFiles = [ open(tfil, 'w') for tfil in tempFilenames ]

    numTemp = len(tempFiles)
    numRaw  = len(rawFilenames)
    if( verbose ): print " - - - Organizing %d raw files into %d temp files" % (numRaw, numTemp)


    ## Iterate over all Illustris Details Files
    #  ----------------------------------------
    if( verbose ): 
        print " - - - Sorting details into times of snapshots"
        pbar = zio.getProgressBar(numRaw)

    for ii,rawName in enumerate(rawFilenames):
        detLines = []
        detScales = []
        # Load all lines and entry scale-factors from raw details file
        for dline in open(rawName):
            detLines.append(dline)
            # Extract scale-factor from line
            detScale = DTYPE.SCALAR( dline.split()[1] )
            detScales.append(detScale)

        # Convert to array
        detLines  = np.array(detLines)
        detScales = np.array(detScales)

        # If file is empty, continue
        if( len(detLines) <= 0 or len(detScales) <= 0 ): continue

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
            if( len(inds) > 0 ): tempFiles[jj].writelines( detLines[inds] )

        # Print Progress
        if( verbose ): pbar.update(ii)

    # ii, rawName

    if( verbose ): pbar.finish()

    # Close out details files
    fileSizes = 0.0
    if( verbose ): print " - - - Closing files, checking sizes"
    for ii, newdf in enumerate(tempFiles):
        newdf.close()
        fileSizes += os.path.getsize(newdf.name)

    if( verbose ):
        aveSize = fileSizes/(1.0*len(tempFiles))
        sizeStr = zio.bytesString(fileSizes)
        aveSizeStr = zio.bytesString(aveSize)
        print " - - - - Total temp size = '%s', average = '%s'" % (sizeStr, aveSizeStr)


    if( verbose ): print " - - - Counting lines"
    inLines = zio.countLines(rawFilenames, progress=True)
    outLines = zio.countLines(tempFilenames, progress=True)
    if( verbose ): print " - - - - Input lines = %d, Output lines = %d" % (inLines, outLines)
    if( inLines != outLines ):
        print "in  file: ", rawFilenames[0]
        print "out file: ", tempFilenames[0]
        raise RuntimeError("WARNING: input lines = %d, output lines = %d!" % (inLines, outLines))

    return

# _reorganizeBHDetailsFiles()


def _convertDetailsASCIItoNPZ(run, verbose=True):
    """
    Convert all snapshot ASCII details files to dictionaries in NPZ files.
    """

    if( verbose ): print " - - BHDetails._convertDetailsASCIItoNPZ()"

    filesSize = 0.0
    sav = None

    ## Iterate over all Snapshots, convert from ASCII to NPZ
    # ------------------------------------------------------
    
    # Go through snapshots in random order to make better estimate of duration
    allSnaps = np.arange(NUM_SNAPS)
    np.random.shuffle(allSnaps)
    if( verbose ): 
        pbar = getProgressBar(NUM_SNAPS)
        print " - - - Converting files to NPZ"

    for ii,snap in enumerate(allSnaps):
        # Convert this particular snapshot
        saveFilename = _convertDetailsASCIItoNPZ_snapshot(run, snap, verbose=False)

        # Find and report progress
        if( verbose ):
            filesSize += os.path.getsize(saveFilename)
            pbar.update(ii)

    # ii,snap

    if( verbose ):
        pbar.finish()
        totSize = zio.bytesString(filesSize)
        aveSize = zio.bytesString(filesSize/NUM_SNAPS)
        print " - - - Total size = %s, Ave Size = %s" % (totSize, aveSize)

    return

# _convertDetailsASCIItoNPZ()


def _convertDetailsASCIItoNPZ_snapshot(run, snap, loadsave=True, verbose=True):
    """
    Convert a single snapshot ASCII Details file to dictionary saved to NPZ file.

    Makes sure the ASCII file exists, if not, ASCII 'temp' files are reloaded
    for all snapshots from the 'raw' details data from illustris.

    Arguments
    ---------

    Returns
    -------

    """

    if( verbose ): print " - - BHDetails._convertDetailsASCIItoNPZ_snapshot()"

    tmp = BHConstants.GET_DETAILS_TEMP_FILENAME(run, snap)
    sav = BHConstants.GET_DETAILS_SAVE_FILENAME(run, snap, VERSION)

    ## Make Sure Temporary Files exist, Otherwise re-create them
    if( not os.path.exists(tmp) ):
        print "BHDetails._convertDetailsASCIItoNPZ_snapshot(): no temp file '%s' " % (tmp)
        print "BHDetails._convertDetailsASCIItoNPZ_snapshot(): Reloading all temp files!!"
        tempFiles = organizeDetails(run, loadsave=loadsave, verbose=verbose)


    ## Try to load from existing save
    if( loadsave ):
        if( verbose ): print " - - - Loading from save '%s'" % (sav)
        if( os.path.exists(sav) ):
            details = zio.npzToDict(sav)
        else:
            if( verbose ): print " - - - '%s' does not exist!" % (sav)
            loadsave = False


    ## Load Details from ASCII, Convert to Dictionary and Save to NPZ
    if( not loadsave ):
        # Load details from ASCII File
        ids, scales, masses, mdots, rhos, cs = _loadBHDetails_ASCII(tmp)

        # Store details in dictionary
        details = { DETAILS.RUN     : run,
                    DETAILS.SNAP    : snap,
                    DETAILS.NUM     : len(ids),
                    DETAILS.FILE    : sav,
                    DETAILS.CREATED : datetime.now().ctime(),
                    DETAILS.VERSION : VERSION,

                    DETAILS.IDS     : ids,
                    DETAILS.SCALES  : scales,
                    DETAILS.MASSES  : masses,
                    DETAILS.MDOTS   : mdots,
                    DETAILS.RHOS    : rhos,
                    DETAILS.CS      : cs }

        # Save Dictionary
        zio.dictToNPZ(details, sav, verbose=verbose)

    return sav

# _convertDetailsASCIItoNPZ_snapshot()



def _loadBHDetails_ASCII(asciiFile, verbose=True):

    ## Files have some blank lines in them... Clean
    lines  = open(asciiFile).readlines()
    nums   = len(lines)

    # Allocate storage
    ids    = np.zeros(nums, dtype=DTYPE.ID)
    scales  = np.zeros(nums, dtype=DTYPE.SCALAR)
    masses = np.zeros(nums, dtype=DTYPE.SCALAR)
    mdots  = np.zeros(nums, dtype=DTYPE.SCALAR)
    rhos   = np.zeros(nums, dtype=DTYPE.SCALAR)
    cs     = np.zeros(nums, dtype=DTYPE.SCALAR)

    count = 0
    # Iterate over lines, storing only those with content (should be all)
    for lin in lines:
        lin = lin.strip()
        if( len(lin) > 0 ):
            tid,tim,mas,dot,rho,tcs = _parseIllustrisBHDetailsLine(lin)
            ids[count] = tid
            scales[count] = tim
            masses[count] = mas
            mdots[count] = dot
            rhos[count] = rho
            cs[count] = tcs
            count += 1

    # Trim excess (shouldn't be needed)
    if( count != nums ):
        trim = np.s_[count:]
        ids    = np.delete(ids, trim)
        scales = np.delete(scales, trim)
        masses = np.delete(masses, trim)
        mdots  = np.delete(mdots, trim)
        rhos   = np.delete(rhos, trim)
        cs     = np.delete(cs, trim)

    return ids, scales, masses, mdots, rhos, cs

# _loadBHDetails_ASCII()


def _parseIllustrisBHDetailsLine(instr):
    """
    Parse a line from an Illustris blachole_details_#.txt file

    The line is formatted (in C) as:
        "BH=%llu %g %g %g %g %g\n",
        (long long) P[n].ID, All.Time, BPP(n).BH_Mass, mdot, rho, soundspeed

    Arguments
    ---------
    
    Returns
    -------
        ID, time, mass, mdot, rho, cs

    """
    args = instr.split()
    # First element is 'BH=########', trim to just the id number
    args[0] = args[0].split("BH=")[-1]
    idn  = DTYPE.ID(args[0])
    time = DTYPE.SCALAR(args[1])
    mass = DTYPE.SCALAR(args[2])
    mdot = DTYPE.SCALAR(args[3])
    rho  = DTYPE.SCALAR(args[4])
    cs   = DTYPE.SCALAR(args[5])
    return idn, time, mass, mdot, rho, cs

# _parseIllustrisBHDetailsLine()




###  ==============================================================  ###
###  =============  BH / MERGER - DETAILS MATCHING  ===============  ###
###  ==============================================================  ###



def loadBHDetails(run, snap, loadsave=True, verbose=True):
    """
    Load Blackhole Details dictionary for the given snapshot.

    If the file does not already exist, it is recreated from the temporary ASCII files, or directly
    from the raw illustris ASCII files as needed.

    Arguments
    ---------
        run     : <int>, illustris simulation number {1,3}
        snap    : <int>, illustris snapshot number {0,135}
        loadsave <bool> :
        verbose  <bool> : print verbose output

    Returns
    -------
        dets    : <dict>, BHDetails dictionary object for target snapshot

    """

    if( verbose ): print " - - BHDetails.loadBHDetails()"

    detsName = BHConstants.GET_DETAILS_SAVE_FILENAME(run, snap, VERSION)

    ## Load Existing Save File
    if( loadsave ):
        if( verbose ): print " - - - Loading details from '%s'" % (detsName)
        if( os.path.exists(detsName) ):
            dets = zio.npzToDict(detsName)
        else:
            loadsave = False
            warnStr =  "%s does not exist!" % (detsName)
            warnings.warn(warnStr, RuntimeWarning)


    # If file does not exist, or is wrong version, recreate it
    if( not loadsave ):
        if( verbose): print " - - - Re-converting details"
        # Convert ASCII to NPZ
        saveFile = _convertDetailsASCIItoNPZ_snapshot(run, snap, loadsave=True, verbose=verbose)
        # Load details from newly created save file
        dets = zio.npzToDict(saveFile)

    return dets

# loadBHDetails()



def _getPrecision(args):
    """
    Estimate the precision needed to differenciate between elements of an array
    """
    diffs = np.fabs(np.diff(sorted(args)))
    inds  = np.nonzero(diffs)
    if( len(inds) > 0 ): minDiff = np.min( diffs[inds] )
    else:                minDiff = np.power(10.0, _DEF_PRECISION)
    order = int(np.log10(0.49*minDiff))
    return order

# _getPrecision()

