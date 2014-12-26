# ==================================================================================================
# DetailsManager.py
# -----------------
#
#
#
#
#
#
# Illustris BH Details Files:
# 0   1          2    3     4  5
# ID scalefactor mass mdot rho cs
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================


### Builtin Modules ###
import os, sys
from glob import glob

import numpy as np
from .. import AuxFuncs as aux
from .. import Constants as const
from .. import illcosmo

import datetime

_DOUBLE = np.float64
_LONG = np.int64


### Default Runtime Parameters ###
RUN = 3                                                                                             # Default Illustris run to load {1,3}
VERBOSE = True                                                                                      # Print verbose output during execution

REDO_TEMP = False                                                                                   # Re-create temporary files

# Where to find the 'raw' Illustris Merger files
_DETAILS_FILE_DIRS = { 3:'/n/ghernquist/Illustris/Runs/L75n455FP/output/blackhole_details/' }
_DETAILS_FILE_NAMES = "blackhole_details_*.txt"

_PRINT_INTERVAL = 2e4                                                                               # Interval at which to print status


# Where to save intermediate files
_MY_PATH = os.path.dirname(os.path.abspath(__file__))
_POST_PROCESS_PATH = _MY_PATH + "/post-process/ill-%d_bh-details/"
_DETAILS_SAVE_FILENAME = "ill-%d_details_snap-%d.npz"
_DETAILS_TEMP_FILENAME = "ill-%d_details_snap-%d_temp.txt"

details_temp_filename = lambda x,y: (_POST_PROCESS_PATH % (x)) + (_DETAILS_TEMP_FILENAME % (x,y))
details_save_filename = lambda x,y: (_POST_PROCESS_PATH % (x)) + (_DETAILS_SAVE_FILENAME % (x,y))


### Dictionary Keys for Details Parametesr ###
_DETAIL_IDS     = 'id'
_DETAIL_TIMES   = 'times'
_DETAIL_MASSES  = 'masses'
_DETAIL_MDOTS   = 'mdots'
_DETAIL_RHOS    = 'rhos'
_DETAIL_CS      = 'cs'
_DETAIL_RUN     = 'run'
_DETAIL_NUM     = 'num'
_DETAIL_CREATED = 'created'




###  ===================================  ###
###  =============  MAIN  ==============  ###
###  ===================================  ###



def main(run=RUN, verbose=VERBOSE):

    if( verbose ): print "\nBHDetails.py\n"

    ### Load Variables and Parameters ###

    # Load cosmology
    cosmo = illcosmo.Cosmology()
    snapTimes = cosmo.snapshotTimes()                                                               # Scale-factor of each snapshot


    ### Organize Details by Snapshot Time; create new, temporary ASCII Files ###

    # See if all temp files already exist
    tempExist = aux.filesExist(tempFiles)

    # If temp files dont exist, or we WANT to redo them, then create temp files
    if( not tempExist or REDO_TEMP ):

        # Get Illustris BH Details Filenames
        if( verbose ): print " - Finding Illustris BH Details files"
        detailsFiles = _getIllustrisDetailsFilenames(run, verbose)
        if( len(detailsFiles) < 1 ): raise RuntimeError("Error no details files found!!")

        # Reorganize into temp files
        if( verbose ): print " - Reorganizing details into temporary files"
        _reorganizeBHDetails(detailsFiles, snapTimes, run, verbose=verbose)

    else:
        if( verbose ): print " - All temporary files already exist."

    # Confirm all temp files exist
    tempExist = aux.filesExist(tempFiles)

    # If files are missing, raise error
    if( not tempExist ):
        print "Temporary Files missing!  First file = '%s'" % (tempFiles[0])
        raise RuntimeError("Temporary Files missing!")



    ### Convert temp ASCII Files, to new Details object files ###

    if( verbose ): print " - Converting temporary files to NPZ"
    _convertDetailsASCIItoNPZ(cosmo.num, run, verbose=verbose)

    return




def _reorganizeBHDetails(detFiles, times, run, verbose=VERBOSE):

    numOldFiles = len(detFiles)

    if( verbose ): 
        print " - - Counting lines in Illustris Details files"
        numLines = aux.estimateLines(detFiles)
        print " - - - Estimate %d (%.1e) lines in %d files" % (numLines, numLines, numOldFiles)


    ### Open new ASCII, Temp details files ###
    if( verbose ): print " - - Opening new files"
    tempFiles = [ open(details_temp_filename(run,snap), 'w') for snap in len(times) ]

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
                    statStr = statusString(count, numLines, dur)
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
    numFiles = len(saves)
    for ii in enumerate(zip(temps,saves)):

        tmp = details_temp_filename(run, ii)
        sav = details_save_filename(run, ii)

        # Load details from ASCII File
        ids, times, masses, mdots, rhos, cs = _loadBHDetails_ASCII(tmp)

        # Store details in dictionary
        details = { _DETAIL_NUM : len(ids),
                    _DETAIL_RUN : run,
                    _DETAIL_CREATED : datetime.datetime.now().ctime(),
                    _DETAIL_IDS    : ids,
                    _DETAIL_TIMES  : times,
                    _DETAIL_MASSES : masses,
                    _DETAIL_MDOTS  : mdots,
                    _DETAIL_RHOS   : rhos,
                    _DETAIL_CS     : cs }

        np.savez(sav, **details)
        if( not os.path.exists(sav) ):
            raise RuntimeError("Could not save to file '%s'!!" % (sav) )


        # Find and report progress
        if( verbose ):
            filesSize += os.path.getsize(sav)

            now = datetime.datetime.now()
            dur = now-start

            statStr = aux.statusString(ii+1, numFiles, dur)
            sys.stdout.write('\r - - - %s' % (statStr))
            sys.stdout.flush()

    # } ii

    if( verbose ): 
        aveFileSize = filesSize / numFiles
        totSize = aux.bytesString(filesSize)
        aveSize = aux.bytesString(filesSize)
        print " - - - Saved Details NPZ files.  Total size = %s, Ave Size = %s" % (totSize, aveSize)
        print " - - - - First: '%s'" % (saves[0])
        print " - - - - Last : '%s'" % (saves[-1])

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




def loadBHDetails_NPZ(run, snap):
    detsName = details_save_filename(run,snap)
    dat = np.load(detsName)
    return dat




###  =====================================================  ###
###  =============  BH / DETAILS MATCHING  ===============  ###
###  =====================================================  ###



def detailsForBH(bhid, snapDets, side=None, log=None):
    """
    Retrieve the details entry for a particular BH at a target snapshot.


    Parameters
    ----------
    bhid : int
        ID of the target BH
    snapDets :
    side : {'left', 'right', default=None}, optional
        Which matching elements to return.
        None : return all matches
        'left' : return the earliest match
        'right' : return the latest match
    log : class ObjLog, optional


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

    if( log ): log.log("detailsForBH()")
    retErr = [None]*5

    '''
    ### Load Details for this Snapshot ###
    snapDets = loadBHDetails_NPZ(run, snap)
    '''

    ### Find the Details index which matched ID and Nearest in Time ###

    # Find details indices with BH ID match
    inds = np.where( bhid == snapDets[DET_ID] )[0]

    # If there are no matches, return None
    if( len(inds) == 0 ):
        if( log ): log.log("No matches in snap %d for ID %d" % (snap, bhid) )
        return retErr

    # Get times for matching details
    detTimes = snapDets[DET_SCALE][inds]
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
    # Create output arrays/value
    scale = snapDets[DET_SCALE][retInds]
    mass  = snapDets[DET_MASS ][retInds]
    mdot  = snapDets[DET_MDOT ][retInds]
    rho   = snapDets[DET_RHO  ][retInds]
    cs    = snapDets[DET_CS   ][retInds]

    return scale, mass, mdot, rho, cs




if __name__ == "__main__": main()
