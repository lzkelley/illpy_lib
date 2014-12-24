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
import os
from glob import glob

import numpy as np
from .. import AuxFuncs as aux
from .. import Constants as const
from .. import Cosmology

'''
from glob import glob
from datetime import datetime

### Import Local Parameters ###
from Settings import *

### Import Library Classes/Functions ###
import ill_lib
from ill_lib import AuxFuncs as aux
from ill_lib.Constants import *

PP_BH_DETAILS_DIR = "/n/home00/lkelley/illustris/post-process/Illustris-%d/bh-details/"
PP_BH_DETAILS_ASCII_FILENAME = "ill-%d_details_ascii_%03d.dat"
PP_BH_DETAILS_NPZ_FILENAME = "ill-%d_details_npz_%03d.npz"
'''

### Default Runtime Parameters ###
RUN = 3                                                                                             # Default Illustris run to load {1,3}
VERBOSE = True                                                                                      # Print verbose output during execution


# Where to find the 'raw' Illustris Merger files
_DETAILS_FILE_DIRS = { 3:'/n/ghernquist/Illustris/Runs/L75n455FP/output/blackhole_details/' }
_DETAILS_FILE_NAMES = "blackhole_details_*.txt"


# Where to save intermediate files
_MY_PATH = os.path.dirname(os.path.abspath(__file__))
_POST_PROCESS_PATH = _MY_PATH + "/post-process/ill-%d_bh-details/"
_DETAILS_SAVE_FILENAME = "ill-%d_details_snap-%d.npz"
_DETAILS_TEMP_FILENAME = "ill-%d_details_snap-%d_temp.txt"




### Dictionary Keys for Details Parametesr ###
DET_ID    = 'id'
DET_SCALE = 'scale'
DET_MASS  = 'mass'
DET_MDOT  = 'mdot'
DET_RHO   = 'rho'
DET_CS    = 'cs'

DET_NUM   = 6



###  ===================================  ###
###  =============  MAIN  ==============  ###
###  ===================================  ###



def main(run=RUN, verbose=VERBOSE):



    ### Get Illustris Details Filenames ###
    if( verbose ): print " - - Searching for details files"
    detailsDir = _DETAILS_FILE_DIRS[run]
    detailsFilenames = detailsDir + _DETAILS_FILE_NAMES
    detailsFiles = sorted(glob(detailsFilenames))
    numFiles = len(detailsFiles)
    if( numFiles < 1 ): raise RuntimeError("Error no files found in '%s'!" % (detailsDir))
    if( verbose ): print " - - - Found %d details Files" % (numFiles)

    
    ### Load Save File Names ###
    saveFiles, tempFiles = __getDetailsSaveFilenames(run)

    cosmo = Cosmology.Cosmology.Cosmology()
    

    ### Initialize Variables ###

    # Get Snapshot Times
    #timesFile = aux.getSnapshotTimesFilename(runNum, workDir)
    #times = aux.loadSnapshotTimes(runNum, runsDir, loadsave=timesFile, log=log)
    #numTimes = len(times)
    

    ### Organize Details by Snapshot Time; create new ASCII Files ###
    if( log ): log.log("Reorganizing by time")
    reorganizeBHDetails(detailsFiles, tempFiles, cosmo)


    ### Convert New Details ASCII Files, to new Details object files ###
    if( log ): log.log("Converting from ASCII to Objects", 1)
    convertDetailsASCIItoObj(newDetASCIIFilenames, newDetObjFilenames, log=log)

    return






def reorganizeBHDetails(oldFilenames, newFilenames, cosmo):

    numNewFiles = len(newFilenames)
    numOldFiles = len(oldFilenames)

    ### Open new ASCII details files ###
    if( log ): log.log("Opening %d New Files" % (numNewFiles) )
    newFiles = [ open(dname, 'w') for dname in newFilenames ]


    ### Iterate over all Illustris Details Files ###
    if( log ): log.log("Organizing data by Time")
    start = datetime.now()
    startOne = datetime.now()
    for ii,oldname in enumerate(oldFilenames):

        # Iterate over each entry in file
        for dline in open(oldname):
            detTime = DBL( dline.split()[1] )
            # Find the time bin given left edges (hence '-1'); include right-edges ('right=True')
            snapNum = np.digitize([detTime], times, right=True) - 1
            # Write line to apropriate file
            newFiles[snapNum].write( dline )
        # } for dline


        # Print where we are, and duration
        now = datetime.now()
        dur = str(now-start)
        durOne = str(now-startOne)
        if( log ): log.log("%d/%d after %s/%s" % (ii, numOldFiles, durOne, dur))
        startOne = now

    # } for ii

    if( log ): log.log("Done after %s" % (str(datetime.now()-start)) )

    # Close out details files.
    aveSize = 0.0
    for ii, newdf in enumerate(newFiles):
        newdf.close()
        aveSize += os.path.getsize(newdf.name)

    aveSize = aveSize/(1.0*len(newFiles))

    sizeStr = aux.convertDataSizeToString(aveSize)
    if( log ): log.log("Closed new files.  Average size %s" % (sizeStr) )

    return



def convert_ASCII_to_NPZ(run):

    start = datetime.now()
    last = start

    for snap in range(NUM_SNAPS):

        ascName = getFilename_BHDetails_ASCII(run, snap)
        npzName = getFilename_BHDetails_NPZ(run, snap)

        # Load ASCII File
        detList = loadBHDetails_ASCII( ascName )
        # Save NPZ File
        saveBHDetails_NPZ(detList, npzName)

        stop = datetime.now()
        print "%d / %d  After  %s / %s" % (snap, NUM_SNAPS, str(stop-last), str(stop-start))
        last = stop


    return



def loadBHDetails_ASCII(fileName, log=None):

    if( log ): log.log("loadIllustrisBHDetails()")

    # Load details Data from Illutris Files
    if( log ): log.log("Parsing details lines for '%s'" % (fileName) )

    ### Files have some blank lines in them... Clean ###
    ascLines = open(fileName).readlines()                                                           # Read all lines at once
    tmpList = [ [] for ii in range(len(ascLines)) ]                                                 # Allocate space for all lines
    num = 0
    # Iterate over lines, storing only those with content
    for aline in ascLines:
        aline = aline.strip()
        if( len(aline) > 0 ):
            tmpList[num] = parseIllustrisBHDetailsLine(aline)
            num += 1

    # Trim excess
    del tmpList[num:]

    return tmpList


def parseIllustrisBHDetailsLine(instr):
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

    return LONG(args[0]), DBL(args[1]), DBL(args[2]), DBL(args[3]), DBL(args[4]), DBL(args[5])


def loadBHDetails_NPZ(run, snap):
    detsName = getFilename_BHDetails_NPZ(run,snap)
    dat = np.load(detsName)
    return dat


def saveBHDetails_NPZ(dets, saveFilename, log=None):
    """
    Save details object using pickle.

    Overwrites any existing file.  If directories along the path don't exist,
    they are created.
    """

    # Make sure output directory exists
    saveDir, saveName = os.path.split(saveFilename)
    aux.checkDir(saveDir)

    ### Convert From List of Lists to Arrays ###
    nums = len(dets)
    ids = np.zeros(nums, dtype=LONG)
    scale = np.zeros(nums, dtype=DBL)
    mass = np.zeros(nums, dtype=DBL)
    mdot = np.zeros(nums, dtype=DBL)
    rho = np.zeros(nums, dtype=DBL)
    cs = np.zeros(nums, dtype=DBL)

    for ii, row in enumerate(dets):
        ids[ii]   = row[0]
        scale[ii] = row[1]
        mass[ii]  = row[2]
        mdot[ii]  = row[3]
        rho[ii]   = row[4]
        cs[ii]    = row[5]


    ### Create Dictionary and save to NPZ ###
    dat = { DET_ID    : ids,
            DET_SCALE : scale,
            DET_MASS  : mass,
            DET_MDOT  : mdot,
            DET_RHO   : rho,
            DET_CS    : cs }

    np.savez(saveFilename, **dat)

    print "\tSaved to '%s'" % (saveFilename)
    print "\tSaved size = %s" % (aux.getFileSizeString(saveFilename))

    return



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





def __getDetailsSaveFilenames(run):
    """
    Construct the NPZ filename to save/load mergers from.

    If the directory doesn't exist, it is created.

    Arguments
    ---------
    run : int, illustris run number {1,3}

    Returns
    -------
    savefile : str, name of the save file
    """

    savedir = _POST_PROCESS_PATH % (run)
    if( savedir[-1] != '/' ): savedir += '/'
    if( not os.path.isdir(savedir) ): os.makedirs(savedir)

    savefiles = []
    tempfiles = []
    for snap in range(const.NUM_SNAPS):
        tempfiles.append(savedir + _DETAILS_TEMP_FILENAME % (run, snap) )
        savefiles.append(savedir + _DETAILS_SAVE_FILENAME % (run, snap) )

    
    return savefiles, tempfiles




if __name__ == "__main__": main()
