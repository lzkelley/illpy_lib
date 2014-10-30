# ==================================================================================================
# AuxFuncs.py - Auxilliary functions
# -----------
#
#
# Functions
# ---------
# - Merger Files
#   + getMergerFiles(target)     : for a given target 'run', get the list of merger filenames
#   + loadMergerFile(mfile)      : load the given merger file into a list of Merger objects
#   + loadAllMergers(target)     : load all mergers from the target run
#
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================

import numpy as np
import warnings
import os
import sys
import matplotlib                  as mpl
import cPickle as pickle
from matplotlib import pyplot      as plt

from glob import glob
from datetime import datetime

from Constants import *
from ObjMergers import Mergers
from ObjDetails import Details

import arepo


DT_THRESH = 1.0e-5                                                                                  # Frac diff b/t times to accept as equal




def verbosePrint(nv, arg):
    if( nv >= 0 ):
        prep = " -"*nv
        if( nv > 0 ): prep += " "
        print prep + arg
    





###  =====================================  ###
###  ==========  DETAILS FILES  ==========  ###
###  =====================================  ###


def getIllustrisBHDetailsFilenames(runNum, runsDir, verbose=-1):
    '''Get a list of 'blackhole_details' files for a target Illustris simulation'''                 

    if( verbose >= 0 ): vb = verbose+1
    else:               vb = -1

    verbosePrint(vb, "getIllustrisBHDetailsFilenames()")

    detailsNames      = np.copy(runsDir).tostring()
    if( not detailsNames.endswith('/') ): detailsNames += '/'
    detailsNames += RUN_DIRS[runNum]
    if( not detailsNames.endswith('/') ): detailsNames += '/'
    detailsNames += BH_DETAILS_FILENAMES

    verbosePrint(vb+1, "Searching '%s'" % detailsNames)
    files        = sorted(glob(detailsNames))                                                       # Find and sort files
    verbosePrint(vb+2, "Found %d files" % (len(files)) )

    return files


def getBHDetailsASCIIFilename(runNum, snapNum, workDir):
    detsDir = workDir + (BH_DETAILS_DIR % (runNum))
    checkDir(detsDir)                                                                               # Make sure directory exists
    asciiFilename = detsDir + (BH_DETAILS_ASCII_FILENAME % (runNum, snapNum))
    return asciiFilename


def getBHDetailsObjFilename(runNum, snapNum, workDir):
    detsDir = workDir + (BH_DETAILS_DIR % (runNum))
    checkDir(detsDir)                                                                               # Make sure directory exists
    asciiFilename = detsDir + (BH_DETAILS_OBJ_FILENAME % (runNum, snapNum))
    return asciiFilename



def loadIllustrisBHDetails(fileName, verbose=-1):

    if( verbose >= 0 ): vb = verbose+1
    else:               vb = -1
    if( verbose >= 0 ): verbosePrint(vb, "loadIllustrisBHDetails()")
    
    # Load details Data from Illutris Files
    if( verbose >= 0 ): verbosePrint(vb+1,"Parsing details lines for '%s'" % (fileName) )
    #tmpList = [ parseIllustrisBHDetailsLine(dline) for dline in open(fileName) ]

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

    ### Fill merger object with Merger Data ###
    if( verbose >= 0 ): verbosePrint(vb+1,"Creating details object")
    details = Details( num )
    for ii, tmp in enumerate(tmpList):
        details[ii] = tmp

    return details



def loadBHDetails(runNum, snapNum, workDir, verbose=-1):

    if( verbose >= 0 ): vb = verbose+1
    else:               vb = -10
    
    verbosePrint(vb, "loadBHDetails()")
    
    detsFilename = getBHDetailsObjFilename(runNum, snapNum, workDir)
    verbosePrint(vb+1, 'File %s' % (detsFilename) )
    if( not os.path.exists(detsFilename) ):
        raise RuntimeError("NO file %s!" % (detsFilename) )

    return loadBHDetailsFromSave(detsFilename)


    '''
    if( verbose >= 0 ): verbosePrint(vb+1, "Loading Details from run %d" % (runNum))

    load = False
    save = False
    if( loadFile ): load = True
    if( saveFile ): save = True

    # Load an existing save file (NPZ)
    if( load ):
        if( verbose >= 0 ): verbosePrint(vb+1,"Trying to load details from '%s'" % (loadFile))
        # Try to load save-file
        try: details = loadBHDetailsFromSave(loadFile)
        # Fall back to loading details from merger-files
        except Exception, err:
            if( verbose >= 0 ): verbosePrint(vb+2,"failed '%s'" % err.message)
            load = False


    # Load Details from Illustris merger files
    if( not load or len(details) == 0 ):
        verbosePrint(vb+1,"Loading details directly from Illustris Merger files")
        details = loadAllIllustrisBHDetails(runNum, runsDir, verbose=vb)

    if( verbose >= 0 ): verbosePrint(vb+1,"Loaded %d details." % (len(details)) )

    # Save Details to save-file if desired
    if( save and len(details) > 0 ): saveBHDetails(details, saveFile, verbose=vb)

    return details
    '''




def parseIllustrisBHDetailsLine(instr):
    '''
    Parse a line from an Illustris blachole_details_#.txt file

    The line is formatted (in C) as:
        "BH=%llu %g %g %g %g %g\n",
        (long long) P[n].ID, All.Time, BPP(n).BH_Mass, mdot, rho, soundspeed

    return ID, time, mass, mdot, rho, cs
    '''
    args = instr.split()

    # First element is 'BH=########', trim to just the id number
    args[0] = args[0].split("BH=")[-1]

    return LONG(args[0]), DBL(args[1]), DBL(args[2]), DBL(args[3]), DBL(args[4]), DBL(args[5])



def saveBHDetails(details, saveFilename, verbose=-1):
    '''
    Save details object using pickle.

    Overwrites any existing file.  If directories along the path don't exist,
    they are created.
    '''

    if( verbose >= 0 ): vb = verbose+1
    if( verbose >= 0 ): verbosePrint(vb,"saveBHDetails()")

    # Make sure output directory exists
    saveDir, saveName = os.path.split(saveFilename)
    checkDir(saveDir)

    # Save binary pickle file
    if( verbose >= 0 ): verbosePrint(vb+1,"Saving details to '%s'" % (saveFilename))
    saveFile = open(saveFilename, 'wb')
    pickle.dump(details, saveFile)
    saveFile.close()
    if( verbose >= 0 ): verbosePrint(vb+1,"Saved, size %s" % getFileSizeString(saveFilename))
    return



def reorganizeBHDetails(oldFilenames, newFilenames, times, verbose=-1):

    if( verbose >= 0 ): vb = verbose+1
    else:               vb = -1

    if( vb >= 0 ): verbosePrint(vb, "reorganizeBHDetails()")
    numNewFiles = len(newFilenames)
    numOldFiles = len(oldFilenames)    

    ### Open new ASCII details files ###
    if( vb >= 0 ): verbosePrint(vb+1, "Opening %d New Files" % (numNewFiles) )
    newFiles = [ open(dname, 'w') for dname in newFilenames ]


    ### Iterate over all Illustris Details Files ###
    if( vb >= 0 ): verbosePrint(vb+1, "Organizing data by Time")
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

        # Print where we are, and duration
        if( vb >= 0 ): 
            now = datetime.now()
            dur = str(now-start)
            durOne = str(now-startOne)
            verbosePrint(vb+2, "%d/%d after %s/%s" % (ii, numOldFiles, durOne, dur) )
            startOne = now

        # } for dline

    # } for ii

    if( vb >= 0 ): verbosePrint(vb+1, "Done after %s" % (str(datetime.now()-start)) )

    # Close out details files.
    aveSize = 0.0
    for ii, newdf in enumerate(newFiles):
        newdf.close()
        aveSize += os.path.getsize(newdf.name)

    aveSize = aveSize/(1.0*len(newFiles))
    if( vb >= 0 ): 
        sizeStr = convertDataSizeToString(aveSize)
        verbosePrint(vb+1, "Closed new files.  Average size %s" % (sizeStr) )

    return



def convertDetailsASCIItoObj(ascFilenames, objFilenames, verbose=-1):

    if( verbose >= 0 ): vb = verbose+1
    else:               vb = -1

    if( vb >= 0 ): verbosePrint(vb, "convertDetailsASCIItoObj()")
    numFiles = len(ascFilenames)

    if( vb >= 0 ): verbosePrint(vb+1, "Converting")
    start = datetime.now()
    startOne = datetime.now()
    for ii, [ascName,objName] in enumerate( zip(ascFilenames,objFilenames) ):

        details = loadIllustrisBHDetails( ascName, verbose=vb+1 )
        saveBHDetails(details, objName, verbose=vb+1)

        # Print where we are, and duration
        if( vb >= 0 ): 
            now = datetime.now()
            dur = str(now-start)
            durOne = str(now-startOne)
            verbosePrint(vb+2, "%d/%d after %s/%s" % (ii, numFiles, durOne, dur) )
            startOne = now

    # } ii


    return


def loadBHDetailsFromSave(loadFilename):
    '''
    Load details object from file.
    '''
    loadFile = open(loadFilename, 'rb')
    details = pickle.load(loadFile)
    return details



def convertBHDetails(runNum, runsDir, workDir, verbose=-1):
    '''
    Move details information from illustris files to new ones organized by time.

    '''

    if( verbose >= 0 ): vb = verbose + 1
    else:               vb = -1
    if( vb >= 0 ): verbosePrint(vb, "convertBHDetails()")

    if( len(workDir) > 0 and not workDir.endswith("/") ): workDir += "/"

    
    ### Initialize Variables ###

    # Get file names
    if( vb >= 0 ): verbosePrint(vb+1, "Loading details filenames")
    illDetFilenames = getIllustrisBHDetailsFilenames(runNum, runsDir, verbose=vb)
    numIllDetFiles = len(illDetFilenames)
    if( vb >= 0 ): verbosePrint(vb+2, "Loaded %d details filenames" % (numIllDetFiles))

    # Get Snapshot Times
    timesFile = getSnapshotTimesFilename(runNum, workDir)
    if( vb >= 0 ): verbosePrint(vb+1, "Loading times ('%s')" % (timesFile) )
    times = loadSnapshotTimes(runNum, runsDir, loadsave=timesFile, verbose=vb)
    numTimes = len(times)
    if( vb >= 0 ): verbosePrint(vb+2, "Loaded %d snapshot times" % (numTimes))

    ### Create names for detail-snapshot files. ###
    if( vb >= 0 ): verbosePrint(vb+1, "Constructing details filenames")
    # ASCII Files
    newDetASCIIFilenames = [ getBHDetailsASCIIFilename(runNum, ii, workDir) 
                             for ii in range(numTimes) ]
    if( vb >= 0 ): 
        verbosePrint(vb+2, "[%s,%s]" % (newDetASCIIFilenames[0], newDetASCIIFilenames[-1]) )             

    # Details Obj Files
    newDetObjFilenames = [ getBHDetailsObjFilename(runNum, ii, workDir) 
                           for ii in range(numTimes) ] 
    if( vb >= 0 ): 
        verbosePrint(vb+2, "[%s,%s]" % (newDetObjFilenames[0], newDetObjFilenames[-1]) )


    ### Organize Details by Snapshot Time; create new ASCII Files ###
    if( vb >= 0 ): verbosePrint(vb+1,"Reorganizing by time")
    #reorganizeBHDetails(illDetFilenames, newDetASCIIFilenames, times, verbose=vb)


    ### Convert New Details ASCII Files, to new Details object files ###
    if( vb >= 0 ): verbosePrint(vb+1,"Converting from ASCII to Objects")
    convertDetailsASCIItoObj(newDetASCIIFilenames, newDetObjFilenames, verbose=vb)
    



###  ======================================  ###
###  ==========  SNAPSHOT FILES  ==========  ###
###  ======================================  ###


def getSnapshotTimesFilename(runNum, workDir):
    timesFile = workDir + (SAVE_SNAPSHOT_TIMES_FILENAME % (runNum))
    return timesFile


def getSnapshotFilename(snapNum, runNum, runsDir, verbose=-1):
    """
    Given a run number and snapshot number, construct the approprirate snapshot filename.

    input
    -----
    snapNum : IN [int] snapshot number,       i.e. 100 for snapdir_100
    runNum  : IN [int] simulation run number, i.e. 3 for Illustris-3

    output
    ------
    return     filename : [str]
    """
    if( verbose >= 0 ): vb = verbose+1
    else:               vb = -1

    verbosePrint(vb, "getSnapshotFilename()")

    snapName      = np.copy(runsDir).tostring()
    if( not snapName.endswith('/') ): snapName += '/'
    snapName += RUN_DIRS[runNum]
    if( not snapName.endswith('/') ): snapName += '/'
    snapName += (SNAPSHOT_DIRS % (snapNum)) + (SNAPSHOT_FILENAMES % (snapNum))

    return snapName


def loadSnapshotTimes(runNum, runsDir, loadFile=None, saveFile=None, loadsave=None, verbose=-1):
    """
    Get the time (scale-factor) of each snapshot

    input
    -----
    runNum  : [int] simulation run number, i.e. 3 for Illustris-3


    output
    ------
    return   times : list of [float]   simulation times (scale-factors)

    """

    if( verbose >= 0 ): vb = verbose+1
    else:               vb = -1

    if( vb >= 0 ): verbosePrint(vb, "loadSnapshotTimes()")

    times = np.zeros(NUM_SNAPS, dtype=DBL)

    load = False
    save = False
    ### If loadsave is specified: load if file exists, otherwise save
    # Make sure 'loadFile'/'saveFile' are not specified along with 'loadsave'
    if( (loadsave and loadFile) or (loadsave and saveFile) ):
        raise RuntimeError("[AuxFuncs.loadSnapshotTimes()] Error: too many files!")
    elif( loadsave ):
        # If file already exists, load from it
        if( os.path.exists(loadsave) ): loadFile = loadsave
        # If file doesn't exist, save to it
        else:                           saveFile = loadsave

    if( loadFile ): load = True
    if( saveFile ): save = True

    # Load pre-existing times file
    if( load ):
        if( vb >= 0 ): verbosePrint(vb+1,"Loading snapshot times from '%s'" % (loadFile) )
        timesFile = np.load( loadFile )
        times[:]  = timesFile['times']
    # Re-extract times
    else:
        if( vb >= 0 ): verbosePrint(vb+1,"Extracting times from snapshots")
        for snapNum in range(NUM_SNAPS):
            snapFile = getSnapshotFilename(snapNum, runNum, runsDir)
            # Load only the header from the given snapshot
            snapHead = arepo.Snapshot(snapFile, onlyHeader=True)
            times[snapNum] = snapHead.time


    # Save snapshot times to NPZ file
    if( save ):
        if( vb >= 0 ): verbosePrint(vb+1, "Saving snapshot times to '%s'" % (saveFile) )
        np.savez(saveFile, times=times)


    return times

'''



###  =========================================  ###
###  =============  GROUP FILES  =============  ###
###  =========================================  ###

def getGroupFilename(snap_num, run_num=lyze.TARGET_RUN):
    """
    Given a run number and snapshot/catalog number (i.e. output time), construct group filename.

    input
    -----
    run_num  : IN [int] simulation run number, i.e. 3 for Illustris-3
    snap_num : IN [int] snapshot number,       i.e. 100 for snapdir_100

    output
    ------
    return     filename : [str]
    """
    groupName = (lyze.GROUP_CAT_DIRS % snap_num) + (lyze.GROUP_CAT_FILENAMES % snap_num)
    filename = lyze.RUN_DIRS[run_num] + groupName
    return filename



def constructOffsetTables(gcat):
    """
    Construct a table of particle offsets for each halo/subhalo.

    Based on code from filter.Halo.reset()

    Note that particles are ordered as follows in each:

        HALO    SUBHALO       PARTICLE
      | ==============================
      |    0 -------------------------     <-- first halo
      |               0 --------------     <--   "     "  's subhalo
      |                              0     <--   "     "        "     's first particle
      |                            ...
      |                        NS_0 -1     <-- number of particles in subhalo 0, minus 1
      |               1 ---------------
      |                           NS_0
      |                            ...
      |                 NS_0 + NS_1 -1
      |
      |               ...           ...
      |
      |         NS_0 -1 ---------------    <-- number of subhalos in halo 0, minus 1
      |                            ...
      |            NONE ---------------    <-- particles NOT IN A SUBHALO (end of halo block)
      |                            ...
      |                         NH_0-1     <-- number of particles in halo 0, minus 1
      |    1 --------------------------
      |
      |  ...         ...           ...
      |
      |  M-1 --------------------------    <-- number of Halos
      |
      |              ...           ...
      |
      | NONE --------------------------    <-- particles NOT IN A HALO    (end of entire file)
      |
      |              ...           ...
      | ===============================


    """

    DTYPE       = np.uint64

    numHalos    = gcat.npart_loaded[0]                                                          # Number of halos
    numSubhalos = gcat.npart_loaded[1]                                                          # Number of subhalos


    # Initialize offset tables for halos and subhalos; dimension for each particle type
    #    last element corresponds to the EOF offset,
    #    i.e. 1+NTOT  where NTOT is the total number of particles
    halo_offsets         = np.zeros( (numHalos+1   , 6), dtype=DTYPE )
    subhalo_offsets      = np.zeros( (numSubhalos+1, 6), dtype=DTYPE )
    halo_subhalo_offsets = np.zeros(  numHalos+1       , dtype=DTYPE )

    # offset_table
    # ------------
    #    One entry for first particles in each subhalo, particles in each halo and NO subhalo,
    #        and a single entry for particles with NO halo and NO subhalo
    #    Each entry is [ HALO, SUBHALO, PART0, ..., PART5 ]
    #        when there is no (sub)halo, the number will be '-1'
    #
    offset_table         = np.zeros( [numHalos+numSubhalos+1, 8], type=UINT)

    ### Determine halo offsets ###

    # For each particle, the offset for each Halo is the number of particles in previous halos
    halo_offsets[1:,:] = np.cumsum(gcat.group.GroupLenType[:,:], axis=0, dtype=DTYPE)

    ### Determine subhalo offsets ###

    subh = 0
    offs = 0
    cumPartTypes = np.zeros(6, dtype=UINT)
    cumHaloPartTypes = np.zeros(6, dtype=UINT)
    # Iterate over each Halo
    for ii in range(numHalos):
        cumHaloPartTypes += gcat.group.GroupLenType[ii,:]                                           # Add the number of particles in this halo
        # Iterate over each Subhalo in halo 'ii'
        for jj in range(gcat.group.GroupNsub[ii]):
            ### Add entry for each subhalo ###
            offset_table[offs] = np.append([ ii, subh ], cumPartTypes)
            subh += 1                                                                               # Increment subhalo number
            offs += 1                                                                               # Increment offset entry
            cumPartTypes += gcat.subhalo.SubhaloLenType[subh]                                       # Add particles in this subhalo

        # If there are more particles than in subhalos
        if( cumPartTypes != cumHaloPartTypes ):
            ### Add Entry for particles with NO subhalo ###
            offset_table[offs] = np.append([ii,-1], cumPartTypes)
            offs += 1                                                                               # Increment offset entry

        cumPartTypes = cumHaloPartTypes                                                             # Increment particle numbers to include this halo


    ### Add entry for end of all halo particles / start of particles with NO halo ###
    offset_table[offs] = np.append([-1,-1], cumPartTypes)

    subh = 0
    # Iterate over all halos
    for ii in np.arange(numHalos):
        # If this halo has subhalos, incorporate them
        if gcat.group.GroupNsubs[ii] > 0:

            # Zeroth subhalo has same offset as host halo (but different lengths)
            tmp = halo_offsets[ii,:]
            subhalo_offsets[subh,:] = tmp

            sub1 = subh + 1                                                                     # First subhalo index
            sub2 = subh + gcat.group.GroupNsubs[ii] + 1                                         # Last  subhalo index

            # To each subhalo after zeroth, add sum of particles up to previous subhalo
            subhalo_offsets[sub1:sub2,:] = (
                tmp +
                np.cumsum(gcat.subhalo.SubhaloLenType[subh:sub2-1,:], axis=0, dtype=DTYPE)
                )

            subh += gcat.group.GroupNsubs[ii]                                                   # Increment to zeroth subhalo of next halo

        halo_subhalo_offsets[ii+1] = ( halo_subhalo_offsets[ii] +
                                       gcat.group.GroupNsubs[ii] )

    # } i

    return halo_offsets, subhalo_offsets, offset_table

'''






###  ======================================  ###
###  =============  PHYSICS  ==============  ###
###  ======================================  ###

def aToZ(a, a0=1.0):
    """ Convert a scale-factor to a redshift """
    z = (a0/a) - 1.0
    return z

def zToA(z, a0=1.0):
    """ Convert a redshift to a scale-factor """
    a = a0/(1.0+z)
    return a



###  =======================================  ###
###  =============  PLOTTING  ==============  ###
###  =======================================  ###



def createFigures(nfigs=1):

    figs = [ plt.figure(figsize=FIG_SIZE) for ii in range(nfigs) ]
    for ff in figs:
        for axpos,axsize in zip(AX_POS,AX_SIZE):
            ff.add_axes(axpos + axsize)

    return figs


def saveFigure(fname, fig, verbose=-1):
    fig.savefig(fname)
    if( verbose >= 0 ): verbosePrint(verbose+1, "Saved figure '%s'" % (fname))
    return



###  ===================================  ###
###  =============  MATH  ==============  ###
###  ===================================  ###

def incrementRollingStats(avevar, count, val):
    """
    Increment a rolling average and stdev calculation with a new value

    avevar   : INOUT [ave, var]  the rolling average and variance respectively
    count    : IN    [int]       the *NEW* count for this bin (including new value)
    val      : IN    [float]     the new value to be included

    return
    ------
    avevar   : [float, float] incremented average and variation

    """
    delta      = val - avevar[0]

    avevar[0] += delta/count
    avevar[1] += delta*(val - avevar[0])

    return avevar


def finishRollingStats(avevar, count):
    """ Finish a rolling average and stdev calculation by find the stdev """

    if( count > 1 ): avevar[1] = np.sqrt( avevar[1]/(count-1) )
    else:            avevar[1] = 0.0

    return avevar



def getMagnitude(vect):
    """ Get the magnitude of a vector of arbitrary length """
    return np.sqrt( np.sum([np.square(vv) for vv in vect]) )




def findBins(target, bins, thresh=DT_THRESH):
    """
    Find the array indices (of "bins") bounding the "target"

    If target is outside bins, the missing bound will be 'None'
    low and high will be the same, if the target is almost exactly[*1] equal to a bin

    [*1] : How close counds as effectively the same is set by 'DEL_TIME_THRESH' below

    intput
    ------

    target  : [ ] value to be compared
    bins    : [ ] list of values to compare to the 'target'


    output
    ------

    Return   low  : [int] index below target (or None if none)
             high : [int] index above target (or None if none)
    """

    # deltat  : test whether the fractional difference between two values is less than threshold
    #           This function allows the later conditions to accomodate smaller numerical
    #           differences, between effectively the same value  (e.g.   1.0 vs. 0.9999999999989)
    #
    if( thresh == 0.0 ): deltat = lambda x,y : False
    else               : deltat = lambda x,y : np.abs(x-y)/np.abs(x) <= thresh

    nums   = len(bins)
    # Find bin above (or equal to) target
    high = np.where( (target <= bins) | deltat(target,bins) )[0]
    if( len(high) == 0 ): high = None
    else:
        high = high[0]                                                                              # Select first bin above target
        dhi  = bins[high] - target


    # Find bin below (or equal to) target
    low  = np.where( (target >= bins) | deltat(target,bins) )[0]
    if( len(low)  == 0 ): low  = None
    else:
        low  = low[-1]                                                                              # Select  last bin below target
        dlo  = bins[low] - target

    # Print warning on error
    if( low == None or high == None ):
        print "[AuxBlackholeFuncs.findBins] target = %e, bins = {%e,%e}; low,high = %s,%s !" % \
            ( target, bins[0], bins[-1], str(low), str(high) )
        raise RuntimeError("Could not find bins!")


    return [low,high,dlo,dhi]



'''

###  ====================================  ###
###  =============  OTHER  ==============  ###
###  ====================================  ###


def guessNumsFromFilename(fname):

    run = fname.split("Illustris-")[-1]
    run = run.split("/")[0]
    run = np.int(run)

    snap = fname.split("groups_")[-1]
    snap = snap.split("/")[0]
    snap = np.int(snap)

    return snap, run


'''






def getFileSizeString(filename, asstr=True):
    size = os.path.getsize(filename)
    return convertDataSizeToString(size)


def convertDataSizeToString(size):
    prefSize, pref = getPrefixed(size)
    unit = pref + 'B'
    return "%.2f %s" % (prefSize, unit)
    

def getPrefixed(tval):
    val = np.copy(tval)
    prefs = [ '', 'K' , 'M' , 'G' ]
    mult  = 1000.0

    cnt = 0
    while( val > mult ):
        val /= mult
        cnt  += 1
        if( cnt > 3 ):
            raise RuntimeError("Error: value too large '%s'" % (str(val)) )
    
    return val, prefs[cnt]



def checkDir(tdir):
    """
    Create the given directory if it doesn't already exist.
    return True if directory exists, false otherwise
    """
    if( len(tdir) > 0 ):
        if( not os.path.isdir(tdir) ): os.makedirs(tdir)
        if( not os.path.isdir(tdir) ): raise RuntimeError("Directory '%s' does not exist!" % (tdir) )

    return


#



