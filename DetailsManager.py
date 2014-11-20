# ==================================================================================================
# DetailsManager.py
# -----------------
# 
# 
#
# 
# 
#
# Contains
# --------
# + 
#
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================


### Builtin Modules ###
import random
import sys
import os

import numpy as np
import traceback as tb
import cPickle as pickle

from glob import glob
from datetime import datetime


### Custom Modules and Files ###
# Import Global Settings
from Settings import *
sys.path.append(*LIB_PATHS)
from Constants import *

# Import local project files and objects
#from ObjDetails import Details
import AuxFuncs as aux




###  ===================================  ###
###  =============  MAIN  ==============  ###
###  ===================================  ###



def main(): 

    ### Initialize Log File ###
    tbOffset = len(tb.extract_stack())-1

    logName  = aux.checkDir(LOGS_DIR)
    logName += "log_details.txt"

    log = Log(verbose=VERBOSE, clean=False, num=tbOffset)
    log.log("DetailsManager.py")

    start = datetime.now()
    
    
    ### Set Basic Parameters ###
    runNum = 3
    runsDir = "./Runs/"
    workDir = WORKING_DIR
    
    
    ### Run Conversion ###
    convertBHDetails(runNum, runsDir, workDir, verbose=verbose)

    stop = datetime.now()
    log.log("Done after %s\n\n" % (str(stop-start)))

    return




def convertBHDetails(runNum, runsDir, workDir, log=None):
    '''
    Move details information from illustris files to new ones organized by time.

    '''

    if( log ): log.log("convertBHDetails()")

    ### Initialize Variables ###

    # Get file names for pure Illustris Details files
    if( log ): log.log("Loading details filenames")
    illDetFilenames = aux.getIllustrisBHDetailsFilenames(runNum, runsDir, log=log)
    numIllDetFiles = len(illDetFilenames)
    if( log ): log.log("Loaded %d details filenames" % (numIllDetFiles) )

    # Get Snapshot Times
    timesFile = aux.getSnapshotTimesFilename(runNum, workDir)
    if( log ): log.log("Loading times ('%s')" % (timesFile) )
    times = aux.loadSnapshotTimes(runNum, runsDir, loadsave=timesFile, log=log)
    numTimes = len(times)
    if( log ): log.log("Loaded %d snapshot times" % (numTimes) )

    ### Create names for detail-snapshot files. ###
    if( log ): log.log("Constructing details filenames")
    # ASCII Files
    newDetASCIIFilenames = [ getBHDetailsASCIIFilename(runNum, ii, workDir)
                             for ii in range(numTimes) ]

    if( log ): log.log("[%s,%s]" % (newDetASCIIFilenames[0], newDetASCIIFilenames[-1]) )

    # Details Obj Files
    newDetObjFilenames = [ getBHDetailsObjFilename(runNum, ii, workDir)
                           for ii in range(numTimes) ]

    if( log ): log.log("[%s,%s]" % (newDetObjFilenames[0], newDetObjFilenames[-1]) )


    ### Organize Details by Snapshot Time; create new ASCII Files ###
    if( log ): log.log("Reorganizing by time")
    reorganizeBHDetails(illDetFilenames, newDetASCIIFilenames, times, verbose=vb)


    ### Convert New Details ASCII Files, to new Details object files ###
    if( log ): log.log("Converting from ASCII to Objects", 1)
    convertDetailsASCIItoObj(newDetASCIIFilenames, newDetObjFilenames, log=log)

    return



def reorganizeBHDetails(oldFilenames, newFilenames, times, log=None):

    if( log ): log.log("reorganizeBHDetails()")

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




def convertDetailsASCIItoObj(ascFilenames, objFilenames, log=None):

    if( log ): log.log("convertDetailsASCIItoObj()")
    numFiles = len(ascFilenames)

    start = datetime.now()
    startOne = datetime.now()
    for ii, [ascName,objName] in enumerate( zip(ascFilenames,objFilenames) ):

        details = loadIllustrisBHDetails( ascName, log=log)
        saveBHDetails(details, objName, log=log)

        # Print where we are, and duration
        now = datetime.now()
        dur = str(now-start)
        durOne = str(now-startOne)
        if( log ): log.log("%d/%d after %s/%s" % (ii, numFiles, durOne, dur) )
        startOne = now

    # } ii

    return


'''
def loadIllustrisBHDetails(fileName, log=None):

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

    ### Fill merger object with Merger Data ###
    if( log ): log.log("Creating details object")
    details = Details( num )
    for ii, tmp in enumerate(tmpList):
        details[ii] = tmp

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




def loadBHDetails(runNum, snapNum, workDir, log=None):

    if( log ): log.log("loadBHDetails()")

    detsFilename = getBHDetailsObjFilename(runNum, snapNum, workDir)
    if( log ): log.log('File %s' % (detsFilename) )
    if( not os.path.exists(detsFilename) ):
        raise RuntimeError("NO file %s!" % (detsFilename) )

    dets = loadBHDetailsFromSave(detsFilename)
    if( log ): log.log('Loaded %d details' % (len(dets)) )

    return dets




def saveBHDetails(details, saveFilename, log=None):
    '''
    Save details object using pickle.

    Overwrites any existing file.  If directories along the path don't exist,
    they are created.
    '''

    if( log ): log.log("saveBHDetails()")

    # Make sure output directory exists
    saveDir, saveName = os.path.split(saveFilename)
    aux.checkDir(saveDir)

    # Save binary pickle file
    if( log ): log.log("Saving details to '%s'" % (saveFilename) )
    saveFile = open(saveFilename, 'wb')
    pickle.dump(details, saveFile)
    saveFile.close()
    if( log ): log.log("Saved, size %s" % getFileSizeString(saveFilename) )

    return



def loadBHDetailsFromSave(loadFilename):
    '''
    Load details object from file.
    '''
    loadFile = open(loadFilename, 'rb')
    details = pickle.load(loadFile)
    return details


def getBHDetailsASCIIFilename(runNum, snapNum, workDir):
    detsDir = workDir + (BH_DETAILS_DIR % (runNum))
    aux.checkDir(detsDir)                                                                           # Make sure directory exists
    asciiFilename = detsDir + (BH_DETAILS_ASCII_FILENAME % (runNum, snapNum))
    return asciiFilename


def getBHDetailsObjFilename(runNum, snapNum, workDir):
    detsDir = workDir + (BH_DETAILS_DIR % (runNum))
    aux.checkDir(detsDir)                                                                           # Make sure directory exists
    asciiFilename = detsDir + (BH_DETAILS_OBJ_FILENAME % (runNum, snapNum))
    return asciiFilename







if __name__ == "__main__": main()
