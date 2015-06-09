
import sys
import os
from datetime import datetime

from lib import readtreeHDF5
from lib import readsubfHDF5
#import arepo

import illpy
from illpy import Cosmology
from illpy.Constants import *
from illpy import AuxFuncs as aux
from illpy.illbh.BHConstants import *

import Constants
from Constants import *


VERSION = 0.2
VERBOSE = True


def loadSubhaloParticles(run, snapNum, subhaloInds, noreturn=False, loadsave=True, verbose=VERBOSE):
    """
    Load the particle data from an Illustris snapshot corresponding to the target subhalo(s).

    Arguments
    ---------
    run         : <int>, illustris simulation number {1,3}
    snapNum     : <int>, illustris snapshot number {0,135}
    subhaloInds : array_like<int>[N], single or multiple subhalo indices to retrieve
    loadsave    : <bool>, optional=True, load previous save if possible
    verbose     : <bool>, optional=VERBOSE, verbose output

    Returns
    -------
    subhalos    : <dict>[N], list of 'Subhalo' dictionaries of particle data

    """

    yesReturn = not noreturn

    # Make sure input indices are iterable
    if( not np.iterable(subhaloInds) ): subhInds = np.array([subhaloInds])
    else:                               subhInds = np.array( subhaloInds )

    if( verbose ): print " - - Subhalos.loadSubhaloParticles()"

    if( verbose ): print " - - - Loading %d subhalos" % (len(subhInds))
    groupCat = None

    if( yesReturn ): subhalos = []

    ### Iterate over Target Subhalo Indices ###
    for ii,shind in enumerate(subhInds):
        fileName = Constants.GET_SUBHALO_PARTICLES_FILENAMES(run, snapNum, shind)
        if( verbose ): print " - - - - %d : Subhalo %d - '%s'" % (ii, shind, fileName)

        loadsave_flag = loadsave
        # Try to load Existing Save
        if( loadsave_flag ):
            # Make sure file exists
            if( os.path.exists(fileName) ):
                if( verbose ): print " - - - - - Loading from previous save"
                subhaloData = aux.npzToDict(fileName)
                # Really old version didn't have ``version`` information... catch that
                try:             loadVers = subhaloData[SUBHALO_VERSION]
                except KeyError: loadVers = -1.0

                # Make sure version is up to date
                if( loadVers != VERSION ):
                    print "Subhalos.loadSubhaloParticles() : Loaded version %s" % (str(loadVers))
                    print "Subhalos.loadSubhaloParticles() : VERSION %s" % (str(VERSION))
                    print "Subhalos.loadSubhaloParticles() : re-importing particle data!!"
                    loadsave_flag = False

            else:
                print "``loadsave`` file '%s' does not exist!" % (fileName)
                loadsave_flag = False

        # Re-import data directly from illustris snapshots
        if( not loadsave_flag ):
            if( verbose ): print " - - - - - Reloading EplusA Particles from snapshot"
            # Import data 
            subhaloData, groupCat = _importSubhaloParticles(run, snapNum, shind, 
                                                            groupCat=groupCat, verbose=verbose)
            # Save data
            aux.dictToNPZ(subhaloData, fileName, verbose=True)


        if( yesReturn ): subhalos.append(subhaloData)

    # } ii

    if( yesReturn ):
        subhalos = np.array(subhalos)
        return subhalos

    return

# loadSubhaloParticles()



def _importSubhaloParticles(run, snapNum, subhaloInd, groupCat=None, verbose=VERBOSE):

    import arepo

    if( verbose ): print " - - Subhalos._importSubhaloParticles()"

    # Get snapshot file path and filename
    snapshotPath = GET_ILLUSTRIS_SNAPSHOT_FIRST_FILENAME(run, snapNum)

    ### If Group Catalog is not Provided, load it ###
    if( groupCat is None ):

        # Get group catalog file path and filename
        groupPath = GET_ILLUSTRIS_GROUPS_FIRST_FILENAME(run, snapNum)

        # Load subfind catalog
        if( verbose ): print " - - - Loading subfind catalog from '%s'" % (groupPath)
        groupCat = arepo.Subfind(groupPath, combineFiles=True)



    ### Load Snapshot Data ###

    # Create filter for target subhalo
    filter = [ arepo.filter.Halo(groupCat, subhalo=subhaloInd) ]
    # Load snapshot
    if( verbose ): print " - - - Loading snapshot data"
    start = datetime.now()
    data = arepo.Snapshot(snapshotPath, filter=filter, fields=SNAPSHOT_PROPERTIES,
                          combineFiles=True, verbose=False )
    stop = datetime.now()
    if( verbose ): print " - - - - Loaded after %s" % (str(stop-start))

    ### Convert Snapshot Data into Dictionary ###

    dataDict = {}
    for snapKey in SNAPSHOT_PROPERTIES:
        dataDict[snapKey] = getattr(data, snapKey)

    dataDict[SUBHALO_ID]       = subhaloInd
    dataDict[SUBHALO_RUN]      = run
    dataDict[SUBHALO_SNAPSHOT] = snapNum
    dataDict[SUBHALO_CREATED]  = datetime.now().ctime()
    dataDict[SUBHALO_VERSION]  = VERSION

    return dataDict, groupCat

# _importSubhaloParticles()
