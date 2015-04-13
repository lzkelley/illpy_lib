
import sys
import os
from datetime import datetime

#import readtreeHDF5
#import readsubfHDF5
import arepo

import illpy
from illpy import Cosmology
from illpy.Constants import *
from illpy import AuxFuncs as aux

from Constants import *





def loadSubhaloParticles(run, snapNum, subhaloInds, loadsave=True, verbose=VERBOSE):
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
    subhalos    : <dict>[N], list of dictionaries of particle data

    """

    # Make sure input indices are iterable
    if( not np.iterable(subhaloInds) ): subhInds = np.array([subhaloInds])
    else:                               subhInds = np.array( subhaloinds )

    if( verbose ): print " - - Subhalos.loadSubhaloParticles()"

    if( verbose ): print " - - - Loading %d subhalos" % (len(subhaloInds))
    groupCat = None
    
    subhalos = []
    for ii,shind in enumerate(subhaloInds):
        fileName = SUBHALO_PARTICLES_FILENAMES(run, snapNum, shind)
        if( verbose ): print " - - - - %d : Subhalo %d - '%s'" % (ii, shind, fileName)

        loadsave_flag = loadsave
        if( loadsave_flag ):
            if( os.path.exists(fileName) ):
                if( verbose ): print " - - - - - Loading from previous save"
                subhaloData = aux.npzToDict(fileName)
            else:
                print "``loadsave`` file '%s' does not exist!" % (fileName)
                loadsave_flag = False


        if( not loadsave_flag ):
            if( verbose ): print " - - - - - Reloading EplusA Particles from snapshot"
            subhaloData, groupCat = _getSubhaloParticles(run, snapNum, shind, 
                                                         groupCat=groupCat, verbose=verbose)
            aux.dictToNPZ(subhaloData, fileName, verbose=True)


        subhalos.append(subhaloData)

    # } ii

    subhalos = np.array(subhalos)
    return subhalos

# loadSubhaloParticles()



def _importSubhaloParticles(run, snapNum, subhaloInd, groupCat=None, verbose=VERBOSE):

    if( verbose ): print " - - Subhalos._importSubhaloParticles()"

    # Get snapshot file path and filename
    snapshotPath = ILLUSTRIS_OUTPUT_SNAPSHOT_FIRST_FILENAME(run, snapNum)

    ### If Group Catalog is not Provided, load it ###
    if( groupCat is None ):

        # Get group catalog file path and filename
        groupPath = ILLUSTRIS_OUTPUT_GROUP_FIRST_FILENAME(run, snapNum)

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

    return dataDict, groupCat

# _importSubhaloParticles()
