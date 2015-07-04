"""
Associate Merger BHs with the Subhalos they occur-in / belong-to.

"""

import numpy as np
from datetime import datetime
import sys
import os

from illpy.Constants import DTYPE, DENS_CONV, DIST_CONV, MASS_CONV, NUM_SNAPS
from illpy.illbh import BHMergers, BHConstants, BHMatcher
from illpy.illbh.BHConstants import *
from illpy.Subhalos import Profiler, Subhalo

from zcode.Constants import PC, KPC, MSOL
import zcode.InOut as zio
import zcode.Math  as zmath

import HaloOffsetTable
import arepo

VERBOSE = True


VERSION = 0.2

CHECK   = False

MSUBH_RUN        = "run"
MSUBH_CREATED    = "created"
MSUBH_VERSION    = "version"
MSUBH_FILENAME   = "filename"

MSUBH_SNAPSHOTS       = "snapshots"
MSUBH_BH_INDICES      = "indices_bh"
MSUBH_BH_IDS          = "ids_bh"
MSUBH_SUBHALO_INDICES = "indices_subhalo"

# Parameters from Group Catalog
MSUBH_CAT_GROUP_NUM        = "SubhaloGrNr"
MSUBH_CAT_HRAD             = "SubhaloHalfmassRad"
MSUBH_CAT_LEN_TYPE         = "SubhaloLenType"
MSUBH_CAT_MASS_TYPE        = "SubhaloMassType"
MSUBH_CAT_MASS_HRAD_TYPE   = "SubhaloMassInHalfRadType"
MSUBH_CAT_PARENT           = "SubhaloParent"
MSUBH_CAT_POS              = "SubhaloPos"
MSUBH_CAT_SFR              = "SubhaloSFR"
MSUBH_CAT_PHOTOS           = "SubhaloStellarPhotometrics"
MSUBH_CAT_VEL_DISP         = "SubhaloVelDisp"
MSUBH_CAT_VMAX             = "SubhaloVmax"
MSUBH_CAT_VMAX_RAD         = "SubhaloVmaxRad"

GROUP_CAT_PARAMS = [ MSUBH_CAT_GROUP_NUM, MSUBH_CAT_HRAD, MSUBH_CAT_LEN_TYPE, MSUBH_CAT_MASS_TYPE,
                     MSUBH_CAT_MASS_HRAD_TYPE, MSUBH_CAT_PARENT, MSUBH_CAT_POS, MSUBH_CAT_SFR,
                     MSUBH_CAT_PHOTOS, MSUBH_CAT_VEL_DISP, MSUBH_CAT_VMAX, MSUBH_CAT_VMAX_RAD ]


convDens = DENS_CONV*np.power(PC, 3.0)/MSOL
convDist = DIST_CONV/KPC


_MERGER_SUBHALOS_SAVE_FILENAME_BASE = "ill-%d_mergers_subhalos.npz"
def GET_MERGER_SUBHALOS_SAVE_FILENAME(run):
    return GET_PROCESSED_DIR(run) + _MERGER_SUBHALOS_SAVE_FILENAME_BASE % (run)



def loadMergerSubhaloCatalog(run, loadsave=True, verbose=VERBOSE, debug=False):

    if( verbose ): print " - - SubhalosOfMergers.loadMergerSubhaloCatalog()"
    saveFile = GET_MERGER_SUBHALOS_SAVE_FILENAME(run)

    NAME = "SubhalosOfMergers.loadMergerSubhaloCatalog()"

    ### Load Existing Save File ###
    if( loadsave ):
        if( verbose ): print " - - - Loading save '%s'" % (saveFile)
        # Make sure path exists
        if( os.path.exists(saveFile) ):
            mergerSubhalos = zio.npzToDict(saveFile)
            loadVers = mergerSubhalos[MSUBH_VERSION]
            # Check version number
            if( loadVers != VERSION ):
                print "%s : Loaded  Version '%s', Current '%s'" % \
                    (NAME, str(loadVers), str(VERSION))
                print "%s : Re-calculating subhalo indices!!"
                loadsave = False
            else:
                if( verbose ): print " - - - - Loaded"

        else:
            print "%s : File doesn't exist!  Re-calculating subhalo indices!" % (NAME)
            loadsave = False


    ### Recalculate Subhalo Indices ###
    if( not loadsave ):
        if( verbose ): print " - - - Recalculating subhalo indices"

        # Calculate Data
        mergerSubhalos = _calculateSubhaloMergerData(run, debug=debug)

        # Add Meta-Data
        mergerSubhalos[MSUBH_RUN]      = run
        mergerSubhalos[MSUBH_CREATED]  = datetime.now().ctime()
        mergerSubhalos[MSUBH_VERSION]  = VERSION
        mergerSubhalos[MSUBH_FILENAME] = saveFile

        # Save
        zio.dictToNPZ(mergerSubhalos, saveFile, verbose=True)


    return mergerSubhalos

# loadMergerSubhaloCatalog()




def _calculateSubhaloMergerData(run, check=CHECK, verbose=VERBOSE, debug=False):

    if( verbose ): print " - - SubhalosOfMergers._calculateSubhaloMergerData()"

    # Load Mergers
    mergers = BHMergers.loadFixedMergers(run)

    mergSnaps = mergers[MERGERS_MAP_MTOS]
    outIDs = mergers[MERGERS_IDS][:,BH_OUT]

    numMergers = len(outIDs)

    if( verbose ): print " - - - %d Total Mergers" % (numMergers)

    # Initialize Dictionary
    mergerSubhalos = {}
    mergerSubhalos[MSUBH_SNAPSHOTS]       = -1*np.ones(numMergers, dtype=DTYPE.INDEX)
    mergerSubhalos[MSUBH_BH_INDICES]      = -1*np.ones(numMergers, dtype=DTYPE.INDEX)
    mergerSubhalos[MSUBH_SUBHALO_INDICES] = -1*np.ones(numMergers, dtype=DTYPE.INDEX)
    mergerSubhalos[MSUBH_BH_IDS]          = -1*np.ones(numMergers, dtype=DTYPE.ID)


    # Iterate over snapshots
    start = datetime.now()
    for snap in reversed(xrange(NUM_SNAPS)):

        # Select BH IDs which merge in this snapshot (between previous and this snapshot)
        mergerInds = np.where( mergSnaps == snap )[0]

        # If there are mergers in this snapshot
        if( len(mergerInds) > 0 ):

            targetIDs = outIDs[mergerInds]
            if( debug ): print " - - - %d mergers for snapshot %d" % (len(mergerInds), snap)

            # Get Subhalo indices for this snapshot
            if( debug ): print " - - - Getting subhalo indices"
            bhInds, subhInds = HaloOffsetTable.getBHSubhaloIndices(run, snap, targetIDs,
                                                                   verbose=debug)

            # Retrieve target parameters from group catalog, perform index match checks if desired
            if( debug ): print " - - - Getting Group-catalog properties for subhalos"

            snapProps = _getSnapshotSubhaloProperties(run, snap, targetIDs, bhInds, subhInds,
                                                      check=check, verbose=debug)

            # Store General Data to Dictionary
            mergerSubhalos[MSUBH_SNAPSHOTS][mergerInds]       = snap
            mergerSubhalos[MSUBH_BH_INDICES][mergerInds]      = bhInds
            mergerSubhalos[MSUBH_SUBHALO_INDICES][mergerInds] = subhInds
            mergerSubhalos[MSUBH_BH_IDS][mergerInds]          = targetIDs

            # Iterate over Group Catalog Parameters and Store Data to Dictionary
            for PAR in GROUP_CAT_PARAMS:

                tempProps = snapProps[PAR]

                # Initialize Array Storage first time through
                if( not hasattr(mergerSubhalos, PAR) ):
                    # Determine appropriate data type, accomodate varying shapes of properties
                    temp = tempProps[0]
                    if( np.iterable(temp) ): useType = type(temp[0])
                    else:                    useType = type(temp)

                    # Determine appropriate shape
                    useShape = np.shape(tempProps)
                    if( len(useShape) == 1 ): useShape = numMergers
                    else:                     useShape = np.append([numMergers],useShape[-1])

                    # Initialize storage
                    mergerSubhalos[PAR] = -1*np.ones(useShape, dtype=useType)


                # Store data from snapshot
                mergerSubhalos[PAR][mergerInds] = tempProps

            # } PAR

        # } if

        # Print Progress
        if( verbose ):
            # Find out current duration
            now = datetime.now()
            dur = now-start

            # Print status and time to completion
            statStr = zio.statusString(snap+1, NUM_SNAPS, dur)
            sys.stdout.write('\r - - - - %s' % (statStr))
            sys.stdout.flush()


    # } snap

    if( verbose ): sys.stdout.write('\n')

    return mergerSubhalos

# _calculateSubhaloMergerData()




def _getSnapshotSubhaloProperties(run, snap, bhIDs, bhInds, subhInds, check=CHECK, verbose=VERBOSE):
    """
    Retrieve desired parameters from the group catalog for target subhalos.

    Target parameters are given by the keys stored in the ``GROUP_CAT_PARAMS`` list.
    Also, if ``check`` is 'True', perform comparison between group catalog and snapshot files
    to confirm that Blackhole-to-host-subhalo matches are correct.

    The properties are returned in a dictionary with keys matching those in ``GROUP_CAT_PARAMS``,
    each key gives an array of values with length equal to the number of target BHs (and subhalos).
    Invalid entries are set to '-1' (due to BHs without good index matches, or no subhalo match).

    Arguments
    ---------
    run       : <int>, illustris simulation number {1,3}
    snap      : <int>, illustris snapshot number {0,135}
    bhIDs     : <long>[N], list of target BH ID numbers
    bhInds    : <long>[N], list of target BH index numbers (for snapshot file)
    subhInds  : <long>[N], list of subhalo host indices of target BHs (for group find files)
    check     : <bool>, (optional=CHECK), perform comparison between groupfind and snapshots
    verbose   : <bool>, (optional=VERBOSE), print verbose output

    Returns
    -------
    props     : <dict>, dictionary of subhalo properties with keys from ``GROUP_CAT_PARAMS``

    """


    if( verbose ): print " - - SubhalosOfMergers._getSnapshotSubhaloProperties()"

    # Load Group Catalog
    groupCatFilename = GET_ILLUSTRIS_GROUPS_FIRST_FILENAME(run, snap)
    if( verbose ): print " - - - Loading group catalog '%s'" % (groupCatFilename)
    groupCat = arepo.Subfind(groupCatFilename, combineFiles=True, verbose=verbose)

    # Load Target Parameters from catalog
    subhaloProps = {}
    for PAR in GROUP_CAT_PARAMS:
        # Load parameters for all indices
        tempProps = getattr(groupCat, PAR)[subhInds]
        # Determine appropriate data type, accomodate varying shapes of properties
        temp = tempProps[0]
        if( np.iterable(temp) ): useType = type(temp[0])
        else:                    useType = type(temp)

        # Determine appropriate shape
        useShape = np.shape(tempProps)

        # Initialize storage array to null values ('-1') using appropriate type and shape
        useProps = -1*np.ones(useShape, dtype=useType)

        # Fill array with valid entries
        valid = np.where( subhInds >= 0 )[0]
        if( len(valid) > 0 ): useProps[valid] = tempProps[valid]

        # Store array to dictionary
        subhaloProps[PAR] = useProps


    # Check the matches based on BH ID numbers, masses, and positions
    if( check ): _checkHostMatches(run, snap, groupCat, bhIDs, bhInds, subhInds)

    return subhaloProps

# _getSnapshotSubhaloProperties()



def _checkHostMatches(run, snap, groupCat, bhIDs, bhInds, subhInds):

    # Mass Comparison Tolerances
    REL_TOL = 1.0e-2                              # Relative Tolerance
    ABS_TOL = 0.0                                 # Absolute Tolerance (0.0 means no affect)
    FIELDS = ['id', 'pos', 'BH_Mass']             # Retrieve parameters from Snapshot file

    print " - - SubhalosOfMergers._checkHostMatches()"

    # Select only valid indices (successful matches are >= 0)
    valid = np.where( (bhInds >= 0) & (subhInds >= 0) )[0]
    useID = bhIDs[valid]
    useBH = bhInds[valid]
    useSH = subhInds[valid]
    numInds = len(bhInds)
    numValid = len(valid)
    fracValid = 1.0*numValid/numInds
    uniqueSH = np.unique(useSH)
    numUnique = len(uniqueSH)

    print " - - - %d Indices.  %d Valid (%.4f).  %d Unique Subhalos" % \
        (numInds, numValid, fracValid, numUnique)


    ### Load Parameters ###

    # Load Snapshot
    snapshotPath = GET_ILLUSTRIS_SNAPSHOT_FIRST_FILENAME(run, snap)
    print " - - - Loading Snapshot '%s'" % (snapshotPath)
    snapshot = arepo.Snapshot(snapshotPath, parttype=[PARTICLE_TYPE_BH], fields=FIELDS,
                              combineFiles=True, verbose=False)

    # Get subhalo properties from Group Catalog
    shPos          = groupCat.SubhaloPos[useSH]
    shMass         = groupCat.SubhaloBHMass[useSH]
    shMultiplicity = groupCat.SubhaloLenType[useSH,PARTICLE_TYPE_BH]
    shRads         = groupCat.SubhaloHalfmassRad[useSH]

    shSingles   = np.where( shMultiplicity == 1 )[0]
    shMultiples = np.where( shMultiplicity > 1 )[0]
    numSingles  = len(shSingles)
    print " - - - Out of %d Subhalos associations, %d with single BH" % (numValid, numSingles)

    # Get BH Particle Properteis from Snapshot
    bhPos   = snapshot.pos[useBH]
    bhMass  = snapshot.BH_Mass[useBH]
    snapIDs = snapshot.id[useBH]


    ### Perform Comparisons ###

    # Compare IDs
    compIDs    = np.array( useID == snapIDs )
    goodIDs    = np.where( compIDs )[0]
    badIDs     = np.where( compIDs == False )[0]
    numGoodIDs = len(goodIDs)
    print " - - - %d/%d ID Matches" % (numGoodIDs, numValid)

    # Compare Masses
    compMass    = np.isclose( shMass, bhMass, atol=ABS_TOL, rtol=REL_TOL )
    goodMass    = np.where( compMass )[0]
    badMass     = np.where( compMass == False )[0]
    numGoodMass = len(goodMass)
    print " - - - %d/%d Mass Matches (relative tolerance %e)" % (numGoodMass, numValid, REL_TOL)

    # Compare Positions
    dists        = np.sqrt( np.sum( np.square(shPos-bhPos), axis=1 ) )
    distFracs    = dists/shRads
    goodDists    = np.where( distFracs <= 1.0 )[0]
    badDists     = np.where( distFracs >  1.0 )[0]
    numGoodDists = len(goodDists)
    print " - - - %d/%d Pos  Matches.  Ave = %.2e, Median = %.2e  [units of half-mass rad]" % \
        (numGoodDists, numValid, np.average(distFracs), np.median(distFracs))


    nums = len(shMultiples)
    nums = 10 if nums >= 10 else nums
    print " - - - First multiples locations : ", shMultiples[:nums]

    nums = len(badIDs)
    nums = 10 if nums >= 10 else nums
    print " - - - First Bad ID    locations : ", badIDs[:nums]

    nums = len(badMass)
    nums = 10 if nums >= 10 else nums
    print " - - - First bad mass  locations : ", badMass[:nums]

    nums = len(badDists)
    nums = 10 if nums >= 10 else nums
    print " - - - First bad dist  locations : ", badDists[:nums]


    return

# checkHostMatches()


