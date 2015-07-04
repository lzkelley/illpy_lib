"""
Manage table of particle offsets for associating particles with halos and subhalos.

The table is in the form of a dictionary with keys given by the constants ``OFFSET_``.  The method
``loadOffsetTable()`` is the only necessary API - it deals with constructing, saving, and loading
the offset tables.


Methods
-------
  API
  - loadOffsetTable()                 : load offset table for target run and snapshot
  - getBHSubhaloIndices()             : get the subhalo indices for given blackhole IDs

  Internal
  - _GET_OFFSET_TABLE_SAVE_FILENAME() : filename which the offset table is saved/loaded to/from
  - _getHostsFromParticleIndex()      : Given a BH index, find its host halo and subhalo
  - _bhIDsToIndices()                 : convert from BH ID number to Index in snapshot files
  - _constructOffsetTable()           : construct the offset table from the group catalog
  - _constructIndexTable()            : construct mapping from BH IDs to indices in snapshot files


Notes
-----
The structure of the table is 3 different arrays with corresponding entries.
``halos``     (``OFFSET_HALOS``)     : <int64>[N],   halo number
``subhalos``  (``OFFSET_SUBHALOS``)  : <int64>[N],   subhalo number
``particles`` (``OFFSET_PARTICLES``) : <int64>[N,6], particle offsets for each halo/subhalo

The table is ordered in the same way as the snapshots, where particles are grouped into subhalos,
which belong to halos.  Each halo also has (or can have) a group of particles not in any subhalo.
Finally, the last entry is for particles with no halo and no subhalo.  When there is no match for
a subhalo or halo, the corresponding number is listed as '-1'.

For a halo 'i', with NS_i subhalos, there are NS_i+1 entries for that halo.
If the total number of subhalos is NS = SUM_i( NS_i ), and there are
NH halos, then the total number of entries is NS + NH + 1.

This is what the table looks like (using made-up numbers):

                        PARTICLES {0,5}
    HALO    SUBHALO     0     1  ...   5
  | ====================================
  |    0          0     0     0  ...   0  <-- halo-0, subhalo-0, no previous particles
  |    0          1    10     4  ...   1  <--  first part0 for this subhalo is 10th part0 overall
  |    0          2    18     7  ...   3  <--  first part1 for this subhalo is  7th part1 overall
  |                              ...
  |    0         -1   130    58  ...  33  <-- particles of halo-0, no subhalo
  |
  |    1         22   137    60  ...  35  <-- In halo-0 there were 22 subhalos and 137 part0, etc
  |    1         23              ...
  |                              ...
  |                              ...
  |   -1         -1  2020   988  ... 400
  | ====================================

Thus, given a particle5 of index 35, we know that that particle belongs to Halo-1, Subhalo-22.
Alternatively, a particle0 index of 134 belongs to halo-0, and has no subhalo.
Finally, any Particle1's with index 988 or after belong to no subhalo, and no halo.

"""

import os
import sys
import numpy as np
from datetime import datetime

import illpy
from illpy.Constants import *
from illpy.illbh.BHConstants import *
from illpy import AuxFuncs as aux

import arepo

# Max number of particles is 1820^3 which is larger than max of uint32,
#     but smaller than max int64... so might as well used signed int
#     and be able to use negative numbers for flags
TYPE_INDEX = np.int64

# ID numbers must be unsigned 64 to accamadate Illustirs-1
TYPE_IDNUM = np.uint64


VERSION = 0.2
VERBOSE = True


OFFSET_RUN        = 'run'
OFFSET_SNAP       = 'snapshot'
OFFSET_VERSION    = 'version'
OFFSET_CREATED    = 'created'
OFFSET_FILENAME   = 'filename'

OFFSET_HALOS      = 'halo_numbers'
OFFSET_SUBHALOS   = 'subhalo_numbers'
OFFSET_PARTICLES  = 'particle_numbers'

OFFSET_BH_IDS     = 'bh_ids'
OFFSET_BH_INDICES = 'bh_indices'


_OFFSET_TABLE_FILENAME_BASE = "offsets/ill-%d_snap-%d_offset-table.npz"





def loadOffsetTable(run, snap, loadsave=True, verbose=VERBOSE):
    """

    """

    if( verbose ): print " - - HaloOffsetTable.loadOffsetTable()"

    saveFile = _GET_OFFSET_TABLE_SAVE_FILENAME(run, snap)

    ### Load Existing Save ###
    if( loadsave ):
        if( verbose ): print " - - - Loading from save '%s'" % (saveFile)
        # Make sure path exists
        if( os.path.exists(saveFile) ):
            offsetTable = aux.npzToDict(saveFile)
            loadVers = offsetTable[OFFSET_VERSION]
            # Make sure version matches
            if( loadVers != VERSION ):
                print "HaloOffsetTable.loadOffsetTable() : Loaded  version %s" % (str(loadVers))
                print "HaloOffsetTable.loadOffsetTable() : Current version %s" % (str(VERSION ))
                print "HaloOffsetTable.loadOffsetTable() : Reconstructing offsets !!"
                loadsave = False
            else:
                if( verbose ): print " - - - - Loaded"

        else:
            print "HaloOffsetTable.loadOffsetTable() : File does not Exist!"
            print "HaloOffsetTable.loadOffsetTable() : Reconstructing offsets !!"
            loadsave = False


    ### Reconstruct Offset Table ###
    if( not loadsave ):
        if( verbose ): print " - - - Constructing Offset Table"
        start = datetime.now()

        # Construct Offset Data
        haloNums, subhNums, offsets = _constructOffsetTable(run, snap, verbose=verbose)

        # Construct index Data
        bhInds, bhIDs = _constructIndexTable(run, snap, verbose=verbose)


        offsetTable = {}

        # Store data
        offsetTable[OFFSET_HALOS]      = haloNums
        offsetTable[OFFSET_SUBHALOS]   = subhNums
        offsetTable[OFFSET_PARTICLES]  = offsets
        offsetTable[OFFSET_BH_INDICES] = bhInds
        offsetTable[OFFSET_BH_IDS]     = bhIDs

        # Add Metadata
        offsetTable[OFFSET_RUN]        = run
        offsetTable[OFFSET_SNAP]       = snap
        offsetTable[OFFSET_VERSION]    = VERSION
        offsetTable[OFFSET_CREATED]    = datetime.now().ctime()
        offsetTable[OFFSET_FILENAME]   = saveFile

        # Save
        aux.dictToNPZ(offsetTable, saveFile)

        stop = datetime.now()
        if( verbose ): print " - - - - Done after %s" % (str(stop-start))


    return offsetTable

# loadOffsetTable()



def getBHSubhaloIndices(run, snap, bhIDs, verbose=VERBOSE):
    """
    Retrieve the subhalo indices which host the given Blackhole ID numbers.

    BHs who were not matched in the snapshot return indices of '-1', which can happen if
    they are the 'in'-BH of a later Merger before the snapshot is written (for example).
    The Subhalo indices corresponding to such missing BHs are also '-1'.

    Arguments
    ---------
    run      : <int>,     illustris simulation number {1,3}
    snap     : <int>,     illustris snapshot   number {0,135}
    bhIDs    : <long>[N], Blackhole ID numbers
    verbose  : <bool>,    (optional=VERBOSE) print verbose output

    Returns
    -------
    bhInds   : <long>[N], array of Blackhole indices (in the snapshot files)
    subhInds : <long>[N], array of Subhalo   indices (in the snapshot/groupfind files)

    """

    
    if( verbose ): print " - - HaloOffsetTable.getBHSubhaloIndices()"

    # Load Offset Table
    if( verbose ): print " - - - Loading offset table"
    offsetTable = loadOffsetTable(run, snap, verbose=verbose)

    # Convert from ID numbers to indices in snapshot files
    if( verbose ): print " - - - Converting from particle IDs to particle Indices"
    bhInds = _bhIDsToIndices(bhIDs, offsetTable, verbose=verbose)

    # Get Hosts
    haloHosts, subhHosts = _getHostsFromParticleIndex(run, snap, bhInds, verbose=verbose)

    return bhInds, subhHosts

# getBHSubhaloIndices()




def _getHostsFromParticleIndex(run, snap, pind, verbose=VERBOSE):
    '''
    Given the index of a particle, find the indices of the halo and subhalo it belongs to.

    This method uses the offset tables calculated in an arepo.Subfind object to reverse out its
    parent (FOF)halo(/group) and subhalo.

    '''

    PTYPE = PARTICLE_TYPE_BH

    if( verbose ): print " - - HaloOffsetTable._getHostsFromParticleIndex()"

    # Convert input indices to np.array
    if( np.iterable(pind) ): bhInds = np.array( pind )
    else:                    bhInds = np.array([pind])

    if( verbose ):  print " - - - Searching for %d indices" % (len(bhInds))


    # Load offset table
    if( verbose ): print " - - - Loading offset table"
    offsetTable = loadOffsetTable(run, snap, verbose=verbose)
    haloNums = offsetTable[OFFSET_HALOS]
    subhNums = offsetTable[OFFSET_SUBHALOS]
    partNums = offsetTable[OFFSET_PARTICLES][:,PTYPE]
    if( verbose ): print " - - - - Loaded %d entries" % (len(haloNums))


    ### Find Host Halos and Subhalos ###

    if( verbose ): print " - - - Finding halo bins"
    # Find the entry for each index, np.digitize assigns values to right bin, shift left with '-1'
    binInds = np.digitize(bhInds, partNums).astype(TYPE_INDEX)-1

    hostHalos = haloNums[binInds]
    hostSubhs = subhNums[binInds]

    # Some `bhInds` are not-found (equal '-1'), set results to also be null ('-1')
    bads = np.where( binInds < 0 )[0]
    if( len(bads) > 0 ):
        hostHalos[bads] = -1
        hostSubhs[bads] = -1


    return hostHalos, hostSubhs

# _getHostsFromParticleIndex()





def _bhIDsToIndices(inIDs, table, verbose=VERBOSE):
    """
    Convert from blackhole ID numbers to indexes using an offset table dictionary.
    
    Notes
    -----
    Not all BH IDs will be found.  Incorrect/missing matches return '-1' elements

    """

    if( verbose ): print " - - HaloOffsetTable._idsToIndices()"

    outIDs  = table[OFFSET_BH_IDS]
    outInds = table[OFFSET_BH_INDICES]

    ### Find Indices of Matches ###

    # Sort IDs for faster searching
    sortIDs = np.argsort(outIDs)
    # Find matches in sorted array
    foundSorted = np.searchsorted(outIDs, inIDs, sorter=sortIDs)
    # Reverse map to find matches in original array
    found = sortIDs[foundSorted]



    foundIDs  = outIDs[found]
    foundInds = outInds[found]

    
    ### Check Matches ###

    # Find incorrect matches 
    inds = np.where( inIDs != foundIDs )[0]
    numBad = len(inds)
    numGood = len(inIDs)-numBad
    if( verbose ): print " - - - %d Good, %d Bad Matches" % (numGood, numBad)
    # Set incorrect matches to '-1'
    if( len(inds) > 0 ):
        foundIDs[inds]  = -1
        foundInds[inds] = -1

    
    return foundInds

# _bhIDsToIndices()


def _constructOffsetTable(run, snap, verbose=VERBOSE):
    """
    Construct offset table from a group catalog
    
    Arguments
    ---------
    run     : <int>, illustris simulation number {1,3}
    snap    : <int>, illustris snapshot number {0,135}
    verbose : <bool>, optional=VERBOSE, print verbose output

    Returns
    -------
    haloNum : <long>[N],   halo      number for each offset entry
    subhNum : <long>[N],   subhalo   number for each offset entry
    offsets : <long>[N,6], particle offsets for each offset entry

    """

    if( verbose ): print " - - HaloOffsetTable._constructOffsetTable()"

    ### Load Group Catalog ###
    groupCatFilename = GET_ILLUSTRIS_GROUPS_FIRST_FILENAME(run, snap)
    if( verbose ): print " - - - Loading group catalog '%s'" % (groupCatFilename)
    start = datetime.now()
    groupCat = arepo.Subfind(groupCatFilename, combineFiles=True)
    stop  = datetime.now()
    if( verbose ): print " - - - - Done after %s" % (str(stop-start))

    numHalos    = groupCat.npart_loaded[0]
    numSubhalos = groupCat.npart_loaded[1]
    tableSize   = numHalos + numSubhalos + 1
    if( verbose ):
        print " - - - - %d Halos, %d Subhalos" % (numHalos, numSubhalos)
        print " - - - - %d Offset entries" % (tableSize)


    if( numHalos < 200 ): interval = 1
    else:                 interval = np.int(np.floor( numHalos/200.0 ))


    ### Initialize ###

    # See object description; recall entries are [HALO, SUBHALO, PART0, ... PART5]
    #    offsetTable = -1*np.ones( [numHalos+numSubhalos+1, 8], dtype=TYPE_INDEX)
    haloNum = -1*np.ones(tableSize, dtype=TYPE_INDEX)
    subhNum = -1*np.ones(tableSize, dtype=TYPE_INDEX)
    offsets = -1*np.ones([tableSize,PARTICLE_TYPE_NUM], dtype=TYPE_INDEX)

    subh = 0
    offs = 0
    cumHaloParts = np.zeros(6, dtype=TYPE_INDEX)
    cumSubhParts = np.zeros(6, dtype=TYPE_INDEX)

    # Iterate over each Halo
    start = datetime.now()
    for ii in xrange(numHalos):

        # Add the number of particles in this halo
        cumHaloParts += groupCat.group.GroupLenType[ii,:]

        # Iterate over each Subhalo in halo 'ii'
        for jj in range(groupCat.group.GroupNsubs[ii]):
            # Add entry for each subhalo
            #     offsetTable[offs] = np.append([ ii, subh ], cumSubhParts)
            haloNum[offs] = ii
            subhNum[offs] = subh
            offsets[offs,:] = cumSubhParts

            # Increment
            #   Add particles in this subhalo
            cumSubhParts += groupCat.subhalo.SubhaloLenType[subh]
            subh += 1
            offs += 1


        # Add Entry for particles with NO subhalo
        #      offsetTable[offs] = np.append([ii,-1], cumSubhParts)
        haloNum[offs] = ii
        subhNum[offs] = -1
        offsets[offs,:] = cumSubhParts

        # Increment particle numbers to include this halo
        cumSubhParts = np.copy(cumHaloParts)
        offs += 1

        if( verbose and ii%interval == 0 ):
            # Find out current duration
            now = datetime.now()
            dur = now-start

            # Print status and time to completion
            statStr = aux.statusString(ii+1, numHalos, dur)
            sys.stdout.write('\r - - - - %s' % (statStr))
            sys.stdout.flush()



    # Add entry for end of all halo particles / start of particles with NO halo
    #    offsetTable[offs] = np.append([-1,-1], cumSubhParts)
    haloNum[offs] = -1
    subhNum[offs] = -1
    offsets[offs,:] = cumSubhParts

    if( verbose ): sys.stdout.write('\n')

    return haloNum, subhNum, offsets

# _constructOffsetTable()


def _constructIndexTable(run, snap, verbose=VERBOSE):
    
    if( verbose ): print " - - HaloOffsetTable._constructIndexTable()"

    # Load Snapshot
    snapshotPath = GET_ILLUSTRIS_SNAPSHOT_FIRST_FILENAME(run, snap)
    if( verbose ): print " - - - Loading snapshot '%s'" % (snapshotPath)
    start = datetime.now()
    snap  = arepo.Snapshot(snapshotPath, fields=['id'], parttype=PARTICLE_TYPE_BH, 
                           combineFiles=True, verbose=False)
    stop  = datetime.now()

    ids  = np.array(snap.id)
    nums = len(ids)
    inds = np.arange(nums)
    if( verbose ): print " - - - - Loaded %d particles after %s" % (nums, str(stop-start))

    return inds, ids


# _constructIndexTable()



def _GET_OFFSET_TABLE_SAVE_FILENAME(run, snap):
    fname  = GET_PROCESSED_DIR(run)
    fname += _OFFSET_TABLE_FILENAME_BASE % (run, snap)
    return fname


