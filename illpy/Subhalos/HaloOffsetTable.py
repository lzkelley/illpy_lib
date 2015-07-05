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
import numpy as np
from datetime import datetime

from illpy.Constants import DTYPE, GET_ILLUSTRIS_OUTPUT_DIR, PARTICLE_NUM
#from illpy.illbh.BHConstants import *

import zcode.InOut as zio

import illustris_python as ill

_VERSION = 0.2


class OFFSET():
    """ Keys for offset table dictionary. """

    RUN        = 'run'
    SNAP       = 'snapshot'
    VERSION    = 'version'
    CREATED    = 'created'
    FILENAME   = 'filename'

    HALOS      = 'halo_numbers'
    SUBHALOS   = 'subhalo_numbers'
    PARTICLES  = 'particle_numbers'

    BH_IDS     = 'bh_ids'
    BH_INDICES = 'bh_indices'



_OFFSET_TABLE_FILENAME_BASE = "offsets/ill-%d_snap-%d_offset-table.npz"

def GET_OFFSET_TABLE_SAVE_FILENAME(run, snap, version=_VERSION):
    fname  = GET_PROCESSED_DIR(run)
    fname += _OFFSET_TABLE_FILENAME_BASE % (run, snap)
    return fname




def loadOffsetTable(run, snap, loadsave=True, verbose=True):
    """

    """

    if( verbose ): print " - - HaloOffsetTable.loadOffsetTable()"

    saveFile = GET_OFFSET_TABLE_SAVE_FILENAME(run, snap)

    ### Load Existing Save ###
    if( loadsave ):
        if( verbose ): print " - - - Loading from save '%s'" % (saveFile)
        # Make sure path exists
        if( os.path.exists(saveFile) ):
            offsetTable = zio.npzToDict(saveFile)
            loadVers = offsetTable[OFFSET.VERSION]
            # Make sure version matches
            if( loadVers != _VERSION ):
                print "HaloOffsetTable.loadOffsetTable() : Loaded  version %s" % (str(loadVers))
                print "HaloOffsetTable.loadOffsetTable() : Current version %s" % (str(_VERSION ))
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
        offsetTable[OFFSET.HALOS]      = haloNums
        offsetTable[OFFSET.SUBHALOS]   = subhNums
        offsetTable[OFFSET.PARTICLES]  = offsets
        offsetTable[OFFSET.BH_INDICES] = bhInds
        offsetTable[OFFSET.BH_IDS]     = bhIDs

        # Add Metadata
        offsetTable[OFFSET.RUN]        = run
        offsetTable[OFFSET.SNAP]       = snap
        offsetTable[OFFSET.VERSION]    = _VERSION
        offsetTable[OFFSET.CREATED]    = datetime.now().ctime()
        offsetTable[OFFSET.FILENAME]   = saveFile

        # Save
        zio.dictToNPZ(offsetTable, saveFile)

        stop = datetime.now()
        if( verbose ): print " - - - - Done after %s" % (str(stop-start))


    return offsetTable

# loadOffsetTable()



def getBHSubhaloIndices(run, snap, bhIDs, verbose=True):
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




def _getHostsFromParticleIndex(run, snap, pind, verbose=True):
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
    haloNums = offsetTable[OFFSET.HALOS]
    subhNums = offsetTable[OFFSET.SUBHALOS]
    partNums = offsetTable[OFFSET.PARTICLES][:,PTYPE]
    if( verbose ): print " - - - - Loaded %d entries" % (len(haloNums))


    ### Find Host Halos and Subhalos ###

    if( verbose ): print " - - - Finding halo bins"
    # Find the entry for each index, np.digitize assigns values to right bin, shift left with '-1'
    binInds = np.digitize(bhInds, partNums).astype(DTYPE.INDEX)-1

    hostHalos = haloNums[binInds]
    hostSubhs = subhNums[binInds]

    # Some `bhInds` are not-found (equal '-1'), set results to also be null ('-1')
    bads = np.where( binInds < 0 )[0]
    if( len(bads) > 0 ):
        hostHalos[bads] = -1
        hostSubhs[bads] = -1


    return hostHalos, hostSubhs

# _getHostsFromParticleIndex()





def _bhIDsToIndices(inIDs, table, verbose=True):
    """
    Convert from blackhole ID numbers to indexes using an offset table dictionary.
    
    Notes
    -----
    Not all BH IDs will be found.  Incorrect/missing matches return '-1' elements

    """

    if( verbose ): print " - - HaloOffsetTable._idsToIndices()"

    outIDs  = table[OFFSET.BH_IDS]
    outInds = table[OFFSET.BH_INDICES]

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


def _constructOffsetTable(run, snap, verbose=True):
    """
    Construct offset table from halo and subhalo catalogs.

    Each 'entry' is the first particle index number for a group of particles.  Particles are
    grouped by the halos and subhalos they belong to.  The first entry is particles in the first
    subhalo of the first halo.  The last entry for the first halo is particles that dont belong to
    any subhalo (but still belong to the first halo).  The very last entry is for particles that
    dont belong to any halo or subhalo.
    
    Arguments
    ---------
       run     <int>  : illustris simulation number {1,3}
       snap    <int>  : illustris snapshot number {0,135}
       verbose <bool> : optional, print verbose output

    Returns
    -------
       haloNum <int>[N]   : halo      number for each offset entry
       subhNum <int>[N]   : subhalo   number for each offset entry
       offsets <int>[N,6] : particle offsets for each offset entry

    """

    if( verbose ): print " - - HaloOffsetTable._constructOffsetTable()"

    ## Load (Sub)Halo Catalogs
    #  -----------------------

    # Illustris Data Directory where catalogs are stored
    illpath = GET_ILLUSTRIS_OUTPUT_DIR(run)
    
    if( verbose ): print " - - - Loading Catalogs from '%s'" % (illpath)
    haloCat = ill.groupcat.loadHalos(illpath, snap, fields=None)
    numHalos    = haloCat['count']
    if( verbose ): print " - - - - Halos    Loaded (%7d)" % (numHalos)
    subhCat = ill.groupcat.loadSubhalos(illpath, snap, fields=None)
    numSubhs = subhCat['count']
    if( verbose ): print " - - - - Subhalos Loaded (%7d)" % (numSubhs)


    ## Initialize Storage
    #  ------------------

    tableSize = numHalos + numSubhs + 1

    # See object description; recall entries are [HALO, SUBHALO, PART0, ... PART5]
    #    (Sub)halo numbers are smaller, use signed-integers for `-1` to be no (Sub)halo
    haloNum = np.zeros( tableSize,               dtype=DTYPE.INDEX)
    subhNum = np.zeros( tableSize,               dtype=DTYPE.INDEX)
    # Offsets approach total number of particles, must be uint64
    offsets = np.zeros([tableSize,PARTICLE_NUM], dtype=DTYPE.ID)

    subh = 0
    offs = 0
    cumHaloParts = np.zeros(PARTICLE_NUM, dtype=DTYPE.ID)
    cumSubhParts = np.zeros(PARTICLE_NUM, dtype=DTYPE.ID)


    ## Iterate Over Each Halo
    #  ----------------------
    for ii in xrange(numHalos):

        # Add the number of particles in this halo
        cumHaloParts[:] += haloCat['GroupLenType'][ii,:]

        
        ## Iterate over each Subhalo, in halo ``ii``
        #  -----------------------------------------
        for jj in xrange(haloCat['GroupNsubs'][ii]):

            # Consistency check: make sure subhalo number is as expected
            if( jj == 0 and subh != haloCat['GroupFirstSub'][ii] ):
                print "ii = %d, jj = %d, subh = %d" % (ii, jj, subh)
                raise RuntimeError("Subhalo iterator doesn't match Halo's first subhalo!")

            # Add entry for each subhalo
            haloNum[offs] = ii
            subhNum[offs] = subh
            offsets[offs,:] = cumSubhParts

            # Add particles in this subhalo to offset counts
            cumSubhParts[:] += subhCat['SubhaloLenType'][subh,:]

            # Increment subhalo and entry number
            subh += 1
            offs += 1

        # } for jj


        # Add Entry for particles with NO subhalo
        haloNum[offs] = ii                        # Still part of halo ``ii``
        subhNum[offs] = -1                        # `-1` means no (sub)halo
        offsets[offs,:] = cumSubhParts

        # Increment particle numbers to include this halo
        cumSubhParts = np.copy(cumHaloParts)

        # Increment entry number
        offs += 1

    # } for ii


    # Add entry for particles with NO halo and NO subhalo
    haloNum[offs] = -1
    subhNum[offs] = -1
    offsets[offs,:] = cumSubhParts

    return haloNum, subhNum, offsets

# _constructOffsetTable()



def _constructIndexTable(run, snap, verbose=True):
    
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





