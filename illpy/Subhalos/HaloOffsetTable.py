"""
Manage table of particle offsets for associating particles with halos and subhalos.

The table is in the form of a dictionary with keys given by the values of the ``OFFTAB`` class.
The method ``loadOffsetTable()`` is the only necessary API - it deals with constructing, saving, 
and loading the offset table.

Classes
-------
    OFFTAB : enumerator-like class for dictionary key-words


Functions
---------
    loadOffsetTable            : load offset table for target run and snapshot

    _GET_OFFSET_TABLE_FILENAME : filename which the offset table is saved/loaded to/from
    _constructOffsetTable      : construct the offset table from the group catalog
    _constructBHIndexTable     : construct mapping from BH IDs to indices in snapshot files


Notes
-----
    The structure of the table is 3 different arrays with corresponding entries.
    ``halos``     (``OFFTAB.HALOS``)    : <int>[N],   halo number
    ``subhalos``  (``OFFTAB.SUBHALOS``) : <int>[N],   subhalo number
    ``particles`` (``OFFTAB.OFFSETS``)  : <int>[N,6], particle offsets for each halo/subhalo

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
import progressbar

from .. Constants import DTYPE, GET_ILLUSTRIS_OUTPUT_DIR, PARTICLE_NUM, GET_PROCESSED_DIR, PARTICLE
from Constants import SNAPSHOT

import zcode.InOut as zio

import illustris_python as ill

_VERSION = 0.4


class OFFTAB():
    """ Keys for offset table dictionary. """

    RUN         = 'run'
    SNAP        = 'snapshot'
    VERSION     = 'version'
    CREATED     = 'created'
    FILENAME    = 'filename'

    HALOS       = 'halo_numbers'
    SUBHALOS    = 'subhalo_numbers'
    OFFSETS     = 'particle_offsets'

    BH_IDS      = 'bh_ids'
    BH_INDICES  = 'bh_indices'
    BH_HALOS    = 'bh_halos'
    BH_SUBHALOS = 'bh_subhalos'



_OFFSET_TABLE_FILENAME_BASE = "offsets/ill%d_snap%d_offset-table_v%.2f.npz"

def _GET_OFFSET_TABLE_FILENAME(run, snap, version=_VERSION):
    fname  = GET_PROCESSED_DIR(run)
    fname += _OFFSET_TABLE_FILENAME_BASE % (run, snap, version)
    return fname




def loadOffsetTable(run, snap, loadsave=True, verbose=True):
    """
    Load pre-existing, or manage the creation of the particle offset table.

    Arguments
    ---------
       run      <int>  : illustris simulation number {1,3}
       snap     <int>  : illustris snapshot number {1,135}
       loadsave <bool> : optional, load existing table
       verbose  <bool> : optional, print verbose output

    Returns
    -------
       offsetTable <dict> : particle offset table, see ``HaloOffsetTable`` docs for more info.

    """

    if( verbose ): print " - - HaloOffsetTable.loadOffsetTable()"

    saveFile = _GET_OFFSET_TABLE_FILENAME(run, snap)

    ## Load Existing Save
    #  ------------------
    if( loadsave ):
        if( verbose ): print " - - - Loading from save '%s'" % (saveFile)
        # Make sure path exists
        if( os.path.exists(saveFile) ):
            offsetTable = zio.npzToDict(saveFile)
            if( verbose ): print " - - - - Table loaded"
        else:
            print "HaloOffsetTable.loadOffsetTable() : File does not Exist!"
            print "HaloOffsetTable.loadOffsetTable() : Reconstructing offsets !!"
            loadsave = False


    ## Reconstruct Offset Table
    #  ------------------------
    if( not loadsave ):
        if( verbose ): print " - - - Constructing Offset Table"
        start = datetime.now()

        # Construct Offset Data
        haloNums, subhNums, offsets = _constructOffsetTable(run, snap, verbose=verbose)

        # Construct BH index Data
        bhInds, bhIDs = _constructBHIndexTable(run, snap, verbose=verbose)

        # Find BH Subhalos
        binInds = np.digitize(bhInds, offsets[:,PARTICLE.BH]).astype(DTYPE.INDEX)-1
        if( any(binInds < 0) ): raise RuntimeError("Some bhInds not matched!! '%s'" % (str(bads)))
        bhHalos = haloNums[binInds]
        bhSubhs = subhNums[binInds]


        offsetTable = {}

        # Metadata
        offsetTable[OFFTAB.RUN]         = run
        offsetTable[OFFTAB.SNAP]        = snap
        offsetTable[OFFTAB.VERSION]     = _VERSION
        offsetTable[OFFTAB.CREATED]     = datetime.now().ctime()
        offsetTable[OFFTAB.FILENAME]    = saveFile

        # Offsets table data
        offsetTable[OFFTAB.HALOS]       = haloNums
        offsetTable[OFFTAB.SUBHALOS]    = subhNums
        offsetTable[OFFTAB.OFFSETS]     = offsets

        # BH Specific data
        offsetTable[OFFTAB.BH_INDICES]  = bhInds
        offsetTable[OFFTAB.BH_IDS]      = bhIDs
        offsetTable[OFFTAB.BH_HALOS]    = bhHalos
        offsetTable[OFFTAB.BH_SUBHALOS] = bhSubhs

        # Save to file
        zio.dictToNPZ(offsetTable, saveFile, verbose=verbose)

        stop = datetime.now()
        if( verbose ): print " - - - - Done after %s" % (str(stop-start))


    return offsetTable

# loadOffsetTable()



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

    # Set Progress Bar Parameters
    widgets = [
        progressbar.Percentage(),
        ' ', progressbar.Bar(),
        ' ', progressbar.AdaptiveETA(),
        ]

    # Start Progress Bar
    with progressbar.ProgressBar(widgets=widgets, maxval=tableSize, term_width=100) as pbar:


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
                pbar.update(offs)

            # } for jj


            # Add Entry for particles with NO subhalo
            haloNum[offs] = ii                        # Still part of halo ``ii``
            subhNum[offs] = -1                        # `-1` means no (sub)halo
            offsets[offs,:] = cumSubhParts

            # Increment particle numbers to include this halo
            cumSubhParts = np.copy(cumHaloParts)

            # Increment entry number
            offs += 1
            pbar.update(offs)

        # } for ii


        # Add entry for particles with NO halo and NO subhalo
        haloNum[offs] = -1
        subhNum[offs] = -1
        offsets[offs,:] = cumSubhParts

    # } with pbar

    return haloNum, subhNum, offsets

# _constructOffsetTable()



def _constructBHIndexTable(run, snap, verbose=True):
    """
    Load all BH ID numbers and associate them with 'index' (i.e. order) numbers.

    Arguments
    ---------
       run     <int>  : illustris simulation number {1,3}
       snap    <int>  : illustris snapshot number {1,135}
       verbose <bool> : optional, print verbose output

    Returns
    -------
       inds    <int>[N] : BH Index numbers
       bhIDs   <int>[N] : BH particle ID numbers

    """

    if( verbose ): print " - - HaloOffsetTable._constructBHIndexTable()"

    # Illustris Data Directory where catalogs are stored
    illpath = GET_ILLUSTRIS_OUTPUT_DIR(run)

    # Load all BH ID numbers from snapshot (single ``fields`` parameters loads array, not dict)
    if( verbose ): print " - - - Loading BHs from Snapshot %d in '%s'" % (snap, illpath)
    bhIDs = ill.snapshot.loadSubset(illpath, snap, PARTICLE.BH, fields=SNAPSHOT.IDS)
    numBHs = len(bhIDs)
    if( verbose ): print " - - - - BHs Loaded (%7d)" % (numBHs)

    # Create 'indices' of BHs
    inds = np.arange(numBHs)

    return inds, bhIDs

# _constructBHIndexTable()


