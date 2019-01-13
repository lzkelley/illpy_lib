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
    # loadOffsetTable            : load offset table for target run and snapshot
    loadBHHostsSnap            : load (sub)halo host associations for blackholes in one snapshot
    loadBHHosts                : load (sub)halo host associations for blackholes in all snapshots
    main                       :
    subhalosForBHIDs           : find subhalos for given BH IDs

    _GET_OFFSET_TABLE_FILENAME : filename which the offset table is saved/loaded to/from

    _constructOffsetTable      : construct the offset table from the group catalog
    _constructBHIndexTable     : construct mapping from BH IDs to indices in snapshot files


Notes
-----
    The structure of the table is 3 different arrays with corresponding entries.
    ``halos``     (``OFFTAB.HALOS``)    : <int>[N],   halo number
    ``subhalos``  (``OFFTAB.SUBHALOS``) : <int>[N],   subhalo number
    ``particles`` (``OFFTAB.OFFSETS``)  : <int>[N, 6], particle offsets for each halo/subhalo

    The table is ordered in the same way as the snapshots, where particles are grouped into subhalos,
    which belong to halos.  Each halo also has (or can have) a group of particles not in any subhalo.
    Finally, the last entry is for particles with no halo and no subhalo.  When there is no match for
    a subhalo or halo, the corresponding number is listed as '-1'.

    For a halo 'i', with NS_i subhalos, there are NS_i+1 entries for that halo.
    If the total number of subhalos is NS = SUM_i(NS_i), and there are
    NH halos, then the total number of entries is NS + NH + 1.

    This is what the table looks like (using made-up numbers):

                            PARTICLES {0, 5}
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
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
from datetime import datetime

from illpy_lib.constants import (DTYPE, NUM_SNAPS, PARTICLE,
                                 GET_ILLUSTRIS_OUTPUT_DIR, GET_PROCESSED_DIR, GET_BAD_SNAPS)
from . Constants import SNAPSHOT

import zcode.inout as zio

_VERSION = 0.5


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

    @staticmethod
    def snapDictKey(snap):
        return "%03d" % (snap)


_OFFSET_TABLE_FILENAME_BASE = "offsets/ill%d_snap%d_offset-table_v%.2f.npz"


def _GET_OFFSET_TABLE_FILENAME(run, snap, version=None):
    if (version is None): version = _VERSION
    fname  = GET_PROCESSED_DIR(run)
    fname += _OFFSET_TABLE_FILENAME_BASE % (run, snap, version)
    return fname


_BH_HOSTS_SNAP_TABLE_FILENAME_BASE = "bh-hosts/ill%d_snap%03d_bh-hosts_v%.2f.npz"


def _GET_BH_HOSTS_SNAP_TABLE_FILENAME(run, snap, version=None):
    if (version is None): version = _VERSION
    fname  = GET_PROCESSED_DIR(run)
    fname += _BH_HOSTS_SNAP_TABLE_FILENAME_BASE % (run, snap, version)
    return fname


_BH_HOSTS_TABLE_FILENAME_BASE = "bh-hosts/ill%d_bh-hosts_v%.2f.npz"


def _GET_BH_HOSTS_TABLE_FILENAME(run, version=None):
    if (version is None): version = _VERSION
    fname  = GET_PROCESSED_DIR(run)
    fname += _BH_HOSTS_TABLE_FILENAME_BASE % (run, version)
    return fname


'''
def loadOffsetTable(run, snap, loadsave=True, verbose=True):
    """
    Load pre-existing, or manage the creation of the particle offset table.

    Arguments
    ---------
       run      <int>  : illustris simulation number {1, 3}
       snap     <int>  : illustris snapshot number {1, 135}
       loadsave <bool> : optional, load existing table
       verbose  <bool> : optional, print verbose output

    Returns
    -------
       offsetTable <dict> : particle offset table, see ``ParticleHosts`` docs for more info.

    """

    if verbose: print(" - - ParticleHosts.loadOffsetTable()")

    saveFile = _GET_OFFSET_TABLE_FILENAME(run, snap)

    # Load Existing Save
    #  ------------------
    if (loadsave):
        if verbose: print(" - - - Loading from save '{:s}'".format(saveFile))
        # Make sure path exists
        if (os.path.exists(saveFile)):
            offsetTable = zio.npzToDict(saveFile)
            if verbose: print(" - - - - Table loaded")
        else:
            if verbose: print(" - - - - File does not Exist, reconstructing offsets")
            loadsave = False


    # Reconstruct Offset Table
    #  ------------------------
    if (not loadsave):
        if verbose: print(" - - - Constructing Offset Table")
        start = datetime.now()

        # Construct Offset Data
        haloNums, subhNums, offsets = _constructOffsetTable(run, snap, verbose=verbose)

        # Construct BH index Data
        bhInds, bhIDs = _constructBHIndexTable(run, snap, verbose=verbose)

        # Find BH Subhalos
        binInds = np.digitize(bhInds, offsets[:, PARTICLE.BH]).astype(DTYPE.INDEX)-1
        if (any(binInds < 0)): raise RuntimeError("Some bhInds not matched!! '%s'" % (str(bads)))
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
        if verbose: print(" - - - - Done after {:s}".format(str(stop-start)))

    return offsetTable

# loadOffsetTable()
'''


def loadBHHosts(run, loadsave=True, version=None, verbose=True, bar=None, convert=None):
    """Merge individual snapshot's blackhole hosts files into a single file.

    Arguments
    ---------
    run      <int>  : illustris simulation number {1, 3}
    loadsave <bool> : optional, load existing save if possible
    version  <flt>  : optional, target version number
    verbose  <bool> : optional,
    bar      <bool> : optional,
    convert  <bool> : optional,

    Returns
    -------
    bhHosts <dict> : table of hosts for all snapshots

    """
    if verbose: print(" - - ParticleHosts.loadBHHosts()")
    if (bar is None): bar = bool(verbose)

    # Load Existing Save
    #  ==================
    if (loadsave):
        saveFile = _GET_BH_HOSTS_TABLE_FILENAME(run, version=version)

        if verbose: print((" - - - Loading from save '{:s}'".format(saveFile)))
        # Make sure path exists
        if (os.path.exists(saveFile)):
            bhHosts = zio.npzToDict(saveFile)
            if verbose: print(" - - - - Table loaded")
        else:
            if verbose: print(" - - - - File does not Exist, reconstructing BH Hosts")
            loadsave = False

    # Reconstruct Hosts Table
    # =======================
    if (not loadsave):

        if verbose: print(" - - - Constructing Hosts Table")
        start = datetime.now()

        if (version is not None): raise RuntimeError("Can only create version '%s'" % _VERSION)
        saveFile = _GET_BH_HOSTS_TABLE_FILENAME(run)

        # Create progress-bar
        pbar = zio.getProgressBar(NUM_SNAPS)
        if bar: pbar.start()

        # Select the dict-keys for snapshot hosts to transfer
        hostKeys = [OFFTAB.BH_IDS, OFFTAB.BH_INDICES, OFFTAB.BH_HALOS, OFFTAB.BH_SUBHALOS]

        # Create dictionary
        # -----------------
        bhHosts = {}

        # Add metadata
        bhHosts[OFFTAB.RUN] = run
        bhHosts[OFFTAB.VERSION] = _VERSION
        bhHosts[OFFTAB.CREATED] = datetime.now().ctime()
        bhHosts[OFFTAB.FILENAME] = saveFile

        # Load All BH-Hosts Files
        # -----------------------
        for snap in range(NUM_SNAPS):
            # Load Snapshot BH-Hosts
            hdict = loadBHHostsSnap(run, snap, loadsave=True, verbose=True, convert=convert)
            # Extract and store target data
            snapStr = OFFTAB.snapDictKey(snap)
            bhHosts[snapStr] = {hkey: hdict[hkey] for hkey in hostKeys}
            if bar: pbar.update(snap)

        if bar: pbar.finish()

        # Save to file
        zio.dictToNPZ(bhHosts, saveFile, verbose=verbose)

        stop = datetime.now()
        if verbose: print(" - - - - Done after %s" % (str(stop-start)))

    return bhHosts


def loadBHHostsSnap(run, snap, version=None, loadsave=True, verbose=True, bar=None, convert=None):
    """Load pre-existing, or manage the creation of the particle offset table.

    Arguments
    ---------
    run      <int>  : illustris simulation number {1, 3}
    snap     <int>  : illustris snapshot number {1, 135}
    loadsave <bool> : optional, load existing table
    verbose  <bool> : optional, print verbose output

    Returns
    -------
    offsetTable <dict> : particle offset table, see `ParticleHosts` docs for more info.

    """
    if verbose: print(" - - ParticleHosts.loadBHHostsSnap()")
    if (bar is None): bar = bool(verbose)

    # Load Existing Save
    # ==================
    if (loadsave):
        saveFile = _GET_BH_HOSTS_SNAP_TABLE_FILENAME(run, snap, version)

        if verbose: print((" - - - Loading from save '{:s}'".format(saveFile)))
        # Make sure path exists
        if (os.path.exists(saveFile)):
            hostTable = zio.npzToDict(saveFile)
            if verbose: print(" - - - - Table loaded")
        else:
            if verbose: print(" - - - - File does not Exist, reconstructing BH Hosts")
            loadsave = False

    # Reconstruct Hosts Table
    # =======================
    if (not loadsave):
        if verbose: print(" - - - Constructing Offset Table")
        start = datetime.now()

        if (version is not None): raise RuntimeError("Can only create version '%s'" % _VERSION)
        saveFile = _GET_BH_HOSTS_SNAP_TABLE_FILENAME(run, snap)

        offsetFile = ''
        if (convert is not None):
            offsetFile = _GET_OFFSET_TABLE_FILENAME(run, snap, version=convert)
            if verbose: print((" - - - Trying to convert from existing '{:s}'".format(offsetFile)))

        # Convert an Existing (Full) Offset Table into BH Hosts
        # -----------------------------------------------------
        if os.path.exists(offsetFile):
            offsetTable = zio.npzToDict(offsetFile)

            bhInds  = offsetTable[OFFTAB.BH_INDICES]
            bhIDs   = offsetTable[OFFTAB.BH_IDS]
            bhHalos = offsetTable[OFFTAB.BH_HALOS]
            bhSubhs = offsetTable[OFFTAB.BH_SUBHALOS]

        else:
            if verbose: print(" - - - Reconstructing offset table")

            # Construct Offset Data
            haloNums, subhNums, offsets = _constructOffsetTable(run, snap, verbose=verbose)

            # Construct BH index Data
            #     Catch errors for bad snapshots
            try:
                bhInds, bhIDs = _constructBHIndexTable(run, snap, verbose=verbose)
            except:
                # If this is a known bad snapshot, set values to None
                if (snap in GET_BAD_SNAPS(run)):
                    if verbose:
                        print((" - - - BAD SNAPSHOT: RUN {:d}, Snap {:d}".format(run, snap)))
                    bhInds  = None
                    bhIDs   = None
                    bhHalos = None
                    bhSubhs = None
                # If this is not a known problem, still raise error
                else:
                    print(("this is not a known bad snapshot: run {:d}, snap {:d}".format(
                        run, snap)))
                    raise

            # On success, Find BH Subhalos
            else:
                binInds = np.digitize(bhInds, offsets[:, PARTICLE.BH]).astype(DTYPE.INDEX)-1
                if (any(binInds < 0)):
                    raise RuntimeError("Some bhInds not matched!! '%s'" % (str(bads)))

                bhHalos = haloNums[binInds]
                bhSubhs = subhNums[binInds]

        # Save To Dict
        # ------------
        hostTable = {}

        # Metadata
        hostTable[OFFTAB.RUN]         = run
        hostTable[OFFTAB.SNAP]        = snap
        hostTable[OFFTAB.VERSION]     = _VERSION
        hostTable[OFFTAB.CREATED]     = datetime.now().ctime()
        hostTable[OFFTAB.FILENAME]    = saveFile

        # BH Data
        hostTable[OFFTAB.BH_INDICES]  = bhInds
        hostTable[OFFTAB.BH_IDS]      = bhIDs
        hostTable[OFFTAB.BH_HALOS]    = bhHalos
        hostTable[OFFTAB.BH_SUBHALOS] = bhSubhs

        # Save to file
        zio.dictToNPZ(hostTable, saveFile, verbose=verbose)

        stop = datetime.now()
        if verbose: print((" - - - - Done after {:s}".format(str(stop-start))))

    return hostTable


def subhalosForBHIDs(run, snap, bhIDs, bhHosts=None, verbose=True):
    """Find the subhalo indices for the given BH ID numbers.

    Arguments
    ---------
    run     <int>    : illustris simulation number {1, 3}
    snap    <int>    : illustris snapshot number {0, 135}
    bhIDs   <int>[N] : target BH ID numbers
    verbose <bool>   : optional, print verbose output

    Returns
    -------
    foundSubh <int>[N] : subhalo index numbers (`-1` for invalid)

    """
    if verbose: print(" - - ParticleHosts.subhalosForBHIDs()")

    # Load (Sub)Halo Offset Table
    # ---------------------------
    if (bhHosts is None):
        if verbose: print(" - - - Loading offset table")
        bhHosts = loadBHHostsSnap(run, snap, loadsave=True, verbose=verbose)

    outIDs  = bhHosts[OFFTAB.BH_IDS]
    outInds = bhHosts[OFFTAB.BH_INDICES]
    outSubh = bhHosts[OFFTAB.BH_SUBHALOS]

    # Convert IDs to Indices
    # ----------------------

    # Sort IDs for faster searching
    sortIDs = np.argsort(outIDs)
    # Find matches in sorted array
    foundSorted = np.searchsorted(outIDs, bhIDs, sorter=sortIDs)
    #    Not found matches will be set to length of array.  These will be caught as incorrect below
    foundSorted[foundSorted == len(sortIDs)] -= 1
    # Reverse map to find matches in original array
    found = sortIDs[foundSorted]

    foundIDs  = outIDs[found]
    foundInds = outInds[found]
    foundSubh = outSubh[found]

    # Check Matches
    # -------------

    # Find incorrect matches
    inds = np.where(bhIDs != foundIDs)[0]
    numIDs = len(bhIDs)
    numBad = len(inds)
    numGood = numIDs-numBad
    if verbose: print((" - - - Matched {:d}/{:d} Good, {:d}/{:d} Bad".format(numGood, numIDs, numBad, numIDs)))
    # Set incorrect matches to '-1'
    if (len(inds) > 0):
        foundIDs[inds]  = -1
        foundInds[inds] = -1
        foundSubh[inds] = -1

    return foundSubh


def _constructOffsetTable(run, snap, verbose=True, bar=None):
    """Construct offset table from halo and subhalo catalogs.

    Each 'entry' is the first particle index number for a group of particles.  Particles are
    grouped by the halos and subhalos they belong to.  The first entry is particles in the first
    subhalo of the first halo.  The last entry for the first halo is particles that dont belong to
    any subhalo (but still belong to the first halo).  The very last entry is for particles that
    dont belong to any halo or subhalo.

    Arguments
    ---------
       run     <int>  : illustris simulation number {1, 3}
       snap    <int>  : illustris snapshot number {0, 135}
       verbose <bool> : optional, print verbose output

    Returns
    -------
       haloNum <int>[N]   : halo      number for each offset entry
       subhNum <int>[N]   : subhalo   number for each offset entry
       offsets <int>[N, 6] : particle offsets for each offset entry

    """

    import illpy_lib as ill

    if verbose: print(" - - ParticleHosts._constructOffsetTable()")

    if (bar is None): bar = bool(verbose)

    # Load (Sub)Halo Catalogs
    #  -----------------------

    # Illustris Data Directory where catalogs are stored
    illpath = GET_ILLUSTRIS_OUTPUT_DIR(run)

    if verbose: print((" - - - Loading Catalogs from '{:s}'".format(illpath)))
    haloCat = ill.groupcat.loadHalos(illpath, snap, fields=None)
    numHalos    = haloCat['count']
    if verbose: print((" - - - - Halos    Loaded ({:7d})".format(numHalos)))
    subhCat = ill.groupcat.loadSubhalos(illpath, snap, fields=None)
    numSubhs = subhCat['count']
    if verbose: print((" - - - - Subhalos Loaded ({:7d})".format(numSubhs)))


    # Initialize Storage
    #  ------------------

    tableSize = numHalos + numSubhs + 1

    # See object description; recall entries are [HALO, SUBHALO, PART0, ... PART5]
    #    (Sub)halo numbers are smaller, use signed-integers for `-1` to be no (Sub)halo
    haloNum = np.zeros(tableSize,               dtype=DTYPE.INDEX)
    subhNum = np.zeros(tableSize,               dtype=DTYPE.INDEX)
    # Offsets approach total number of particles, must be uint64
    offsets = np.zeros([tableSize, PARTICLE._NUM], dtype=DTYPE.ID)

    subh = 0
    offs = 0
    cumHaloParts = np.zeros(PARTICLE._NUM, dtype=DTYPE.ID)
    cumSubhParts = np.zeros(PARTICLE._NUM, dtype=DTYPE.ID)

    pbar = zio.getProgressBar(tableSize)
    if bar: pbar.start()

    # Iterate Over Each Halo
    # ----------------------
    for ii in range(numHalos):

        # Add the number of particles in this halo
        cumHaloParts[:] += haloCat['GroupLenType'][ii, :]

        # Iterate over each Subhalo, in halo ``ii``
        #  -----------------------------------------
        for jj in range(haloCat['GroupNsubs'][ii]):

            # Consistency check: make sure subhalo number is as expected
            if (jj == 0 and subh != haloCat['GroupFirstSub'][ii]):
                print(("ii = {:d}, jj = {:d}, subh = {:d}".format(ii, jj, subh)))
                raise RuntimeError("Subhalo iterator doesn't match Halo's first subhalo!")

            # Add entry for each subhalo
            haloNum[offs] = ii
            subhNum[offs] = subh
            offsets[offs, :] = cumSubhParts

            # Add particles in this subhalo to offset counts
            cumSubhParts[:] += subhCat['SubhaloLenType'][subh, :]

            # Increment subhalo and entry number
            subh += 1
            offs += 1
            if bar: pbar.update(offs)

        # Add Entry for particles with NO subhalo
        haloNum[offs] = ii                        # Still part of halo ``ii``
        subhNum[offs] = -1                        # `-1` means no (sub)halo
        offsets[offs, :] = cumSubhParts

        # Increment particle numbers to include this halo
        cumSubhParts = np.copy(cumHaloParts)

        # Increment entry number
        offs += 1
        if bar: pbar.update(offs)

    # Add entry for particles with NO halo and NO subhalo
    haloNum[offs] = -1
    subhNum[offs] = -1
    offsets[offs, :] = cumSubhParts

    if bar: pbar.finish()

    return haloNum, subhNum, offsets


def _constructBHIndexTable(run, snap, verbose=True):
    """
    Load all BH ID numbers and associate them with 'index' (i.e. order) numbers.

    Arguments
    ---------
       run     <int>  : illustris simulation number {1, 3}
       snap    <int>  : illustris snapshot number {1, 135}
       verbose <bool> : optional, print verbose output

    Returns
    -------
       inds    <int>[N] : BH Index numbers
       bhIDs   <int>[N] : BH particle ID numbers

    """

    if verbose: print(" - - ParticleHosts._constructBHIndexTable()")

    # Illustris Data Directory where catalogs are stored
    illpath = GET_ILLUSTRIS_OUTPUT_DIR(run)

    # Load all BH ID numbers from snapshot (single ``fields`` parameters loads array, not dict)
    if verbose: print((" - - - Loading BHs from Snapshot {:d} in '{:s}'".format(snap, illpath)))
    bhIDs = ill.snapshot.loadSubset(illpath, snap, PARTICLE.BH, fields=SNAPSHOT.IDS)
    numBHs = len(bhIDs)
    if verbose: print((" - - - - BHs Loaded ({:7d})".format(numBHs)))
    # Create 'indices' of BHs
    inds = np.arange(numBHs)
    return inds, bhIDs


def main():
    titleStr = "illpy_lib.subhalos.ParticleHosts.main()"
    print(("\n{:s}\n{:s}\n".format(titleStr, "="*len(titleStr))))

    import sys

    try:
        run   = np.int(sys.argv[1])
        start = np.int(sys.argv[2])
        stop  = np.int(sys.argv[3])
        skip  = np.int(sys.argv[4])

    except:
        # Print Usage
        print("usage:  ParticleHosts RUN SNAP_START SNAP_STOP SNAP_SKIP")
        print("arguments:")
        print("    RUN        <int> : illustris simulation number {1, 3}")
        print("    SNAP_START <int> : illustris snapshot   number {0, 135} to start on")
        print("    SNAP_STOP  <int> :                                     to stop  before")
        print("    SNAP_SKIP  <int> : spacing of snapshots to work on")
        print("")
        # Raise Exception
        raise

    else:
        snaps = np.arange(start, stop, skip)
        print(snaps)

        for sn in snaps:
            sys.stdout.write('\t%3d ... ' % (sn))
            sys.stdout.flush()

            beg = datetime.now()
            table = loadBHHostsSnap(run, sn, convert=0.4, bar=False)
            end = datetime.now()

            sys.stdout.write(' After %s\n' % (str(end-beg)))
            sys.stdout.flush()

    return


if (__name__ == "__main__"):
    main()
