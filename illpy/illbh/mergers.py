"""Module to handle Illustris BH Merger Files.

This module is an interface to the 'blackhole_mergers_<#>.txt' files produced
by Illustris.  Raw Illustris files are only used to initially load data, then
an intermediate numpy npz-file is produced to store a dictionary of merger data
for easier access in all future calls.  Executing the `main()` routine will
prepare the intermediate file, as will calls to the `loadMergers()` function -
if the intermediate file hasn't already been loaded.

The `mergers` are represented as a dictionary object with keys given by the
variables `MERGERS_*`, e.g. `MERGERS_NUM` is the key for the number of mergers.

Internal Parameters
-------------------

Functions
---------


Mergers Dictionary
------------------

Examples
--------

Notes
-----
 - 'Raw Mergers' : these are mergers directly from the illustris files with NO modifications or
                   filtering of any kind.



   The underlying data is in the illustris bh merger files, 'blackhole_mergers_<#>.txt', which are
   processed by `_loadMergersFromIllustris()`.  Each line of the input files is processed by
   `_parse_line_merger()` which returns the redshift ('time') of the merger, and the IDs and masses
   of each BH.  Each `merger` is sorted by time (redshift) in `_importMergers()` and placed in a
   `dict` of all results.  This merger dictionary is saved to a 'raw' savefile whose name is given
   by `savedMergers_rawFilename()`.
   The method `processMergers()` not only loads the merger objects, but also creates mappings of
   mergers to the snapshots nearest where they occur (``mapM2S`) and visa-versa (``mapS2M``); as
   well as mergers which take place exactly during a snapshot iteration (``ontop``).  These three
   maps are included in the merger dictionary.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import datetime
import h5py
import numpy as np
import os
import shutil

from constants import DTYPE, GET_ILLUSTRIS_BH_MERGERS_FILENAMES, GET_MERGERS_COMBINED_FILENAME, \
    MERGERS, _backup_exists

# import zcode.inout as zio

__version__ = '1.0.1'

# VERSION_MAP = 0.21
# VERSION_FIX = 0.31


def main(run=1, output_dir=None, verbose=True, ):
    # Load Mapped Mergers ###
    # re-creates them if needed
    # mergersMapped = loadMappedMergers(run, verbose=verbose)

    # Load Fixed Mergers ###
    # mergersFixed = loadFixedMergers(run, verbose=verbose)

    return


# def combine_raw_merger_files(in_fnames, out_fname, verbose=False):
def combine_raw_merger_files(run, verbose=False):
    """Concatenate the contents of a set of input files into a single output file.

    Arguments
    ---------
    in_fnames : iterable<str>, list of input file names
    out_fname : <str>, output file name
    verbose : <bool> (optional=_VERBOSE), print verbose output

    Returns

    """
    beg = datetime.now()
    print(" - Combining merger files")
    # Raw illustris merger filenames
    in_filenames = GET_ILLUSTRIS_BH_MERGERS_FILENAMES(run)
    # Filename for combined mergers file, raw = unfiltered
    out_raw_fname = GET_MERGERS_COMBINED_FILENAME(run, filtered=False, type='txt')
    # Filename for combined mergers file, filtered (try to remove duplicates)
    out_filtered_fname = GET_MERGERS_COMBINED_FILENAME(run, filtered=True, type='txt')
    if verbose:
        print(" - - Writing   raw    combined mergers to '{}'".format(out_raw_fname))
        print(" - - Writing filtered combined mergers to '{}'".format(out_filtered_fname))

    # Make backups of existing output files
    _backup_exists(out_raw_fname, verbose=verbose)
    _backup_exists(out_filtered_fname, verbose=verbose)

    nums_in = len(in_filenames)
    interv = int(np.floor(nums_in/10))
    count_raw = 0
    count_filt = 0
    num_deleted = 0
    # Open output file for writing
    with open(out_raw_fname, 'w') as out_raw, open(out_filtered_fname, 'w') as out_filt:
        # Iterate over input files
        if verbose: print(" - Iterating over {} input merger files".format(nums_in))
        for ii, in_name in enumerate(in_filenames):
            filt_lines = []    # Stores each line from input file
            filt_scales = []   # Stores the scale-factor of each entry (each line)
            last_scale = 0.0  # Stores the previous scale-factor (time)

            # Iterate over lines in input file
            for mline in open(in_name, 'r'):
                # Write *all* lines to 'raw' output file
                out_raw.write(mline)
                count_raw += 1

                # Scale factor is second, space-separated entry in each line
                this_scale = DTYPE.SCALAR(mline.split()[1])

                # If times go backwards, simulation was restarted.  Erase (previous) overlap segment
                if this_scale < last_scale:
                    # Find lines matching or following the newest read scale-factor
                    time_mask = (this_scale < filt_scales) | np.isclose(filt_scales, this_scale)
                    bads = np.where(time_mask)[0]
                    # Delete these lines
                    if bads.size:
                        # Reverse through so that subsequent indices are still correct
                        for idx in reversed(bads):
                            del filt_lines[idx]
                            del filt_scales[idx]
                        num_deleted += bads.size

                # Store new line
                filt_scales.append(this_scale)
                filt_lines.append(mline)
                count_filt += 1
                last_scale = this_scale

            # Write all stored (filtered) lines to filtered file
            out_filt.writelines(filt_lines)

            if verbose and ii % interv == 0:
                dur = datetime.now()-beg
                print("\t{:5d}/{} = {:.4f} after {}.  Raw: {:5d}; Filtered: {:5d}, Deleted: {:5d}"
                      "".format(ii, nums_in, ii/nums_in, dur, count_raw, count_filt, num_deleted))

    return count_raw, count_filt


def convert_txt_to_hdf5(run, verbose=True, recombine=False):
    """

    Raw mergers are the data directly from illustris without modification.
    """
    # Intermediate filename to store all mergers in single text file
    #    Mergers will be in effectively random order
    combined_txt_fname = GET_MERGERS_RAW_FILENAME(run, type='txt')
    if recombine or not os.path.exists(combined_txt_fname):
        merger_fnames = GET_ILLUSTRIS_BH_MERGERS_FILENAMES(run)
        if verbose:
            print(" - Combining Merger Data from {} files".format(len(merger_fnames)))
        num_lines = combine_raw_merger_files(merger_fnames, combined_txt_fname, verbose)
    else:
        num_lines = sum(1 for line in open(combined_txt_fname))

    scales = np.zeros(num_lines, dtype=DTYPE.SCALAR)
    id_in = np.zeros(num_lines, dtype=DTYPE.ID)
    id_out = np.zeros(num_lines, dtype=DTYPE.ID)
    mass_in = np.zeros(num_lines, dtype=DTYPE.SCALAR)
    mass_out = np.zeros(num_lines, dtype=DTYPE.SCALAR)

    count = 0
    # Go through each merger (each line) in combined file, store parsed values
    if verbose:
        print(" - Loading combined merger data from '{}'".format(combined_txt_fname))
    beg = datetime.now()
    for line in open(combined_txt_fname, 'r'):
        # Get target elements, as appropriate data types, from each line of file
        #    NOTE: `out_mass` is incorrect in illustris
        time, out_id, out_mass, in_id, in_mass = _parse_line_merger(line)
        # Store values
        scales[count] = time
        id_in[count] = in_id
        id_out[count] = out_id
        mass_in[count] = in_mass
        mass_out[count] = out_mass
        count += 1

    # Resize arrays for the number of values actually read
    if verbose:
        print(" - Read {} lines after {}".format(count, datetime.now()-beg))
    if count != num_lines:
        raise ValueError("`count` = {} != `num_lines` = {}".format(count, num_lines))

    # Find indices which sort by time
    inds = np.argsort(scales)
    # Use indices to reorder arrays
    scales   = scales[inds]
    id_in    = id_in[inds]
    id_out   = id_out[inds]
    mass_in  = mass_in[inds]
    mass_out = mass_out[inds]

    # Calculate Meta-Data
    # -------------------
    # Find all unique BH IDs
    all_ids = np.append(id_in, id_out)
    all_ids = np.unique(all_ids)
    if verbose: print(" - Mergers: {}, Unique BH: {}".format(inds.size, all_ids.size))

    # Build merger tree
    '''
    mnext = -1*np.ones(num_lines, dtype=int)
    mprev_in = -1*np.ones(num_lines, dtype=int)
    mprev_out = -1*np.ones(num_lines, dtype=int)
    if verbose: print("Building merger tree")
    for this, this_out in enumerate(id_out[:-1]):
        # See if the 'out' BH from this merger become the 'in' BH from another one
        next_in = np.where(this_out == id_in[this:])[0]
        if next_in.size:
            # Select first, if multiple subsequent mergers
            next_in = next_in[0]
            # The next merger for this ('out') BH is `next_in`
            #    Shouldnt have been set before
            if mnext[this] >= 0:
                raise ValueError("`this` = {}, `next_in` = {}, mnext = `{}`".format(
                    this, next_in, mnext[this]))
            mnext[this] = next_in
            # The previous merger for the `next_in`--'in' BH, is this one
            #    Shouldnt have been set before
            if mprev_in[this] >= 0:
                raise ValueError("`this` = {}, `next_in` = {}, mprev_in = `{}`".format(
                    this, next_in, mprev_in[this]))

            mprev_in[next_in] = this
        # See if the 'out' BH from this merger become the 'out' BH from another one
        next_out = np.where(this_out == id_out[this:])[0]
        if next_out.size:
            # Select first, if multiple subsequent mergers
            next_out = next_out[0]
            # The next merger for this ('out') BH is `next_in`
            #    Shouldnt have been set before
            if mnext[this] >= 0:
                raise ValueError("`this` = {}, `next_out` = {}, mnext = `{}`".format(
                    this, next_out, mnext[this]))
            mnext[this] = next_out
            # The previous merger for the `next_in`--'in' BH, is this one
            #    Shouldnt have been set before
            if mnext[this] >= 0:
                raise ValueError("`this` = {}, `next_out` = {}, mprev_out = `{}`".format(
                    this, next_out, mprev_out[this]))
            mprev_out[next_out] = this
    '''

    # Write Raw data to hdf5 file
    hdf5_fname = GET_MERGERS_RAW_FILENAME(run, type='hdf5')
    if verbose: print("Saving merger data to '{}'".format(hdf5_fname))
    with h5py.File(hdf5_fname, 'w') as h5file:
        # Add metadata in "Header" dataset
        head = h5file.create_group('Header')
        head.attrs['script'] = str(__file__)
        head.attrs['script_version'] = str(__version__)
        head.attrs['created'] = str(datetime.now().ctime())
        head.attrs['simulation'] = 'Illustris-{}'.format(run)
        head.attrs['description'] = (
            "Illustris blackhole merger data, combined from all of the "
            "individual blackhole (BH) merger text files.  The content of the "
            "data is completely unchanged.  Each merger involves two BH, the 'out' "
            "BH which persists after the merger, and the 'in' BH which is consumed.  "
            "NOTE: the mass of the 'out' BH is incorrect in this data.  The values "
            "given correspond to the total cell (dynamical) mass, instead of the BH "
            "mass itself."
        )
        head['unique_ids'] = all_ids

        # Add merger data
        time_dset = h5file.create_dataset(MERGERS.SCALE, data=scales)
        time_dset.attrs['units'] = 'Cosmological scale factor'
        h5file.create_dataset(MERGERS.ID_IN, data=id_in)
        h5file.create_dataset(MERGERS.ID_OUT, data=id_out)
        h5file.create_dataset(MERGERS.MASS_IN, data=mass_in)
        h5file.create_dataset(MERGERS.MASS_OUT, data=mass_out)

        '''
        # Merger tree data
        h5file.create_dataset(MERGERS.NEXT, data=mnext)
        h5file.create_dataset(MERGERS.PREV_IN, data=mprev_in)
        h5file.create_dataset(MERGERS.PREV_OUT, data=mprev_out)
        '''

    if verbose:
        fsize = os.path.getsize(hdf5_fname)/1024/1024
        print(" - Saved to '{}', Size: '{}' MB".format(hdf5_fname, fsize))

    return  # scales, id_in, id_out, mass_in, mass_out, hdf5_fname

'''
def loadMappedMergers(run, verbose=True, loadsave=True):
    """Load or create Mapped Mergers Dictionary as needed.
    """

    if verbose: print(" - - BHMergers.loadMappedMergers")

    mappedFilename = GET_MERGERS_RAW_MAPPED_FILENAME(run, VERSION_MAP)

    ## Load Existing Mapped Mergers
    #  ----------------------------
    if(loadsave):
        if verbose: print(" - - - Loading saved data from '%s'" % (mappedFilename))
        # If file exists, load data
        if(os.path.exists(mappedFilename)):
            mergersMapped = zio.npzToDict(mappedFilename)
        else:
            print(" - - - - '%s' does not exist.  Recreating" % (mappedFilename))
            loadsave = False


    ## Recreate Mappings
    #  -----------------
    if(not loadsave):
        if verbose: print(" - - - Recreating mapped mergers")

        # Load Raw Mergers
        scales, ids, masses, filename = load_raw_mergers(run, verbose=verbose)

        ### Create Mapping Between Mergers and Snapshots ###
        mapM2S, mapS2M, ontop = _mapToSnapshots(scales)

        # Store in dictionary
        mergersMapped = { MERGERS.FILE      : mappedFilename,
                          MERGERS.RUN       : run,
                          MERGERS.NUM       : len(scales),
                          MERGERS.CREATED   : datetime.now().ctime(),
                          MERGERS.VERSION   : VERSION_MAP,

                          MERGERS.SCALES    : scales,
                          MERGERS.IDS       : ids,
                          MERGERS.MASSES    : masses,

                          MERGERS.MAP_MTOS  : mapM2S,
                          MERGERS.MAP_STOM  : mapS2M,
                          MERGERS.MAP_ONTOP : ontop,
                          }

        zio.dictToNPZ(mergersMapped, mappedFilename, verbose=verbose)


    return mergersMapped


def loadFixedMergers(run, verbose=True, loadsave=True):
    """
    Load BH Merger data with duplicats removes, and masses corrected.

    Arguments
    ---------
       run      <int>  : illustris simulation run number {1, 3}
       verbose  <bool> : optional, print verbose output
       loadsave <bool> : optional, load existing save file (recreate if `False`)

    Returns
    -------
       mergersFixed <dict> : dictionary of 'fixed' mergers, most entries shaped [N, 2] for `N`
                             mergers, and an entry for each {``BH_TYPE.IN``, ``BH_TYPE.OUT``}

    """

    if verbose: print(" - - BHMergers.loadFixedMergers")

    fixedFilename = GET_MERGERS_FIXED_FILENAME(run, VERSION_FIX)

    ## Try to Load Existing Mapped Mergers
    if(loadsave):
        if verbose: print(" - - - Loading from save '%s'" % (fixedFilename))
        if(os.path.exists(fixedFilename)):
            mergersFixed = zio.npzToDict(fixedFilename)
        else:
            print(" - - - - '%s' does not exist.  Recreating." % (fixedFilename))
            loadsave = False


    ## Recreate Fixed Mergers
    if(not loadsave):
        if verbose: print(" - - - Creating Fixed Mergers")
        # Load Mapped Mergers
        mergersMapped = loadMappedMergers(run, verbose=verbose)
        # Fix Mergers
        mergersFixed = _fixMergers(run, mergersMapped, verbose=verbose)
        # Save
        zio.dictToNPZ(mergersFixed, fixedFilename, verbose=verbose)


    return mergersFixed


def _fixMergers(run, mergers, verbose=True):
    """
    Filter and 'fix' input merger catalog.

    This includes:
     - Remove duplicate entries (Note-1)
     - Load 'fixed' out-BH masses from ``BHMatcher`` (which uses ``BHDetails`` entries)

    Arguments
    ---------
       run     <int>  : illustris simulation number {1, 3}
       mergers <dict> : input dictionary of unfiltered merger events
       verbose <bool> : optional, print verbose output

    Returns
    -------
       fixedMergers <dict> : filtered merger dictionary

    Notes
    -----
       1 : There are 'duplicate' entries which have different occurence times (scale-factors)
           suggesting that there is a problem with the actual merger, not just the logging.
           This is not confirmed.  Currently, whether the times match or not, the *later*
           merger entry is the only one that is preserved in ``fixedMergers``

    """
    from illpy.illbh import BHMatcher

    if verbose: print(" - - BHMergers._fixMergers")

    # Make copy to modify
    fixedMergers = dict(mergers)

    # Remove Repeated Entries
    # =======================
    # Remove entries where IDs match a second time (IS THIS ENOUGH?!)

    ids    = fixedMergers[MERGERS.IDS]
    scales = fixedMergers[MERGERS.SCALES]

    # First sort by ``BH_TYPE.IN`` then ``BH_TYPE.OUT`` (reverse of given order)
    sort = np.lexsort((ids[:, BH_TYPE.OUT], ids[:, BH_TYPE.IN]))

    badInds = []
    numMismatch = 0

    if verbose: print(" - - - Examining %d merger entries" % (len(sort)))

    # Iterate over all entries
    for ii in range(len(sort)-1):

        this = ids[sort[ii]]
        jj = ii+1

        # Look through all examples of same BH_TYPE.IN
        while(ids[sort[jj], BH_TYPE.IN] == this[BH_TYPE.IN]):
            # If BH_TYPE.OUT also matches, this is a duplicate -- store first entry as bad |NOTE-1|
            if(ids[sort[jj], BH_TYPE.OUT] == this[BH_TYPE.OUT]):

                # Double check that time also matches
                if(scales[sort[ii]] != scales[sort[jj]]): numMismatch += 1
                badInds.append(sort[ii])
                break

            jj += 1

        # } while
    # ii

    if verbose: print(" - - - Total number of duplicates = %d" % (len(badInds)))
    if verbose: print(" - - - Number with mismatched times = %d" % (numMismatch))

    # Remove Duplicate Entries
    for key in MERGERS_PHYSICAL_KEYS:
        fixedMergers[key] = np.delete(fixedMergers[key], badInds, axis=0)

    # Recalculate maps
    mapM2S, mapS2M, ontop = _mapToSnapshots(fixedMergers[MERGERS.SCALES])
    fixedMergers[MERGERS.MAP_MTOS] = mapM2S
    fixedMergers[MERGERS.MAP_STOM] = mapS2M
    fixedMergers[MERGERS.MAP_ONTOP] = ontop

    # Change number, creation date, and version
    oldNum = len(mergers[MERGERS.SCALES])
    newNum = len(fixedMergers[MERGERS.SCALES])
    fixedMergers[MERGERS.NUM] = newNum
    fixedMergers[MERGERS.CREATED] = datetime.now().ctime()
    fixedMergers[MERGERS.VERSION] = VERSION_FIX

    if verbose: print(" - - - Number of Mergers %d ==> %d" % (oldNum, newNum))

    # Fix Merger 'Out' Masses
    #  =======================
    if verbose: print(" - - - Loading reconstructed 'out' BH masses")
    masses = fixedMergers[MERGERS.MASSES]
    aveBef = np.average(masses[:, BH_TYPE.OUT])
    massOut = BHMatcher.inferMergerOutMasses(run, mergers=fixedMergers, verbose=verbose)
    masses[:, BH_TYPE.OUT] = massOut
    aveAft = np.average(masses[:, BH_TYPE.OUT])
    if verbose: print(" - - - - Ave mass:  %.4e ===> %.4e" % (aveBef, aveAft))

    return fixedMergers
'''

def _parse_line_merger(line):
    """Process quantities from each line of the illustris merger files.

    See 'http://www.illustris-project.org/w/index.php/Blackhole_Files' for
    details regarding the illustris BH file structure.

    The format of each line is:
        "PROC-NUM  TIME  ID1  MASS1  ID0  MASS0"
        where
            '1' corresponds to the 'out'/'accretor'/surviving BH
            '0' corresponds to the 'in' /'accreted'/eliminated BH
        NOTE: that `MASS1` (`out_mass`) is INCORRECT in illustris (dynamical mass, instead of BH)

    Returns
    -------
    time     : scalar, redshift of merger
    out_id   : long, id number of `out` BH
    out_mass : scalar, mass of `out` BH in simulation units (INCORRECT VALUE)
    in_id    : long, id number of `in` BH
    in_mass  : scalar, mass of `in` BH in simulation units

    """
    strs     = line.split()
    # Convert to proper types
    time     = DTYPE.SCALAR(strs[1])
    out_id   = DTYPE.ID(strs[2])
    out_mass = DTYPE.SCALAR(strs[3])
    in_id    = DTYPE.ID(strs[4])
    in_mass  = DTYPE.SCALAR(strs[5])
    return time, out_id, out_mass, in_id, in_mass

'''
def _mapToSnapshots(scales, verbose=True):
    """Find the snapshot during which, or following each merger
    """

    if verbose: print(" - - BHMergers._mapToSnapshots")

    numMergers = len(scales)

    # Load Cosmology
    import illpy.illcosmo
    cosmo = illpy.illcosmo.cosmology.Cosmology()
    snapScales = cosmo.scales()

    # Map Mergers-2-Snapshots: snapshot before (or ontop) of each merger
    mapM2S = np.zeros(numMergers, dtype=DTYPE.INDEX)
    # Map Snapshots-2-Mergers: list of mergers just-after (or ontop) of each snapshot
    mapS2M = [[] for ii in range(cosmo.num)]
    # Flags if merger happens exactly on a snapshot (init to False=0)
    ontop  = np.zeros(numMergers, dtype=bool)

    # Find snapshots on each side of merger time ###

    # Find the snapshot just below and above each merger.
    #     each entry (returned) is [low, high, dist-low, dist-high]
    #     low==high if the times match (within function's default uncertainty)
    snapBins = [_findBoundingBins(sc, snapScales) for sc in scales]

    # Create Mappings
    # ---------------

    if verbose:
        print(" - - - Creating mappings")
        pbar = zio.getProgressBar(numMergers)

    for ii, bins in enumerate(snapBins):
        tsnap = bins[1]                                                                             # Set snapshot to upper bin
        mapM2S[ii] = tsnap                                                                          # Set snapshot for this merger
        mapS2M[tsnap].append(ii)                                                                    # Add merger to this snapshot
        # If this merger takes place ontop of snapshot, set flag
        if(bins[0] == bins[1]): ontop[ii] = True

        # Print Progress
        if verbose: pbar.update(ii)

    # ii

    if verbose: pbar.finish()

    # Find the most mergers in a snapshot
    numPerSnap = np.array([len(s2m) for s2m in mapS2M])
    mostMergers = np.max(numPerSnap)
    mostIndex = np.where(mostMergers == numPerSnap)[0]
    # Find the number of ontop mergers
    numOntop = np.count_nonzero(ontop)
    if verbose: print(" - - - Snapshot %d with the most (%d) mergers" % (mostIndex, mostMergers))
    if verbose: print(" - - - %d (%.2f) ontop mergers" % (numOntop, 1.0*numOntop/nums))

    return mapM2S, mapS2M, ontop
'''

'''
def _findBoundingBins(target, bins, thresh=1.0e-5):
    """
    Find the array indices (of "bins") bounding the "target"

    If target is outside bins, the missing bound will be 'None'
    low and high will be the same, if the target is almost exactly[*1] equal to a bin

    [*1] : How close counds as effectively the same is set by 'DEL_TIME_THRESH' below

    arguments
    ---------
        target  : [] value to be compared
        bins    : [] list of values to compare to the 'target'

    output
    ------
        low  : [int] index below target (or None if none)
        high : [int] index above target (or None if none)

    """

    # deltat  : test whether the fractional difference between two values is less than threshold
    #           This function allows the later conditions to accomodate smaller numerical
    #           differences, between effectively the same value  (e.g.   1.0 vs. 0.9999999999989)
    #
    if(thresh == 0.0): deltat = lambda x, y : False
    else               : deltat = lambda x, y : np.abs(x-y)/np.abs(x) <= thresh

    nums   = len(bins)
    # Find bin above (or equal to) target
    high = np.where((target <= bins) | deltat(target, bins))[0]
    if(len(high) == 0): high = None
    # Select first bin above target
    else:
        high = high[0]
        dhi  = bins[high] - target

    # Find bin below (or equal to) target
    low  = np.where((target >= bins) | deltat(target, bins))[0]
    if(len(low)  == 0): low  = None
    # Select  last bin below target
    else:
        low  = low[-1]
        dlo  = bins[low] - target

    # Print warning on error
    if(low == None or high == None):
        print("BHMergers._findBoundingBins: target = %e, bins = {%e, %e}; low, high = %s, %s !" % \
            (target, bins[0], bins[-1], str(low), str(high)))
        raise RuntimeError("Could not find bins!")

    return [low, high, dlo, dhi]
'''
if __name__ == "__main__":
    main()
