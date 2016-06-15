"""Handle Illustris blackhole details files.

Details are accessed via 'intermediate' files which are reorganized versions of the 'raw' illustris
files 'blackhole_details_<#>.txt'.  The `main()` function assures that details entries are properly
converted from raw to processed form, organized by time of entry instead of processor.  Those
details can then be accessed by snapshot and blackhole ID number.

Functions
---------
    processDetails
    organize_txt_by_snapshot
    formatDetails

    _reorganizeBHDetailsFiles
    _convertDetailsASCIItoNPZ
    _convertDetailsASCIItoNPZ_snapshot
    _load_details_txt
    _parse_illustris_details_line

    loadBHDetails
    _getPrecision

Details Dictionary
------------------
   { DETAILS_RUN       : <int>, illustris simulation number in {1, 3}
     DETAILS_NUM       : <int>, total number of mergers `N`
     DETAILS_FILE      : <str>, name of save file from which mergers were loaded/saved
     DETAILS_CREATED   : <str>, date and time this file was created
     DETAILS_VERSION   : <flt>, version of BHDetails used to create file

     DETAILS_IDS       : <uint64>[N], BH particle ID numbers for each entry
     DETAILS_SCALES    : <flt64> [N], scale factor at which each entry was written
     DETAILS_MASSES    : <flt64> [N], BH mass
     DETAILS_MDOTS     : <flt64> [N], BH Mdot
     DETAILS_RHOS      : <flt64> [N], ambient mass-density
     DETAILS_CS        : <flt64> [N], ambient sound-speed
   }


Notes
-----
  - The BH Details files from illustris, 'blackhole_details_<#>.txt' are organized by the processor
    on which each BH existed in the simulation.  The method `_reorganizeBHDetails()` sorts each
    detail entry instead by the time (scalefactor) of the entry --- organizing them into files
    grouped by which snapshot interval the detail entry corresponds to.  The reorganization is
    first done into 'temporary' ASCII files before being converted into numpy `npz` files by the
    method `_convertDetailsASCIItoNPZ()`.  The `npz` files are effectively dictionaries storing
    the select details parameters (i.e. mass, BH ID, mdot, rho, cs), along with some meta data
    about the `run` number, and creation time, etc.  Execution of the BHDetails ``main`` routine
    checks to see if the npz files exist, and if they do not, they are created.

  - There are also routines to obtain the details entries for a specific BH ID.  In particular,
    the method `detailsForBH()` will return the details entry/entries for a target BH ID and
    run/snapshot.

  - Illustris Blackhole Details Files 'blackhole_details_<#>.txt'
    - Each entry is given as
      0   1            2     3     4    5
      ID  scalefactor  mass  mdot  rho  cs


"""

import h5py
import os
import warnings
import numpy as np
from datetime import datetime

import zcode.inout as zio

# from illpy.Constants import DTYPE, NUM_SNAPS
# from . import BHConstants
# from . BHConstants import DETAILS
from . bhconstants import DETAILS, DTYPE, GET_DETAILS_ORGANIZED_FILENAME, \
    GET_ILLUSTRIS_BH_DETAILS_FILENAMES, GET_SNAPSHOT_SCALES, NUM_SNAPS

__version__ = '1.0'

VERSION = 0.23                                    # Version of BHDetails

_DEF_SCALE_PRECISION = -8                               # Default precision


def process(run, output_dir=None, verbose=True, reorganize=False):
    """
    """

    output_dir = get_out_dir(run, output_dir)
    if verbose: print("Output directory: '{}'".format(output_dir))

    # Raw ('txt') ==> Organized ('txt')
    # ---------------------------------
    organized_fnames_txt = [GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='txt')
                            for snap in range(NUM_SNAPS)]
    org_txt_exists = _all_exist(organized_fnames_txt)
    if verbose and org_txt_exists:
        print("All 'organized' details txt files exist.")

    # Move all Details Entries from Illustris Raw Details files, into files with details organized
    #    by snapshot.  No processing of data.
    if reorganize or not org_txt_exists:
        if verbose: print("Reorganizing details files.")
        # Organize Details by Snapshot Time; create new, temporary ASCII Files
        organize_txt_by_snapshot(run, verbose=verbose)

    # Organized ('txt') ==> Organized ('hdf5')
    # ----------------------------------------
    # Convert data from ASCII to hdf5, still no processing / trimming of data
    organized_fnames_hdf5 = [GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='hdf5')
                             for snap in range(NUM_SNAPS)]
    org_hd5f_exists = _all_exist(organized_fnames_hdf5)
    if verbose and org_hd5f_exists:
        print("All 'organized' details hdf5 files exist.")

    # Create Dictionary Details Files
    # formatDetails(run, verbose=verbose)

    return


def organize_txt_by_snapshot(run, verbose=True):
    """
    """
    if verbose: print(" - BHDetails.organize_txt_by_snapshot")

    # Load cosmology
    snap_scales = GET_SNAPSHOT_SCALES()

    organized_fnames = [GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='txt')
                        for snap in range(NUM_SNAPS)]
    raw_fnames = GET_ILLUSTRIS_BH_DETAILS_FILENAMES(run)

    # Make sure output path is okay
    head_dir, tail_path = os.path.split(organized_fnames[0])
    if not os.path.exists(head_dir):
        os.makedirs(head_dir)
    if not os.path.isdir(head_dir):
        raise RuntimeError("Output '{}' is not a valid directory".format(head_dir))
    # Open each temp file
    org_files = [open(tfil, 'w') for tfil in organized_fnames]

    num_org = len(organized_fnames)
    num_raw  = len(raw_fnames)
    num_iter = int(np.floor(num_raw/100))
    if verbose:
        print((" - - Organizing %d raw files into %d organized files" % (num_raw, num_org)))
    beg = datetime.now()
    num_lines = 0
    for ii, raw_fn in enumerate(raw_fnames):
        raw_lines = []
        raw_scales = []
        # Load all lines and entry scale-factors from raw details file
        for dline in open(raw_fn):
            raw_lines.append(dline)
            # Extract scale-factor from line
            detScale = DTYPE.SCALAR(dline.split()[1])
            raw_scales.append(detScale)

        # Convert to array
        raw_lines  = np.array(raw_lines)
        raw_scales = np.array(raw_scales)

        # If file is empty, continue
        if not raw_lines.size or not raw_scales.size:
            continue

        # Round snapshot scales to desired precision
        round_scales = np.around(snap_scales, -_DEF_SCALE_PRECISION)

        # Find snapshots following each entry (right-edge) or equal (include right: 'right=True')
        snap_bins = np.digitize(raw_scales, round_scales, right=True)

        # For each Snapshot, write appropriate lines
        for jj in range(num_org):
            inds = np.where(snap_bins == jj)[0]
            if inds.size:
                org_files[jj].writelines(raw_lines[inds])
                num_lines += inds.size

        if verbose and ii % num_iter == 0:
            dur = datetime.now() - beg
            print(" - - - {:5d}/{} = {:.4f} after {}; {:.3e} lines written".format(
                ii, num_raw, ii/num_raw, dur, num_lines))

    # Close out details files
    tot_file_sizes = 0.0
    for ii, orgf in enumerate(org_files):
        orgf.close()
        tot_file_sizes += os.path.getsize(orgf.name)

    if verbose:
        tot_file_sizes /= 1024/1024
        ave_size = tot_file_sizes/num_org
        print(" - - Total organized size = '{:.2f}' [MB], average = '{:.2f}' [MB]".format(
            tot_file_sizes, ave_size))

    return


def convert_txt_to_hdf5(run, verbose=True):
    """
    """
    fnames_txt = [GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='txt')
                  for snap in range(NUM_SNAPS)]
    txt_exist = _all_exist(fnames_txt)
    if not txt_exist:
        raise ValueError("'txt' files (e.g. '{}') do not all exist.".format(fnames_txt[0]))

    fnames_hdf5 = [GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='hdf5')
                   for snap in range(NUM_SNAPS)]

    if verbose: print(" - Converting files from 'txt' to 'hdf5'")
    tot_num_entries = 0
    beg = datetime.now()
    for snap in range(NUM_SNAPS):
        with open(fnames_txt[snap], 'r') as tfile, h5py.File(fnames_hdf5[snap], 'w') as h5file:
            # Load details from ASCII File
            id, scale, mass, mdot, rho, cs = _load_details_txt(tfile)
            # Sort by BH ID, then by time (scale-factor)
            inds = np.lexsort((scale, id))
            id = id[inds]
            scale = scale[inds]
            mass = mass[inds]
            mdot = mdot[inds]
            rho = rho[inds]
            cs = cs[inds]
            # Find unique IDs
            q_ids, q_locs, q_counts = np.unique(id, return_index=True, return_counts=True)

            # Create Header with meta/summary data
            head = h5file.create_group('Header')
            head.attrs['script'] = str(__file__)
            head.attrs['script_version'] = str(__version__)
            head.attrs['created'] = str(datetime.now().ctime())
            head.attrs['simulation'] = 'Illustris-{}'.format(run)
            head.attrs['num_entries'] = id.size
            head.attrs['num_blackholes'] = q_ids.size

            # If there are no Blackholes in this snapshot, only write meta-data, skip rest
            if not id.size:
                continue

            head.attrs['time_first'] = np.min(scale)
            head.attrs['time_last'] = np.max(scale)

            # Store data on the unique BHs
            unique_group = h5file.create_group('unique')
            unique_group.create_dataset(DETAILS.ID, data=q_ids)
            unique_group.create_dataset('first_index', data=q_locs)
            unique_group.create_dataset('num_entries', data=q_counts)

            # Store all 'details' data
            time_dset = h5file.create_dataset(DETAILS.SCALE, data=scale)
            time_dset.attrs['desc'] = 'Cosmological scale factor'
            h5file.create_dataset(DETAILS.ID, data=id)
            h5file.create_dataset(DETAILS.MASS, data=mass)
            h5file.create_dataset(DETAILS.MDOT, data=mdot)
            h5file.create_dataset(DETAILS.RHO, data=rho)
            h5file.create_dataset(DETAILS.CS, data=cs)

            tot_num_entries += id.size

        if verbose:
            dur = datetime.now() - beg
            print(" - {:3d} BH: {:5d}, entries: {:6d}, median per: {:5.0f}."
                  " Total: {:.2e} entries after {}".format(
                      snap, q_ids.size, id.size, np.median(q_counts), tot_num_entries, dur))

    return


def merge_downsample_hdf5(run, verbose=True):
    fnames_hdf5 = [GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='hdf5')
                   for snap in range(NUM_SNAPS)]

    target_scales = GET_SUBBOX_TIMES(run)

    beg = datetime.now()
    for snap in range(NUM_SNAPS):
        with open(fnames_txt[snap], 'r') as tfile, h5py.File(fnames_hdf5[snap], 'w') as h5file:
            # Load details from ASCII File
            id, scale, mass, mdot, rho, cs = _load_details_txt(tfile)
            # Sort by BH ID, then by time (scale-factor)
            inds = np.lexsort((scale, id))
            id = id[inds]
            scale = scale[inds]
            mass = mass[inds]
            mdot = mdot[inds]
            rho = rho[inds]
            cs = cs[inds]
            # Find unique IDs
            q_ids, q_locs, q_counts = np.unique(id, return_index=True, return_counts=True)

            # Create Header with meta/summary data
            head = h5file.create_group('Header')
            head.attrs['script'] = str(__file__)
            head.attrs['script_version'] = str(__version__)
            head.attrs['created'] = str(datetime.now().ctime())
            head.attrs['simulation'] = 'Illustris-{}'.format(run)
            head.attrs['num_entries'] = id.size
            head.attrs['num_blackholes'] = q_ids.size

            # If there are no Blackholes in this snapshot, only write meta-data, skip rest
            if not id.size:
                continue

            head.attrs['time_first'] = np.min(scale)
            head.attrs['time_last'] = np.max(scale)

            # Store data on the unique BHs
            unique_group = h5file.create_group('unique')
            unique_group.create_dataset(DETAILS.ID, data=q_ids)
            unique_group.create_dataset('first_index', data=q_locs)
            unique_group.create_dataset('num_entries', data=q_counts)

            # Store all 'details' data
            time_dset = h5file.create_dataset(DETAILS.SCALE, data=scale)
            time_dset.attrs['desc'] = 'Cosmological scale factor'
            h5file.create_dataset(DETAILS.ID, data=id)
            h5file.create_dataset(DETAILS.MASS, data=mass)
            h5file.create_dataset(DETAILS.MDOT, data=mdot)
            h5file.create_dataset(DETAILS.RHO, data=rho)
            h5file.create_dataset(DETAILS.CS, data=cs)

            tot_num_entries += id.size

        if verbose:
            dur = datetime.now() - beg
            print(" - {:3d} BH: {:5d}, entries: {:6d}, median per: {:5.0f}."
                  " Total: {:.2e} entries after {}".format(
                      snap, q_ids.size, id.size, np.median(q_counts), tot_num_entries, dur))

    return


def _load_details_txt(txt_file):
    # Files have some blank lines in them... Clean
    lines = txt_file.readlines()
    num_lines = len(lines)

    # Allocate storage
    id    = np.zeros(num_lines, dtype=DTYPE.ID)
    scale = np.zeros(num_lines, dtype=DTYPE.SCALAR)
    mass  = np.zeros(num_lines, dtype=DTYPE.SCALAR)
    mdot  = np.zeros(num_lines, dtype=DTYPE.SCALAR)
    rho   = np.zeros(num_lines, dtype=DTYPE.SCALAR)
    cs    = np.zeros(num_lines, dtype=DTYPE.SCALAR)

    count = 0
    # Iterate over lines, storing only those with content (should be all)
    for lin in lines:
        lin = lin.strip()
        if len(lin):
            tid, tim, mas, dot, den, tcs = _parse_illustris_details_line(lin)
            id[count] = tid
            scale[count] = tim
            mass[count] = mas
            mdot[count] = dot
            rho[count] = den
            cs[count] = tcs
            count += 1

    # Trim excess (shouldn't be needed)
    if count != num_lines:
        id    = id[:count]
        scale = scale[:count]
        mass  = mass[:count]
        mdot  = mdot[:count]
        rho   = rho[:count]
        cs    = cs[:count]

    return id, scale, mass, mdot, rho, cs


def _parse_illustris_details_line(instr):
    """Parse a line from an Illustris blachole_details_#.txt file

    The line is formatted (in C) as:
        "BH=%llu %g %g %g %g %g\n",
        (long long) P[n].ID, All.Time, BPP(n).BH_Mass, mdot, rho, soundspeed

    Arguments
    ---------

    Returns
    -------
        ID, time, mass, mdot, rho, cs

    """
    args = instr.split()
    # First element is 'BH=########', trim to just the id number
    args[0] = args[0].split("BH=")[-1]
    idn  = DTYPE.ID(args[0])
    time = DTYPE.SCALAR(args[1])
    mass = DTYPE.SCALAR(args[2])
    mdot = DTYPE.SCALAR(args[3])
    rho  = DTYPE.SCALAR(args[4])
    cs   = DTYPE.SCALAR(args[5])
    return idn, time, mass, mdot, rho, cs


def loadBHDetails(run, snap, loadsave=True, verbose=True):
    """Load Blackhole Details dictionary for the given snapshot.

    If the file does not already exist, it is recreated from the temporary ASCII files, or directly
    from the raw illustris ASCII files as needed.

    Arguments
    ---------
        run     : <int>, illustris simulation number {1, 3}
        snap    : <int>, illustris snapshot number {0, 135}
        loadsave <bool> :
        verbose  <bool> : print verbose output

    Returns
    -------
        dets    : <dict>, BHDetails dictionary object for target snapshot

    """
    if verbose: print(" - - BHDetails.loadBHDetails")

    detsName = BHConstants.GET_DETAILS_SAVE_FILENAME(run, snap, VERSION)

    # Load Existing Save File
    if(loadsave):
        if verbose: print((" - - - Loading details from '%s'" % (detsName)))
        if(os.path.exists(detsName)):
            dets = zio.npzToDict(detsName)
        else:
            loadsave = False
            warnStr = "%s does not exist!" % (detsName)
            warnings.warn(warnStr, RuntimeWarning)

    # If file does not exist, or is wrong version, recreate it
    if(not loadsave):
        if verbose: print(" - - - Re-converting details")
        # Convert ASCII to NPZ
        saveFile = _convertDetailsASCIItoNPZ_snapshot(run, snap, loadsave=True, verbose=verbose)
        # Load details from newly created save file
        dets = zio.npzToDict(saveFile)

    return dets


def _getPrecision(args):
    """Estimate the precision needed to differenciate between elements of an array
    """
    diffs = np.fabs(np.diff(sorted(args)))
    inds  = np.nonzero(diffs)
    if len(inds) > 0:
        minDiff = np.min(diffs[inds])
    else:
        minDiff = np.power(10.0, _DEF_SCALE_PRECISION)
    order = int(np.log10(0.49*minDiff))
    return order


def get_out_dir(run, output_dir=None):
    """
    /n/home00/lkelley/ghernquistfs1/illustris/data/%s/blackhole/details/
    """
    if output_dir is None:
        output_dir = os.path.join(_PROCESSED_DIR % (_ILLUSTRIS_RUN_NAMES[run]), _DEF_OUTPUT_DIR, '')
    return output_dir
