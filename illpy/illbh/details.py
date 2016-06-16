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


NOTE: 'details' entries at the same time as a 'merger' seem to always be 'after' merger based on
      the masses.

"""

from datetime import datetime
import h5py
import os
import numpy as np
import sys
import warnings

import zcode.inout as zio

# from illpy.Constants import DTYPE, NUM_SNAPS
# from . import BHConstants
# from . BHConstants import DETAILS
from constants import _all_exist, DETAILS, DTYPE, GET_DETAILS_ORGANIZED_FILENAME, \
    GET_ILLUSTRIS_BH_DETAILS_FILENAMES, GET_MERGERS_DETAILS_FILENAME, GET_MERGERS_RAW_FILENAME, \
    GET_OUTPUT_DETAILS_FILENAME, GET_SNAPSHOT_SCALES, GET_SUBBOX_TIMES, MERGERS, NUM_SNAPS

__version__ = '1.0'

VERSION = 0.23                                    # Version of BHDetails

_DEF_SCALE_PRECISION = -8                               # Default precision


class _MTYPE:
    M_BEF_D = "Merger then detail entry"
    D_BEF_M = "Detail entry then merger entry"
    UNKNOWN = "Unknown merger vs. detail order"
    CONFLICT = "Details in conflict with merger"
    ASYNC = "No details at merger time"
    ASYNC_CONFLICT = "Surrounding details conflict with merger"

    _KEYS = ["M_BEF_D", "D_BEF_M", "UNKNOWN", "CONFLICT", "ASYNC", "ASYNC_CONFLICT"]
    @classmethod
    def _VALS(cls):
        for key in cls._KEYS:
            yield getattr(cls, key)

def main(run, output_dir=None, verbose=True, reorganize=False, reconvert=False):
    """
    """

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
        if verbose: print("Organizing details files.")
        # Organize Details by Snapshot Time; create new, temporary ASCII Files
        beg = datetime.now()
        organize_txt_by_snapshot(run, verbose=verbose)
        if verbose: print("Organization complete after {}".format(datetime.now()-beg))

    # Organized ('txt') ==> Organized ('hdf5')
    # ----------------------------------------
    # Convert data from ASCII to hdf5, still no processing / trimming of data
    organized_fnames_hdf5 = [GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='hdf5')
                             for snap in range(NUM_SNAPS)]
    org_hd5f_exists = _all_exist(organized_fnames_hdf5)
    if verbose and org_hd5f_exists:
        print("All 'organized' details hdf5 files exist.")

    if reconvert or not org_hd5f_exists:
        if verbose: print("Converting txt files to hdf5")
        convert_txt_to_hdf5(run, verbose)
        if verbose: print("Conversion complete after {}".format(datetime.now()-beg))

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
    num_deleted = 0
    for ii, raw_fn in enumerate(raw_fnames):
        raw_lines = []
        raw_scales = []
        # raw_ids = []
        last_scale = 0.0
        # Load all lines and entry scale-factors from raw details file
        for dline in open(raw_fn):
            # Extract scale-factor from line
            det_scale = DTYPE.SCALAR(dline.split()[1])
            # Extract ID number from line
            # det_id = DTYPE.ID(dline.split()[0].split("BH=")[-1])
            # If the times go backwards, simulation was restarted.  erase (previous) overlap segment
            if det_scale < last_scale:
                # Find the beginning of the overlap
                time_mask = (det_scale < raw_scales) | np.isclose(raw_scales, det_scale)
                # ids_mask = (det_id == raw_ids)
                # bads = np.where(time_mask & ids_mask)[0]
                bads = np.where(time_mask)[0]
                if bads.size:
                    for idx in reversed(bads):
                        del raw_lines[idx]
                        del raw_scales[idx]
                        # del raw_ids[idx]
                    num_deleted += bads.size

            raw_lines.append(dline)
            raw_scales.append(det_scale)
            # raw_ids.append(det_id)
            last_scale = det_scale

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
            print(" - - - {:5d}/{} = {:.4f} after {}; {:.3e} lines written, {:.3e} deleted".format(
                ii, num_raw, ii/num_raw, dur, num_lines, num_deleted))

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
            print(" - {:3d} BH: {:5d}, entries: {:.2e}, median per: {:5.0f}."
                  " Total: {:.2e} entries after {}".format(
                      snap, q_ids.size, id.size, np.median(q_counts), tot_num_entries, dur))

    return


def combine_downsample_and_mergers_hdf5(run, verbose=True, error_in_aft=False):
    fnames_hdf5 = [GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='hdf5')
                   for snap in range(NUM_SNAPS)]
    if not _all_exist(fnames_hdf5):
        raise RuntimeError("Organized details files dont exist (e.g. '{}')".format(fnames_hdf5[0]))

    # Times at which to store details entries
    target_scales = GET_SUBBOX_TIMES(run)
    if verbose: print(" - {} Target times.".format(target_scales.size))

    # New details output file
    dets_fname = GET_OUTPUT_DETAILS_FILENAME(run)
    with h5py.File(dets_fname, 'w') as dets:
        if verbose: print(" - Saving Details to '{}'".format(dets_fname))
        # Write Meta-Data
        # Create Header with meta/summary data
        head = dets.create_group('Header')
        head.attrs['script'] = str(__file__)
        head.attrs['script_version'] = str(__version__)
        head.attrs['created'] = str(datetime.now().ctime())
        head.attrs['simulation'] = 'Illustris-{}'.format(run)
        head.attrs['target_times'] = target_scales
        # head.attrs['num_entries'] = id.size
        # head.attrs['num_blackholes'] = q_ids.size

        # Load mergers file
        mergs = h5py.File(GET_MERGERS_RAW_FILENAME(run, type='hdf5'), 'r')
        mscales = mergs[MERGERS.SCALE]
        m_mass_in = mergs[MERGERS.MASS_IN]
        m_ids_in = mergs[MERGERS.ID_IN]
        m_ids_out = mergs[MERGERS.ID_OUT]
        num_mergers = mscales.size
        # Create merger-details file
        mdets = h5py.File(GET_MERGERS_DETAILS_FILENAME(run), 'w')
        if verbose:
            print(" - Loaded {} mergers from '{}'".format(num_mergers, mergs.filename))
            print(" - Saving Merger-Details to: '{}'".format(mdets.filename))

        # Storage for merger-details
        md_id    = np.zeros([num_mergers, 3], dtype=DTYPE.ID)
        md_scale = np.zeros([num_mergers, 3], dtype=DTYPE.SCALAR)
        md_mass  = np.zeros([num_mergers, 3], dtype=DTYPE.SCALAR)
        md_mdot  = np.zeros([num_mergers, 3], dtype=DTYPE.SCALAR)
        md_rho   = np.zeros([num_mergers, 3], dtype=DTYPE.SCALAR)
        md_cs    = np.zeros([num_mergers, 3], dtype=DTYPE.SCALAR)

        # Make initial guess for how many entries there will be (resized later as-needed)
        BUF_SIZE = int(2e6)
        id    = np.zeros(BUF_SIZE, dtype=DTYPE.ID)
        scale = np.zeros(BUF_SIZE, dtype=DTYPE.SCALAR)
        mass  = np.zeros(BUF_SIZE, dtype=DTYPE.SCALAR)
        mdot  = np.zeros(BUF_SIZE, dtype=DTYPE.SCALAR)
        rho   = np.zeros(BUF_SIZE, dtype=DTYPE.SCALAR)
        cs    = np.zeros(BUF_SIZE, dtype=DTYPE.SCALAR)
        det_ids = []    # ID numbers for BHs with some entries already
        num_det_ids = 0

        beg_all = datetime.now()
        count = 0
        match_count = {val: 0 for val in _MTYPE._VALS()}
        # for snap in range(NUM_SNAPS):
        for snap in range(115, NUM_SNAPS):
            beg = datetime.now()
            with h5py.File(fnames_hdf5[snap], 'r') as h5file:
                s_num_entries = h5file['Header'].attrs['num_entries']
                # If there are no details in this snapshot, skip
                if not s_num_entries:
                    continue

                # Find BH IDs in next snapshot ('None' if this is the last snapshot)
                if snap < NUM_SNAPS-1:
                    with h5py.File(fnames_hdf5[snap+1], 'r') as next_h5file:
                        next_q_ids = next_h5file[DETAILS.UNIQUE_IDS][:]
                else:
                    next_q_ids = None

                # Find which target times are covered in this snapshot
                first = h5file['Header'].attrs['time_first']
                last = h5file['Header'].attrs['time_last']
                targets = target_scales[(target_scales >= first) & (target_scales <= last)]

                # Load information about unique BHs
                q_ids = h5file[DETAILS.UNIQUE_IDS]
                q_first = h5file[DETAILS.UNIQUE_FIRST]
                q_num = h5file[DETAILS.UNIQUE_NUM_PER]
                print("q_ids.size = {:.4e} ({})".format(q_ids.size, datetime.now()-beg))

                # Merger-Details : Find BH mergers in this snapshot
                # -------------------------------------------------
                # Find the indices of 'unique' details IDs matching BH from which mergers numbers
                #    `q_locs_` are unique details IDs (corresponding to `q_ids`...)
                #    `m_locs_` are merger numbers
                #    'in' are matches for the 'in' BH, and 'out' are matches for the 'out' BH
                beg = datetime.now()
                q_locs_in, m_locs_in, snap_mergers_in = _unique_locs_for_mergers(
                    mscales, m_ids_in, q_ids, first, last)
                q_locs_out, m_locs_out, snap_mergers_out = _unique_locs_for_mergers(
                    mscales, m_ids_out, q_ids, first, last)
                # Mergers matching w/ 'in' *or* 'out' BH
                snap_mergers_one = set(snap_mergers_in).union(snap_mergers_out)
                # Mergers matching w/ 'in' *and* 'out' BH
                snap_mergers_both = set(snap_mergers_in).intersection(snap_mergers_out)
                print("Mergers matched after {}".format(datetime.now()-beg))
                beg = datetime.now()

                # 'in' BH
                for ql, mm in zip(q_locs_in, m_locs_in):
                    mtime = mscales[mm]
                    bef, aft = _bef_aft_scales_in(ql, mtime, h5file)
                    # Store this 'before' entry if its the first one, or its better
                    md_loc = (mm, 0)
                    if ((bef is not None and
                         (np.isclose(md_scale[md_loc], 0.0) or
                          h5file[DETAILS.SCALE][bef] > md_scale[md_loc]))):
                        _store_dets_from_hdf5(bef, h5file, md_loc,
                                              md_id, md_scale, md_mass, md_mdot, md_rho, md_cs)
                    # Raise error if 'in' BH has an 'after' entry.
                    if aft is not None and error_in_aft:
                        print("Snap {}, Merger {}, time = {}".format(snap, mm, mtime))
                        raise RuntimeError("'in' has `aft` = {}, scale = {}".format(
                            aft, h5file[DETAILS.SCALE][aft]))

                # 'out' BH
                for ql, mm in zip(q_locs_out, m_locs_out):
                    mtime = mscales[mm]
                    bef, aft, match_type = _bef_aft_scales_out(ql, m_mass_in[mm], mtime, h5file)
                    match_count[match_type] += 1

                    # Store this 'before' entry if its the first one, or its better
                    md_loc = (mm, 1)
                    if ((bef is not None and
                         (np.isclose(md_scale[md_loc], 0.0) or
                          h5file[DETAILS.SCALE][bef] > md_scale[md_loc]))):
                        _store_dets_from_hdf5(bef, h5file, md_loc,
                                              md_id, md_scale, md_mass, md_mdot, md_rho, md_cs)

                    md_loc = (mm, 2)
                    if ((aft is not None and
                         (np.isclose(md_scale[md_loc], 0.0) or
                          h5file[DETAILS.SCALE][aft] > md_scale[md_loc]))):
                        _store_dets_from_hdf5(aft, h5file, md_loc,
                                              md_id, md_scale, md_mass, md_mdot, md_rho, md_cs)

                print("Mergers matches stored after {}".format(datetime.now()-beg))
                print("Match Types:")
                nums = []
                names = []
                count_tot = 0
                for key, val in zip(_MTYPE._KEYS, _MTYPE._VALS()):
                    num = match_count[val]
                    nums.append(num)
                    count_tot += num
                    names.append("{}".format(key))

                for num, nam in zip(nums, names):
                    print("\t{}: {} = {:.4f}".format(nam, num, num/count_tot))

                # sys.exit(2523)
                # continue

                print("Storing details for {} BH".format(q_ids.size))
                beg = datetime.now()
                # Details : for each unique BH, store downsampled details
                # -------------------------------------------------------
                for qid, qf, qn in zip(q_ids, q_first, q_num):
                    lo = qf
                    hi = qf+qn
                    # Store the details entries nearest in time to the target times
                    src = lo + _args_nearest(h5file[DETAILS.SCALE][lo:hi], targets)
                    # Select only unique entries
                    src = np.unique(src)
                    # If this is the first time seeing the BH, make sure first entry is stored
                    if src[0] != lo and num_det_ids > 0:
                        if det_ids[np.searchsorted(det_ids, qid).clip(max=num_det_ids-1)] != qid:
                            src = np.append(lo, src)
                    # If this BH not in the next snapshot (or this is last snap), store last entry
                    if src[-1] != hi-1:
                        if (next_q_ids is None or
                            det_ids[np.searchsorted(next_q_ids, qid).clip(
                                max=num_det_ids-1)] != qid):
                            src = np.append(src, hi)

                    # Resize arrays if we reach edge
                    if count+src.size >= id.size:
                        id = np.pad(id, (0, BUF_SIZE), mode='constant', constant_values=0)
                        scale = np.pad(scale, (0, BUF_SIZE), mode='constant', constant_values=0)
                        mass = np.pad(mass, (0, BUF_SIZE), mode='constant', constant_values=0)
                        mdot = np.pad(mdot, (0, BUF_SIZE), mode='constant', constant_values=0)
                        rho = np.pad(rho, (0, BUF_SIZE), mode='constant', constant_values=0)
                        cs = np.pad(cs, (0, BUF_SIZE), mode='constant', constant_values=0)

                    des = slice(count, count+src.size)
                    _store_dets_from_hdf5(src, h5file, des, id, scale, mass, mdot, rho, cs)
                    count += src.size

                det_ids = np.unique(np.append(det_ids, q_ids))
                num_det_ids = det_ids.size
                print("Details stored after {}".format(datetime.now()-beg))

            if verbose and count:
                n_one = len(snap_mergers_one)
                n_both = len(snap_mergers_both)
                print(" - - {} after {}: Lines = {}, Blackholes = {}"
                      ", Mergers = {}, {} (one, both, none)".format(snap, datetime.now()-beg_all,
                                                                    count, num_det_ids,
                                                                    n_one, n_both))

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

'''
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
'''

'''
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
'''

'''
def get_out_dir(run, output_dir=None):
    """
    /n/home00/lkelley/ghernquistfs1/illustris/data/%s/blackhole/details/
    """
    if output_dir is None:
        output_dir = os.path.join(_PROCESSED_DIR % (_ILLUSTRIS_RUN_NAMES[run]), _DEF_OUTPUT_DIR, '')
    return output_dir
'''


def _store_dets_from_hdf5(h5_loc, h5dat, save_loc, id, scale, mass, mdot, rho, cs):
    if np.iterable(h5_loc):
        _loc = list(h5_loc)
    else:
        _loc = h5_loc
    id[save_loc] = h5dat[DETAILS.ID][_loc]
    scale[save_loc] = h5dat[DETAILS.SCALE][_loc]
    mass[save_loc] = h5dat[DETAILS.MASS][_loc]
    mdot[save_loc] = h5dat[DETAILS.MDOT][_loc]
    rho[save_loc] = h5dat[DETAILS.RHO][_loc]
    cs[save_loc] = h5dat[DETAILS.CS][_loc]


def _args_nearest(options, targets):
    idx = np.searchsorted(options, targets, side="left").clip(max=options.size-1)
    dist_lo = np.fabs(targets - options[idx-1])
    dist_hi = np.fabs(targets - options[idx])
    mask = (idx > 0) & ((idx == len(options)) | (dist_lo < dist_hi))
    idx = idx - mask
    return idx


def _bef_med_aft(vals, targ):
    idx = np.searchsorted(vals, targ, side="left")
    lo = idx-1
    hi = idx
    if lo < 0: lo = 0
    if idx < vals.size and np.isclose(vals[idx], targ):
        me = idx
        if lo == me:
            lo = []
            hi = me + 1
        if hi == me:
            hi = me + 1
    else:
        me = []

    if hi > vals.size-1: hi = vals.size-1

    return lo, me, hi


def _bef_aft_scales_in(id_loc, mtime, h5file):
    i0 = h5file[DETAILS.UNIQUE_FIRST][id_loc]
    i1 = i0 + h5file[DETAILS.UNIQUE_NUM_PER][id_loc]

    dscales = h5file[DETAILS.SCALE][i0:i1]
    lo, me, hi = _bef_med_aft(dscales, mtime)
    # For the 'in' BH, if there is an entry at time of merger, must be before
    if me:
        bef = me
        aft = hi
    else:
        bef = lo
        aft = hi

    # Both `bef` and `aft` can be before/after `mtime` if `mtime` ourside range
    if dscales[bef] > mtime:
        bef = None
    else:
        bef += i0
    if dscales[aft] < mtime:
        aft = None
    else:
        aft += i0

    return bef, aft


def _bef_aft_scales_out(id_loc, m_mass, mtime, h5file):
    i0 = h5file[DETAILS.UNIQUE_FIRST][id_loc]
    i1 = i0 + h5file[DETAILS.UNIQUE_NUM_PER][id_loc]

    dscales = h5file[DETAILS.SCALE][i0:i1]
    d_mass = h5file[DETAILS.MASS][i0:i1]
    lo, me, hi = _bef_med_aft(dscales, mtime)
    # For the 'out' BH, determine if exact matching time is before or after merger based on masses
    if me:
        delta_mass = np.diff(d_mass[[lo, me, hi]]) - m_mass
        if delta_mass[0] >= 0.0 and delta_mass[1] < 0.0:
            bef = lo
            aft = me
            match_type = _MTYPE.M_BEF_D
        elif delta_mass[0] < 0.0 and delta_mass[1] >= 0.0:
            bef = me
            aft = hi
            match_type = _MTYPE.D_BEF_M
        elif delta_mass[0] >= 0.0 and delta_mass[1] >= 0.0:
            bef = lo
            aft = hi
            match_type = _MTYPE.UNKNOWN
        elif delta_mass[0] < 0.0 and delta_mass[1] < 0.0:
            match_type = _MTYPE.CONFLICT
            return None, None, match_type
        else:
            raise RuntimeError("WHATS HAPPENING?!?!  delta_mass = {}".format(delta_mass))
    else:
        delta_mass = np.diff(d_mass[[lo, hi]]) - m_mass
        if delta_mass > 0.0:
            bef = lo
            aft = hi
            match_type = _MTYPE.ASYNC
        else:
            match_type = _MTYPE.ASYNC_CONFLICT
            return None, None, match_type

    # Both `bef` and `aft` can be before/after `mtime` if `mtime` ourside range
    if dscales[bef] > mtime:
        bef = None
    else:
        bef += i0
    if dscales[aft] < mtime:
        aft = None
    else:
        aft += i0

    return bef, aft, match_type


def _unique_locs_for_mergers(mscales, mids, unique_ids, first, last, pad=0.05):
    """Find which mergers match which unique (details) ID numbers for this snapshot.
    """
    # Find mergers occuring in (or near: `pad`) this snapshot
    _pad = pad * (last - first)
    m_inds_snap = np.where((mscales + _pad >= first) & (mscales - _pad <= last))[0]
    # Get the IDs for these mergers
    m_ids = mids[list(m_inds_snap)]
    # Sort required to index hdf5
    sort_m_ids = np.argsort(m_ids)
    # Find the locations in details unique IDs to place 'in' IDs
    d_locs_m_ids = np.searchsorted(unique_ids, m_ids[sort_m_ids]).clip(max=unique_ids.size-1)
    # Find which of those locations correspond to actual matches
    match = np.where(unique_ids[:][list(d_locs_m_ids)] == m_ids[sort_m_ids])[0]
    # Merger numbers with IDs matching details-unique-ID entries
    match_mergers = m_inds_snap[sort_m_ids[match]]
    # Details-unique-ID entries matching merger IDs
    match_unique_locs = d_locs_m_ids[list(match)]
    # print(q_ids[match_unique_locs[0]], mids[match_mergers[0]])
    return match_unique_locs, match_mergers, m_inds_snap
