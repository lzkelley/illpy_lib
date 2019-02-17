"""
"""

import os
import shutil
from datetime import datetime

import numpy as np
import h5py

import zcode.inout as zio

# from illpy_lib.constants import DTYPE, NUM_SNAPS
from ..constants import DTYPE, NUM_SNAPS

from .bh_constants import DETAILS
from . import Core

VERSION = 0.3                                    # Version of details

_DEF_PRECISION = -8                               # Default precision


def main():

    core = Core()  # sets=dict(RECREATE=True))
    log = core.log

    log.info("details.main()")
    print(log.filename)

    beg = datetime.now()

    # Organize Details by Snapshot Time; create new, temporary ASCII Files
    reorganize(core)  # run, loadsave=loadsave, verbose=verbose)

    # Create Dictionary Details Files
    reformat(core)  # run, loadsave=loadsave, verbose=verbose)

    end = datetime.now()
    log.info("Done after '{}'".format(end-beg))

    return


def reorganize(core=None):
    core = Core.load(core)
    log = core.log

    log.debug("details.reorganize()")

    RUN = core.sets.RUN_NUM

    # temp_fnames = [constants.GET_DETAILS_TEMP_FILENAME(run, snap) for snap in range(NUM_SNAPS)]
    temp_fnames = [core.paths.fname_details_temp_snap(snap, RUN) for snap in range(NUM_SNAPS)]

    loadsave = (not core.sets.RECREATE)
    print("core.sets.RECREATE = {}, loadsave = {}".format(core.sets.RECREATE, loadsave))

    # Check if all temp files already exist
    if loadsave:
        temps_exist = [os.path.exists(tfil) for tfil in temp_fnames]
        if all(temps_exist):
            log.info("All temp files exist.")
        else:
            bad = np.argmin(temps_exist)
            bad = temp_fnames[bad]
            log.warning("Temp files do not exist e.g. '{}'".format(bad))
            loadsave = False

    # If temp files dont exist, or we WANT to redo them, then create temp files
    if not loadsave:
        log.debug("Finding Illustris BH Details files")
        # Get Illustris BH Details Filenames
        # raw_fnames = constants.GET_ILLUSTRIS_BH_DETAILS_FILENAMES(run, verbose)
        raw_fnames = core.paths.fnames_details_input
        if len(raw_fnames) < 1:
            log.raise_error("Error no dets files found!!")

        # Reorganize into temp files
        log.warning("Reorganizing details into temporary files")
        _reorganize_files(core, raw_fnames, temp_fnames)

    # Confirm all temp files exist
    temps_exist = all([os.path.exists(tfil) for tfil in temp_fnames])

    # If files are missing, raise error
    if not temps_exist:
        print(("Temporary Files still missing!  '{:s}'".format(temp_fnames[0])))
        raise RuntimeError("Temporary Files missing!")

    return temp_fnames


def _reorganize_files(core, raw_fnames, temp_fnames):

    log = core.log
    log.debug("details._reorganize_files()")

    import illpy_lib.illcosmo
    cosmo = illpy_lib.illcosmo.cosmology.Cosmology()
    snap_scales = cosmo.scales()

    temps = [zio.modify_filename(tt, prepend='_') for tt in temp_fnames]

    # Open new ASCII, Temp dets files
    #    Make sure path is okay
    zio.check_path(temps[0])
    # Open each temp file
    # temp_files = [open(tfil, 'w') for tfil in temp_fnames]
    temp_files = [open(tfil, 'w') for tfil in temps]

    num_temp = len(temp_files)
    num_raw  = len(raw_fnames)
    log.info("Organizing {:d} raw files into {:d} temp files".format(num_raw, num_temp))

    prec = _DEF_PRECISION
    all_num_lines_in = 0
    all_num_lines_out = 0

    # Iterate over all Illustris Details Files
    # ----------------------------------------
    for ii, raw in enumerate(core.tqdm(raw_fnames, desc='Raw files')):
        log.debug("File {}: '{}'".format(ii, raw))
        lines = []
        scales = []
        # Load all lines and entry scale-factors from raw dets file
        for dline in open(raw):
            lines.append(dline)
            # Extract scale-factor from line
            detScale = DTYPE.SCALAR(dline.split()[1])
            scales.append(detScale)

        # Convert to array
        lines = np.array(lines)
        scales = np.array(scales)
        num_lines_in = scales.size

        # If file is empty, continue
        if num_lines_in == 0:
            log.debug("\tFile empty")
            continue

        log.debug("\tLoaded {}".format(num_lines_in))

        # Round snapshot scales to desired precision
        scales_round = np.around(snap_scales, -prec)

        # Find snapshots following each entry (right-edge) or equal (include right: 'right=True')
        #    `-1` to put binaries into the snapshot UP-TO that scalefactor
        snap_nums = np.digitize(scales, scales_round, right=True) - 1

        # For each Snapshot, write appropriate lines
        num_lines_out = 0
        for snap in range(NUM_SNAPS):
            inds = (snap_nums == snap)
            num_lines_out_snap = np.count_nonzero(inds)
            if num_lines_out_snap == 0:
                continue

            temp_files[snap].writelines(lines[inds])
            log.debug("\t\tWrote {} lines to snap {}".format(num_lines_out_snap, snap))
            num_lines_out += num_lines_out_snap

        if num_lines_out != num_lines_in:
            log.error("File {}, '{}'".format(ii, raw))
            log.raise_error("Wrote {} lines, loaded {} lines!".format(num_lines_out, num_lines_in))

        all_num_lines_in += num_lines_in
        all_num_lines_out += num_lines_out

    # Close out dets files
    tot_size = 0.0
    log.info("Closing files, checking sizes")

    for ii, newdf in enumerate(temp_files):
        newdf.close()
        tot_size += os.path.getsize(newdf.name)

    ave_size = tot_size/(1.0*len(temp_files))
    size_str = zio.bytes_string(tot_size)
    ave_size_str = zio.bytes_string(ave_size)
    log.info("Total temp size = '{}', average = '{}'".format(size_str, ave_size_str))

    log.info("Input lines = {:d}, Output lines = {:d}".format(all_num_lines_in, all_num_lines_out))
    if (all_num_lines_in != all_num_lines_out):
        log.raise_error("input lines {}, does not match output lines {}!".format(
            all_num_lines_in, all_num_lines_out))

    log.info("Renaming temporary files...")
    for ii, (aa, bb) in enumerate(zip(temps, temp_fnames)):
        if ii == 0:
            log.debug("'{}' ==> '{}'".format(aa, bb))
        shutil.move(aa, bb)

    return


def reformat(core=None):
    core = Core.load(core)
    log = core.log

    log.debug("details.reformat()")

    out_fnames = [core.paths.fname_details_snap(snap) for snap in range(NUM_SNAPS)]

    loadsave = (not core.sets.RECREATE)

    # Check if all save files already exist, and correct versions
    if loadsave:
        out_exist = [os.path.exists(sfil) for sfil in out_fnames]
        if all(out_exist):
            log.info("All output files exist")
        else:
            bad = np.argmin(out_exist)
            bad = out_fnames[bad]
            log.warning("Output files do not exist e.g. '{}'".format(bad))
            loadsave = False

    # Re-convert files
    if (not loadsave):
        log.warning("Processing temporary files")
        temp_fnames = [core.paths.fname_details_temp_snap(snap) for snap in range(NUM_SNAPS)]
        for snap in core.tqdm(range(NUM_SNAPS)):
            temp = temp_fnames[snap]
            out = out_fnames[snap]
            _reformat_to_hdf5(core, snap, temp, out)

    # Confirm save files exist
    out_exist = [os.path.exists(sfil) for sfil in out_fnames]

    # If files are missing, raise error
    if (not all(out_exist)):
        log.raise_error("Output files missing!")

    return out_fnames


def _reformat_to_hdf5(core, snap, temp_fname, out_fname):
    """
    """

    log = core.log
    log.debug("details._reformat_to_hdf5()")
    log.info("Snap {}, {} ==> {}".format(snap, temp_fname, out_fname))

    loadsave = (not core.sets.RECREATE)

    # Make Sure Temporary Files exist, Otherwise re-create them
    if (not os.path.exists(temp_fname)):
        log.raise_error("Temp file '{}' does not exist!".format(temp_fname))

    # Try to load from existing save
    if loadsave:
        if os.path.exists(out_fname):
            log.info("\tOutput file '{}' already exists.".format(out_fname))
            return

    # Load dets from ASCII File
    vals = _load_bhdetails_ascii(temp_fname)
    ids, scales, masses, mdots, rhos, cs = vals
    # Sort by ID number, then by scale-factor
    sort = np.lexsort((scales, ids))
    vals = [vv[sort] for vv in vals]
    ids, scales, masses, mdots, rhos, cs = vals

    # Find unique ID numbers, their first occurence indices, and the number of occurences
    u_ids, u_inds, u_counts = np.unique(ids, return_index=True, return_counts=True)
    num_unique = u_ids.size
    log.info("\tunique IDs: {}".format(zio.frac_str(num_unique, ids.size)))

    with h5py.File(out_fname, 'w') as out:
        out.attrs[DETAILS.RUN] = core.sets.RUN_NUM
        out.attrs[DETAILS.SNAP] = snap
        out.attrs[DETAILS.NUM] = len(ids)
        out.attrs[DETAILS.CREATED] = str(datetime.now().ctime())
        out.attrs[DETAILS.VERSION] = VERSION

        out.create_dataset(DETAILS.IDS, data=ids)
        out.create_dataset(DETAILS.SCALES, data=scales)
        out.create_dataset(DETAILS.MASSES, data=masses)
        out.create_dataset(DETAILS.MDOTS, data=mdots)
        out.create_dataset(DETAILS.RHOS, data=rhos)
        out.create_dataset(DETAILS.CS, data=cs)

        out.create_dataset('unique_ids', data=u_ids)
        out.create_dataset('unique_indices', data=u_inds)
        out.create_dataset('unique_counts', data=u_counts)

    size_str = zio.get_file_size(out_fname)
    log.info("\tSaved snap {} to '{}', size {}".format(snap, out_fname, size_str))

    return


def _load_bhdetails_ascii(temp_fname):
    # Files have some blank lines in them... Clean
    with open(temp_fname, 'r') as temp:
        lines = temp.readlines()

    nums = len(lines)

    # Allocate storage
    ids    = np.zeros(nums, dtype=DTYPE.ID)
    scales = np.zeros(nums, dtype=DTYPE.SCALAR)
    masses = np.zeros(nums, dtype=DTYPE.SCALAR)
    mdots  = np.zeros(nums, dtype=DTYPE.SCALAR)
    rhos   = np.zeros(nums, dtype=DTYPE.SCALAR)
    cs     = np.zeros(nums, dtype=DTYPE.SCALAR)

    count = 0
    # Iterate over lines, storing only those with content (should be all)
    for lin in lines:
        lin = lin.strip()
        if (len(lin) > 0):
            tid, tim, mas, dot, rho, tcs = _parse_bhdetails_line(lin)
            ids[count] = tid
            scales[count] = tim
            masses[count] = mas
            mdots[count] = dot
            rhos[count] = rho
            cs[count] = tcs

            count += 1

    # Trim excess (shouldn't be needed)
    if (count != nums):
        trim = np.s_[count:]

        ids    = np.delete(ids, trim)
        scales = np.delete(scales, trim)
        masses = np.delete(masses, trim)
        mdots  = np.delete(mdots, trim)
        rhos   = np.delete(rhos, trim)
        cs     = np.delete(cs, trim)

    return ids, scales, masses, mdots, rhos, cs


def _parse_bhdetails_line(instr):
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
    # First element is 'BH=#', trim to just the id number
    args[0] = args[0].split("BH=")[-1]
    idn  = DTYPE.ID(args[0])
    time = DTYPE.SCALAR(args[1])
    mass = DTYPE.SCALAR(args[2])
    mdot = DTYPE.SCALAR(args[3])
    rho  = DTYPE.SCALAR(args[4])
    cs   = DTYPE.SCALAR(args[5])
    return idn, time, mass, mdot, rho, cs


def load_details(snap, core=None):
    """
    """
    core = Core.load(core)
    log = core.log

    log.debug("details.load_details()")

    fname = core.paths.fname_details_snap(snap)
    log.debug("Filename for snap {}: '{}'".format(snap, fname))

    err = []
    with h5py.File(fname, 'r') as data:
        if data.attrs[DETAILS.RUN] != core.sets.RUN_NUM:
            err.append("Run numbers do not match!")

        if data.attrs[DETAILS.SNAP] != snap:
            err.append("Snap numbers do not match!")

        if data.attrs[DETAILS.VERSION] != VERSION:
            err.append("Unexpected version number!")

        if len(err) > 0:
            err = ",  ".join(err)
            log.raise_error(err)

        log.debug("File saved at '{}'".format(data.attrs[DETAILS.CREATED]))

        ids = data[DETAILS.IDS][:]
        scales = data[DETAILS.SCALES][:]
        masses = data[DETAILS.MASSES][:]
        mdots = data[DETAILS.MDOTS][:]
        rhos = data[DETAILS.RHOS][:]
        cs = data[DETAILS.CS][:]

    return ids, scales, masses, mdots, rhos, cs
