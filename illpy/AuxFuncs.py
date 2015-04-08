# ==================================================================================================
# AuxFuncs.py - Auxilliary functions
# -----------
#
#
# Functions
# ---------
# - Merger Files
#   + getMergerFiles(target)     : for a given target 'run', get the list of merger filenames
#   + loadMergerFile(mfile)      : load the given merger file into a list of Merger objects
#   + loadAllMergers(target)     : load all mergers from the target run
#
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================

from __future__ import division
import numpy as np
import warnings
import os
import sys
import types
from datetime import datetime
import matplotlib                  as mpl
from matplotlib import pyplot      as plt

from glob import glob
import datetime

from Constants import *

DT_THRESH = 1.0e-5                                                                                  # Frac diff b/t times to accept as equal



###  ======================================  ###
###  ==========  SNAPSHOT FILES  ==========  ###
###  ======================================  ###



'''



###  =========================================  ###
###  =============  GROUP FILES  =============  ###
###  =========================================  ###

def getGroupFilename(snap_num, run_num=lyze.TARGET_RUN):
    """
    Given a run number and snapshot/catalog number (i.e. output time), construct group filename.

    input
    -----
    run_num  : IN [int] simulation run number, i.e. 3 for Illustris-3
    snap_num : IN [int] snapshot number,       i.e. 100 for snapdir_100

    output
    ------
    return     filename : [str]
    """
    groupName = (lyze.GROUP_CAT_DIRS % snap_num) + (lyze.GROUP_CAT_FILENAMES % snap_num)
    filename = lyze.RUN_DIRS[run_num] + groupName
    return filename



def constructOffsetTables(gcat):
    """
    Construct a table of particle offsets for each halo/subhalo.

    Based on code from filter.Halo.reset()

    Note that particles are ordered as follows in each:

        HALO    SUBHALO       PARTICLE
      | ==============================
      |    0 -------------------------     <-- first halo
      |               0 --------------     <--   "     "  's subhalo
      |                              0     <--   "     "        "     's first particle
      |                            ...
      |                        NS_0 -1     <-- number of particles in subhalo 0, minus 1
      |               1 ---------------
      |                           NS_0
      |                            ...
      |                 NS_0 + NS_1 -1
      |
      |               ...           ...
      |
      |         NS_0 -1 ---------------    <-- number of subhalos in halo 0, minus 1
      |                            ...
      |            NONE ---------------    <-- particles NOT IN A SUBHALO (end of halo block)
      |                            ...
      |                         NH_0-1     <-- number of particles in halo 0, minus 1
      |    1 --------------------------
      |
      |  ...         ...           ...
      |
      |  M-1 --------------------------    <-- number of Halos
      |
      |              ...           ...
      |
      | NONE --------------------------    <-- particles NOT IN A HALO    (end of entire file)
      |
      |              ...           ...
      | ===============================


    """

    DTYPE       = np.uint64

    numHalos    = gcat.npart_loaded[0]                                                          # Number of halos
    numSubhalos = gcat.npart_loaded[1]                                                          # Number of subhalos


    # Initialize offset tables for halos and subhalos; dimension for each particle type
    #    last element corresponds to the EOF offset,
    #    i.e. 1+NTOT  where NTOT is the total number of particles
    halo_offsets         = np.zeros( (numHalos+1   , 6), dtype=DTYPE )
    subhalo_offsets      = np.zeros( (numSubhalos+1, 6), dtype=DTYPE )
    halo_subhalo_offsets = np.zeros(  numHalos+1       , dtype=DTYPE )

    # offset_table
    # ------------
    #    One entry for first particles in each subhalo, particles in each halo and NO subhalo,
    #        and a single entry for particles with NO halo and NO subhalo
    #    Each entry is [ HALO, SUBHALO, PART0, ..., PART5 ]
    #        when there is no (sub)halo, the number will be '-1'
    #
    offset_table         = np.zeros( [numHalos+numSubhalos+1, 8], type=UINT)

    ### Determine halo offsets ###

    # For each particle, the offset for each Halo is the number of particles in previous halos
    halo_offsets[1:,:] = np.cumsum(gcat.group.GroupLenType[:,:], axis=0, dtype=DTYPE)

    ### Determine subhalo offsets ###

    subh = 0
    offs = 0
    cumPartTypes = np.zeros(6, dtype=UINT)
    cumHaloPartTypes = np.zeros(6, dtype=UINT)
    # Iterate over each Halo
    for ii in range(numHalos):
        cumHaloPartTypes += gcat.group.GroupLenType[ii,:]                                           # Add the number of particles in this halo
        # Iterate over each Subhalo in halo 'ii'
        for jj in range(gcat.group.GroupNsub[ii]):
            ### Add entry for each subhalo ###
            offset_table[offs] = np.append([ ii, subh ], cumPartTypes)
            subh += 1                                                                               # Increment subhalo number
            offs += 1                                                                               # Increment offset entry
            cumPartTypes += gcat.subhalo.SubhaloLenType[subh]                                       # Add particles in this subhalo

        # If there are more particles than in subhalos
        if( cumPartTypes != cumHaloPartTypes ):
            ### Add Entry for particles with NO subhalo ###
            offset_table[offs] = np.append([ii,-1], cumPartTypes)
            offs += 1                                                                               # Increment offset entry

        cumPartTypes = cumHaloPartTypes                                                             # Increment particle numbers to include this halo


    ### Add entry for end of all halo particles / start of particles with NO halo ###
    offset_table[offs] = np.append([-1,-1], cumPartTypes)

    subh = 0
    # Iterate over all halos
    for ii in np.arange(numHalos):
        # If this halo has subhalos, incorporate them
        if gcat.group.GroupNsubs[ii] > 0:

            # Zeroth subhalo has same offset as host halo (but different lengths)
            tmp = halo_offsets[ii,:]
            subhalo_offsets[subh,:] = tmp

            sub1 = subh + 1                                                                     # First subhalo index
            sub2 = subh + gcat.group.GroupNsubs[ii] + 1                                         # Last  subhalo index

            # To each subhalo after zeroth, add sum of particles up to previous subhalo
            subhalo_offsets[sub1:sub2,:] = (
                tmp +
                np.cumsum(gcat.subhalo.SubhaloLenType[subh:sub2-1,:], axis=0, dtype=DTYPE)
                )

            subh += gcat.group.GroupNsubs[ii]                                                   # Increment to zeroth subhalo of next halo

        halo_subhalo_offsets[ii+1] = ( halo_subhalo_offsets[ii] +
                                       gcat.group.GroupNsubs[ii] )

    # } i

    return halo_offsets, subhalo_offsets, offset_table

'''




###  ======================================  ###
###  =============  PHYSICS  ==============  ###
###  ======================================  ###

def aToZ(a, a0=1.0):
    """ Convert a scale-factor to a redshift """
    z = (a0/a) - 1.0
    return z

def zToA(z, a0=1.0):
    """ Convert a redshift to a scale-factor """
    a = a0/(1.0+z)
    return a



###  ===================================  ###
###  =============  MATH  ==============  ###
###  ===================================  ###


def histogram(args, bins, weights=None, scale=None, ave=False):

    if( ave ):
        assert weights is not None, "To average over each bin, ``weights`` must be provided!"

    hist,edge = np.histogram( args, bins=bins, weights=weights )

    # Find the average of each weighed bin instead.
    if( ave and weights is not None ):
        hist = [ hh/nn if nn > 0 else 0.0
                 for hh,nn in zip(hist,np.histogram( args, bins=bins)[0]) ]

    hist = np.array(hist)

    # Rescale the bin values
    if( scale != None ):
        hist *= scale

    return hist


def extrema(args, nonzero=False):
    if( nonzero ): useMin = np.min( args[np.nonzero(args)] )
    else:          useMin = np.min(args)
    extr = np.array([useMin, np.max(args)])
    return extr


def space(args, num=100, log=True):

    extr = extrema(args)
    if( log ): bins = np.logspace( *np.log10(extr), num=num)
    else:      bins = np.linspace( *extr,           num=num)

    return bins


def frexp10(xx):
    """
    Decompose the given number into a mantissa and exponent for scientific notation.
    """
    exponent = int(np.log10(xx))
    mantissa = xx / np.power(10.0, exponent)
    return mantissa, exponent


def nonzeroMin(arr):
    """
    Set all less-than or equal to zero values to the otherwise minimum value.
    """

    fix = np.array(arr)

    good = np.where( fix >  0.0 )[0]
    bad  = np.where( fix <= 0.0 )[0]

    fix[bad] = np.min( fix[good] )
    return fix



def minmax(arr):
    """
    Get the minimum and maximum of the given array
    """
    return np.min(arr), np.max(arr)


def avestd(arr):
    """
    Get the average and standard deviation of the given array.
    """
    return np.average(arr), np.std(arr)


def incrementRollingStats(avevar, count, val):
    """
    Increment a rolling average and stdev calculation with a new value

    avevar   : INOUT [ave, var]  the rolling average and variance respectively
    count    : IN    [int]       the *NEW* count for this bin (including new value)
    val      : IN    [float]     the new value to be included

    return
    ------
    avevar   : [float, float] incremented average and variation

    """
    delta      = val - avevar[0]

    avevar[0] += delta/count
    avevar[1] += delta*(val - avevar[0])

    return avevar


def finishRollingStats(avevar, count):
    """ Finish a rolling average and stdev calculation by find the stdev """

    if( count > 1 ): avevar[1] = np.sqrt( avevar[1]/(count-1) )
    else:            avevar[1] = 0.0

    return avevar



def getMagnitude(vect):
    """ Get the magnitude of a vector of arbitrary length """
    return np.sqrt( np.sum([vv*vv for vv in vect]) )


def isApprox(v1, v2, TOL=1.0e-6):
    """
    Check if two scalars are eqeual to within some tolerance.

    Parameters
    ----------
    v1 : scalar
        first value
    v2 : scalar
        second value
    TOL : scalar
        Fractional tolerance within which to return True

    Returns
    -------
    retval : bool
        True if the (conservative) fractional difference between `v1` and `v2`
        is less than or equal to the tolerance.

    """

    # Find the lesser value to find conservative fraction
    less = np.min([v1,v2])
    # Compute fractional difference
    diff = np.fabs((v1-v2)/less)

    # Compare fractional difference to tolerance
    retval = (diff <= TOL)

    return retval



def groupIndices(arr, bins, right=True):
    selects = []
    nums = len(bins)
    # Iterate over each bin and find the members of array which belong inside
    for ii in range(nums):

        # If bins give *right* edges
        if( right ):
            if( ii > 0 ): inds = np.where((arr > bins[ii-1]) & (arr <= bins[ii]))[0]
            else:         inds = np.where(arr <= bins[ii])[0]

        # If bins give *left* edges
        else:
            if( ii < nums-1 ): inds = np.where((arr >= bins[ii]) & (arr < bins[ii+1]))[0]
            else:              inds = np.where(arr > bins[ii])[0]

        selects.append(inds)

    return selects



def findBins(target, bins, thresh=DT_THRESH):
    """
    Find the array indices (of "bins") bounding the "target"

    If target is outside bins, the missing bound will be 'None'
    low and high will be the same, if the target is almost exactly[*1] equal to a bin

    [*1] : How close counds as effectively the same is set by 'DEL_TIME_THRESH' below

    intput
    ------

    target  : [ ] value to be compared
    bins    : [ ] list of values to compare to the 'target'


    output
    ------

    Return   low  : [int] index below target (or None if none)
             high : [int] index above target (or None if none)
    """

    # deltat  : test whether the fractional difference between two values is less than threshold
    #           This function allows the later conditions to accomodate smaller numerical
    #           differences, between effectively the same value  (e.g.   1.0 vs. 0.9999999999989)
    #
    if( thresh == 0.0 ): deltat = lambda x,y : False
    else               : deltat = lambda x,y : np.abs(x-y)/np.abs(x) <= thresh

    nums   = len(bins)
    # Find bin above (or equal to) target
    high = np.where( (target <= bins) | deltat(target,bins) )[0]
    if( len(high) == 0 ): high = None
    else:
        high = high[0]                                                                              # Select first bin above target
        dhi  = bins[high] - target


    # Find bin below (or equal to) target
    low  = np.where( (target >= bins) | deltat(target,bins) )[0]
    if( len(low)  == 0 ): low  = None
    else:
        low  = low[-1]                                                                              # Select  last bin below target
        dlo  = bins[low] - target

    # Print warning on error
    if( low == None or high == None ):
        print "AuxFuncs.findBins: target = %e, bins = {%e,%e}; low,high = %s,%s !" % \
            ( target, bins[0], bins[-1], str(low), str(high) )
        raise RuntimeError("Could not find bins!")


    return [low,high,dlo,dhi]





def stringArray(arr, format='%.2f'):
    out = [ format % elem for elem in arr ]
    out = "[ " + " ".join(out) + " ]"
    return out

def statusString(count, total, durat=None):
    """
    Return a description of the status and completion of an iteration.

    If ``durat`` is provided it is used as the duration of time that the
    iterations have been going.  This time is used to estimate the time
    to completion.  ``durat`` can either be a `datetime.timedelta` object
    of if it is a scalar (int or float) then it will be converted to
    a `datetime.timedelta` object for string formatting purposes.

    Parameters
    ----------
    count : int, number of iterations completed (e.g. [0...9])
    total : int, total number of iterations (e.g. 10)
    durat : datetime.timedelta OR scalar, (optional, default=None)
    """

    # Calculate Percentage completed
    frac = 1.0*count/(total)
    stat = '%.2f%%' % (100*frac)

    if( durat != None ):
        # Make sure `durat` is a datetime.timedelta object
        if( type(durat) is not datetime.timedelta ): durat = datetime.timedelta(seconds=durat)

        # Calculate time left
        timeLeft = 1.0*durat.total_seconds()*(1.0/frac - 1.0)
        timeLeft = np.max([timeLeft, 0.0])
        timeLeft = datetime.timedelta(seconds=timeLeft)

        # Append to status string
        stat += ' after %s, completion in ~ %s' % (str(durat), str(timeLeft))


    return stat


def bytesString(bytes, precision=1):
    """
    Return a humanized string representation of a number of bytes.

    Arguments
    ---------
    bytes : <scalar>, number of bytes
    precision : <int>, target precision in number of decimal places

    Returns
    -------
    strSize : <string>, human readable size

    Examples
    --------
    >> humanize_bytes(1024*12342,2)
    '12.05 MB'

    """

    abbrevs = (
        (1<<50L, 'PB'),
        (1<<40L, 'TB'),
        (1<<30L, 'GB'),
        (1<<20L, 'MB'),
        (1<<10L, 'kB'),
        (1, 'bytes')
    )

    for factor, suffix in abbrevs:
        if bytes >= factor: break

    # NOTE: for this to work right, must "from __future__ import division" else integer
    strSize = '%.*f %s' % (precision, bytes / factor, suffix)

    return strSize




###  ====================================  ###
###  =============  FILES  ==============  ###
###  ====================================  ###

def combineFiles(inFilenames, outFilename, verbose=False):
    """
    Concatenate the contents of a set of input files into a single output file.

    Arguments
    ---------
    inFilenames : iterable<str>, list of input file names
    outFilename : <str>, output file name
    verbose : <bool> (optional=_VERBOSE), print verbose output

    Returns

    """

    precision = 2


    if( verbose ): print " - - AuxFuncs.combineFiles()"
    # Make sure outfile path exists
    checkPath(outFilename)

    inSize = 0.0
    count  = 0
    nums = len(inFilenames)-1
    if( nums <= 100 ): interval = 1
    else:              interval = np.int( np.floor(nums/100.0) )

    # Open output file for writing
    with open(outFilename, 'w') as outfil:

        # Iterate over input files
        for inname in inFilenames:
            inSize += os.path.getsize(inname)

            if( verbose ):
                if( count % interval == 0 or count == nums):
                    sys.stdout.write('\r - - - %.2f%% Complete' % (100.0*count/nums))

                if( count == nums ): sys.stdout.write('\n')
                sys.stdout.flush()


            # Open input file for reading
            with open(inname, 'r') as infil:

                # Iterate over input file lines
                for line in infil:
                    outfil.write(line)

            # } infil

            count += 1

        # } inname

    # } outfil

    outSize = os.path.getsize(outFilename)

    inStr   = bytesString(inSize, precision)
    outStr  = bytesString(outSize, precision)

    if( verbose ): print " - - - Total input size = %s, output size = %s" % (inStr, outStr)

    return



def dictToNPZ(dataDict, savefile, verbose=False):
    """
    Save the given dictionary to the given npz file.

    If the path to the given filename doesn't already exist, it is created.
    If ``verbose`` is True, the saved file size is printed out.
    """

    # Make sure path to file exists
    checkPath(savefile)

    # Save and confirm
    np.savez(savefile, **dataDict)
    if( not os.path.exists(savefile) ):
        raise RuntimeError("Could not save to file '%s'!!" % (savefile) )

    if( verbose ): print " - - Saved dictionary to '%s'" % (savefile)
    if( verbose ): print " - - - Size '%s'" % ( getFileSize(savefile) )
    return




def getFileSize(fnames, precision=1):
    """
    Return a human-readable size of a file or set of files.

    Arguments
    ---------
    fnames : <string> or list/array of <string>, paths to target file(s)
    precisions : <int>, desired decimal precision of output

    Returns
    -------
    byteStr : <string>, human-readable size of file(s)

    """

    ftype = type(fnames)
    if( ftype is not list and ftype is not np.ndarray ): fnames = [ fnames ]

    byteSize = 0.0
    for fil in fnames: byteSize += os.path.getsize(fil)

    byteStr = bytesString(byteSize, precision)
    return byteStr




def countLines(files, progress=False):
    """ Count the number of lines in the given file """

    # If string, or otherwise not-iterable, convert to list
    if( not iterableNotString(files) ): files = [ files ]

    if( progress ):
        numFiles = len(files)
        if( numFiles < 100 ): interval = 1
        else:                 interval = np.int(np.floor(numFiles/100.0))
        start = datetime.now()

    nums = 0
    # Iterate over each file
    for ii,fil in enumerate(files):
        # Count number of lines
        nums += sum(1 for line in open(fil))

        # Print progresss
        if( progress ):
            now = datetime.now()
            dur = now-start

            statStr = aux.statusString(ii+1, numFiles, dur)
            sys.stdout.write('\r - - - %s' % (statStr))
            sys.stdout.flush()
            if( ii+1 == numFiles ): sys.stdout.write('\n')


    return nums


def iterableNotString(args):
    """
    Check if the arguments is iterable and not a string.
    """

    # if NOT iterable, return false
    if( not np.iterable(args) ): return False
    # if string, return False
    if( isinstance(args, types.StringTypes) ): return False

    return True



def estimateLines(files):
    """ Count the number of lines in the given file """

    if( not np.iterable(files) ): files = [files]

    lineSize = 0.0
    count = 0
    AVE_OVER = 20
    with open(files[0], 'rb') as file:
        # Average size of `AVE_OVER` lines
        for line in file:
            # Count number of bytes in line
            thisLine = len(line) // line.count(b'\n')
            lineSize += thisLine
            count += 1
            if( count >= AVE_OVER ): break

    # Convert to average line size
    lineSize /= count
    # Get total size of all files
    totalSize = sum( os.path.getsize(fil) for fil in files )
    # Estimate total number of lines
    numLines = totalSize // lineSize

    return numLines


def filesExist(files):

    # Assume all files exist
    allExist = True
    # Iterate over each, if any dont exist, break
    for fil in files:
        if( not os.path.exists(fil) ):
            allExist = False
            break


    return allExist


def checkPath(tpath):
    path,name = os.path.split(tpath)
    if( len(path) > 0 ):
        if( not os.path.isdir(path) ): os.makedirs(path)

    return


def npzToDict(npz):
    """
    Given a numpy npz file, convert it to a dictionary with the same keys and values.

    Arguments
    ---------
    npz : <NpzFile>, input dictionary-like object

    Returns
    -------
    newDict : <dict>, output dictionary with key-values from npz file.

    """
    if( type(npz) is str ): npz = np.load(npz)

    newDict = { key : npz[key] for key in npz.keys() }
    return newDict

