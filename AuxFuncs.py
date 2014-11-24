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

import numpy as np
import warnings
import os
import sys
import matplotlib                  as mpl
import cPickle as pickle
import matplotlib                  as mpl
from matplotlib import pyplot      as plt

from glob import glob
from datetime import datetime

from Constants import *
#from ObjMergers import Mergers
#from ObjDetails import Details

import ObjLog
from ObjLog import Log

import arepo


DT_THRESH = 1.0e-5                                                                                  # Frac diff b/t times to accept as equal




###  =====================================  ###
###  ==========  DETAILS FILES  ==========  ###
###  =====================================  ###


def getIllustrisBHDetailsFilenames(runNum, runsDir, log=None):
    '''Get a list of 'blackhole_details' files for a target Illustris simulation'''

    if( log ):
        log += 1
        if( log ): log.log("getIllustrisBHDetailsFilenames()")

    detailsNames      = np.copy(runsDir).tostring()
    if( not detailsNames.endswith('/') ): detailsNames += '/'
    detailsNames += RUN_DIRS[runNum]
    if( not detailsNames.endswith('/') ): detailsNames += '/'
    detailsNames += BH_DETAILS_FILENAMES

    if( log ): log.log("Searching '%s'" % detailsNames, 1)
    files        = sorted(glob(detailsNames))                                                       # Find and sort files
    if( log ): log.log("Found %d files" % (len(files)), 2)

    if( log ): log -= 1
    return files






###  ======================================  ###
###  ==========  SNAPSHOT FILES  ==========  ###
###  ======================================  ###


def getSnapshotTimesFilename(runNum, workDir):
    #timesFile = workDir + (SAVE_SNAPSHOT_TIMES_FILENAME % (runNum))
    timesFile = PP_TIMES_FILENAME(runNum)
    return timesFile


def getSnapshotFilename(snapNum, runNum, log=None):
    """
    Given a run number and snapshot number, construct the approprirate snapshot filename.

    input
    -----
    snapNum : IN [int] snapshot number,       i.e. 100 for snapdir_100
    runNum  : IN [int] simulation run number, i.e. 3 for Illustris-3

    output
    ------
    return     filename : [str]
    """

    if( log ): log.log("getSnapshotFilename()")
    snapName = SNAPSHOT_NAMES(runNum, snapNum)

    return snapName


def loadSnapshotTimes(runNum, runsDir, loadFile=None, saveFile=None, loadsave=None, log=None):
    """
    Get the time (scale-factor) of each snapshot

    input
    -----
    runNum  : [int] simulation run number, i.e. 3 for Illustris-3


    output
    ------
    return   times : list of [float]   simulation times (scale-factors)

    """

    if( log ): log.log("loadSnapshotTimes()", 1)

    times = np.zeros(NUM_SNAPS, dtype=DBL)

    load = False
    save = False
    ### If loadsave is specified: load if file exists, otherwise save
    # Make sure 'loadFile'/'saveFile' are not specified along with 'loadsave'
    if( (loadsave and loadFile) or (loadsave and saveFile) ):
        raise RuntimeError("[AuxFuncs.loadSnapshotTimes()] Error: too many files!")
    elif( loadsave ):
        # If file already exists, load from it
        if( os.path.exists(loadsave) ): loadFile = loadsave
        # If file doesn't exist, save to it
        else:                           saveFile = loadsave

    if( loadFile ): load = True
    if( saveFile ): save = True

    # Load pre-existing times file
    if( load ):
        if( log ): log.log("Loading snapshot times from '%s'" % (loadFile), 2)
        timesFile = np.load( loadFile )
        times[:]  = timesFile['times']
    # Re-extract times
    else:
        if( log ): log.log("Extracting times from snapshots", 2)
        for snapNum in range(NUM_SNAPS):
            snapFile = getSnapshotFilename(snapNum, runNum, runsDir)
            # Load only the header from the given snapshot
            snapHead = arepo.Snapshot(snapFile, onlyHeader=True)
            times[snapNum] = snapHead.time


    # Save snapshot times to NPZ file
    if( save ):
        if( log ): log.log("Saving snapshot times to '%s'" % (saveFile), 2)
        np.savez(saveFile, times=times)


    return times

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



###  =======================================  ###
###  =============  PLOTTING  ==============  ###
###  =======================================  ###



def createFigures(nfigs=1):

    figs = [ plt.figure(figsize=FIG_SIZE) for ii in range(nfigs) ]
    for ff in figs:
        for axpos,axsize in zip(AX_POS,AX_SIZE):
            ff.add_axes(axpos + axsize)

    return figs


def saveFigure(fname, fig, log=None):
    fig.savefig(fname)
    if( log ): log.log("Saved figure '%s'" % (fname) )
    return


def setColorCycle(num, ax=None, cmap=plt.cm.spectral, left=0.1, right=0.9):
    if(ax == None): ax = plt.gca()
    cols = [cmap(it) for it in np.linspace(left, right, num)]
    ax.set_color_cycle(cols[::-1])
    return cols


def plotRect(ax, loc):
    rect = mpl.patches.Rectangle((loc[0], loc[1]), loc[2], loc[3],
                                 alpha=0.4, facecolor='None', ls='dashed', lw=1.0, transform=ax.transData)
    ax.add_patch(rect)
    return


def histPlot(ax, values, bins, weights=None, ls='-', lw=1.0, color='k', ave=False, scale=None, label=None):
    """
    Manually plot a histogram.

    Uses numpy.histogram to obtain binned values, then plots them manually
    as connected lines with the given parameters.  If `weights` are provided,
    they are the values summed for each bin instead of 1.0 for each entry in
    `values`.

    Parameters
    ----------
    ax : object, matplotlib.axes
        Axes on which to make plot

    values : array_like, scalar
        Array of values to be binned and plotted.  Each entry which belongs in
        a bin, increments that bin by 1.0 or the corresponding entry in
        `weights` if it is provived.

    bins : array_like, scalar
        Edges of bins to use for histogram, with both left and right edges.
        If `bins` has length N, there will be N-1 bins.

    weights : array_like, scalar, optional
        Array of the same length as `values` which contains the weights to be
        added to each bin.

    ls : str, optional
        linestyle to plot with

    lw : scalar, optional
        lineweight to plot with

    color : str, optional
        color of line to plot with

    scale : scalar or array of scalars
        Rescale the resulting histogram by this/these values
        (e.g. 1/binVol will convert to density)

    label : str, optional
        label to associate with plotted histogram line

    Returns
    -------
    ll : object, matplotlib.lines.Line2D
        Line object which was plotted to axes `ax`
        (can then be used to make a legend, etc)

    hist : array, scalar
        The histogram which is plotted

    """

    hist,edge = np.histogram( values, bins=bins, weights=weights )

    # Find the average of each weighed bin instead.
    if( ave and weights != None ): 
        hist = [ hh/nn if nn > 0 else 0.0 
                 for hh,nn in zip(hist,np.histogram( values, bins=bins)[0]) ]

    # Rescale the bin values
    if( scale != None ):
        hist *= scale

    yval = np.concatenate([ [hh,hh] for hh in hist ])
    xval = np.concatenate([ [edge[jj],edge[jj+1]] for jj in range(len(edge)-1) ])
    ll, = ax.plot( xval, yval, ls, lw=lw, color=color, label=label)

    return ll, hist



def plotHistLine(ax, bins, values, ls='-', lw=1.0, color='k'):
    """
    Manually plot a histogrammed data.


    Parameters
    ----------
    ax : object matplotlib.axes
        Axes on which to make plot
    bins : array_like, scalar
        Edges of bins to use for histogram, with both left and right edges.
        If `bins` has length N, there will be N-1 bins.
    values : array_like, scalar
        Array of binned values
    (ls : str, optional)
        linestyle to plot with
    (lw : scalar, optional)
        lineweight to plot with
    (color : str, optional)
        color of line to plot with

    Returns
    -------
    ll : object, matplotlib.lines.Line2D
        Line object which was plotted to axes `ax`
        (can then be used to make a legend, etc)

    """

    yval = np.concatenate([ [vv,vv] for vv in values ])
    xval = np.concatenate([ [bins[jj],bins[jj+1]] for jj in range(len(bins)-1) ])
    ll, = ax.plot( xval, yval, ls, lw=lw, color=color)

    return ll




def configPlot(ax, xlabel=None, ylabel=None, title=None, logx=False, logy=False, grid=True,
               symlogx=0.0, symlogy=0.0):
    """ Configure an axis object with the given settings. """

    # Set Labels
    if( xlabel ): ax.set_xlabel(xlabel)
    if( ylabel ): ax.set_ylabel(ylabel)
    if( title  ): ax.set_title(title)

    # Grid
    ax.grid(grid)

    # Axis Scales
    if( symlogx != 0.0 ): ax.set_xscale('symlog', linthreshx=symlogx)
    elif( logx ):         ax.set_xscale('log')
    if( symlogy != 0.0 ): ax.set_yscale('symlog', linthreshy=symlogy)
    elif( logy ):         ax.set_yscale('log')

    return



'''
def plotVLine(ax, pos, style='-', col='0.5', lw=1.0, text=None, tpos=None):
    ylo,yhi = ax.get_ylim()
    ll, = ax.plot([pos,pos], [ylo,yhi], style, color=col, lw=lw)
    ax.set_ylim(ylo,yhi)

    if( text != None ):
        if( tpos == None ): tpos = 0.99*yhi
        ax.text(pos, tpos, text, horizontalalignment='center', verticalalignment='top',
                transform = ax.transData, bbox=dict(facecolor='white', alpha=0.7), color=col )


    return ax, ll


def plotHLine(ax, pos, style='-', col='0.5', lw=1.0, text=None, tpos=None):
    xlo,xhi = ax.get_xlim()
    ll, = ax.plot([xlo,xhi], [pos,pos], style, color=col, lw=lw)
    ax.set_xlim(xlo,xhi)

    if( text != None ):
        if( tpos == None ): tpos = 0.99*xhi
        ax.text(tpos, pos, text, horizontalalignment='right', verticalalignment='center',
                transform = ax.transData, bbox=dict(facecolor='white', alpha=0.7), color=col )

    return ax, ll
'''



###  ===================================  ###
###  =============  MATH  ==============  ###
###  ===================================  ###

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
        print "[AuxBlackholeFuncs.findBins] target = %e, bins = {%e,%e}; low,high = %s,%s !" % \
            ( target, bins[0], bins[-1], str(low), str(high) )
        raise RuntimeError("Could not find bins!")


    return [low,high,dlo,dhi]



'''

###  ====================================  ###
###  =============  OTHER  ==============  ###
###  ====================================  ###


def guessNumsFromFilename(fname):

    run = fname.split("Illustris-")[-1]
    run = run.split("/")[0]
    run = np.int(run)

    snap = fname.split("groups_")[-1]
    snap = snap.split("/")[0]
    snap = np.int(snap)

    return snap, run


'''



def stringArray(arr, format='%.2f'):
    out = [ format % elem for elem in arr ]
    out = "[ " + " ".join(out) + " ]"
    return out



def getFileSizeString(filename, asstr=True):
    size = os.path.getsize(filename)
    return convertDataSizeToString(size)


def convertDataSizeToString(size):
    prefSize, pref = getPrefixed(size)
    unit = pref + 'B'
    return "%.2f %s" % (prefSize, unit)


def getPrefixed(tval):
    val = np.copy(tval)
    prefs = [ '', 'K' , 'M' , 'G' ]
    mult  = 1000.0

    cnt = 0
    while( val > mult ):
        val /= mult
        cnt  += 1
        if( cnt > 3 ):
            raise RuntimeError("Error: value too large '%s'" % (str(val)) )

    return val, prefs[cnt]



def checkFileDir(tpath):
    """
    For any given path make sure the 'head' portion (directories) exist.

    e.g. checkFileDir('one/two/three/file.txt') will make sure that
         'one/two/three/' exists.  'file.txt' is ignored
    """
    head,tail = os.path.split(tpath)
    if( len(head) > 0 ):
        checkDir(head)

    return



def checkDir(tdir):
    """
    Create the given directory if it doesn't already exist.

    Return the same directory assuring it terminates with a '/'
    """
    ndir = str(tdir)
    if( len(ndir) > 0 ):
        # If directory doesn't exist, create it
        if( not os.path.isdir(ndir) ): os.makedirs(ndir)
        # If it still doesn't exist, error
        if( not os.path.isdir(ndir) ): raise RuntimeError("Directory '%s' does not exist!" % (ndir) )
        # Make sure pattern ends with '/'
        if( not ndir.endswith('/') ): ndir += '/'

    return ndir


#



