import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import PlotFuncs as pfunc
import illpy
from illpy.Constants import *

from ..Constants import *
import illpy
from illpy import Cosmology


FIG_SIZE = [12, 8]
LEFT     = 0.1
RIGHT    = 0.9
BOTTOM   = 0.1
TOP      = 0.95
HSPACE   = 0.4
WSPACE   = 0.4

FS = 14
FS_TITLE = 16
FS_LEGEND = 10
LW = 1.0

PS = 20                                                                                              # Point-size
ALPHA = 0.4

SCALE = 'log'

GRID = False

DASHES = [8,4]
DOTS = [3,2]



NUM_SUBS = 100

'''
NUM = 40
PERCENTILES = [10,50,90]

AXIS_BH      = 0
AXIS_STELLAR = 1
AXIS_TOTAL   = 2

AXIS_TYPES = [ AXIS_BH, AXIS_STELLAR, AXIS_TOTAL ]

COL_SFR_SP = 'red'
'''

COL_SFR = '0.3'

DEF_FNAME = '/n/home00/lkelley/illustris/EplusA/Subhalos/output/subhalos/fig-a02_ill-%d_branches_sfr.png'


def plotFigA02_Branches_SFR(run, snaps, branches, subs=NUM_SUBS, fname=None, verbose=True):

    if( verbose ): print " - - FigA02_Branches_SFR.plotFigA02_Branches_SFR()"

    if( fname is None ): fname = DEF_FNAME % (run)

    ### Create Figure and Axes ###
    fig, ax = plt.subplots(figsize=FIG_SIZE, nrows=1, ncols=1, squeeze=False)
    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, hspace=HSPACE, top=TOP, wspace=WSPACE)
    
    ### Choose Random Subset of Subhalos ###
    numSubhalos = len(branches[SH_SFR])

    # If trying to select more than available, plot all
    if( subs >= numSubhalos ): 
        subs = numSubhalos
        inds = np.arange(subs)
    # If actually a subset, select randomly
    else:
        inds = np.random.choice( np.arange(numSubhalos), size=subs, replace=False )

    if( verbose ): print " - - - Plotting %d/%d" % (subs, numSubhalos)

    ## Plot Scatter SFR versus BH Mass
    lines, names = figa02_sfr_lines(ax[0,0], snaps, branches[SH_SFR][inds], cols=branches[SH_BH_MASS][inds])

    '''
    # Add Legend
    fig.legend(dots, names, loc='center right', ncol=1, prop={'size':FS_LEGEND},
               bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0.98,0.5) );
    '''

    # Save Figure
    fig.savefig(fname)
    if( verbose ): print " - - - Saved Figure to '%s'" % (fname)

    return fig

# plotFigA01_Subfind_SFR()


def figa02_sfr_lines(ax, snaps, sfr, cols=None):


    nums = len(sfr)

    cosmo = Cosmology()
    scales = cosmo.snapshotTimes(snaps)
    redz   = cosmo.redshift(scales)
    lbtime = cosmo.lookbackTime(scales)
    lbtime[np.where(lbtime < 0.0)] = 0.0

    lines = []
    names = []


    start = np.argmin(snaps)
    other = np.where(snaps != np.min(snaps))[0]

    weight = sfr[:,start]/np.max(sfr[:,other], axis=1)

    print sfr[4]

    print sfr[:20,start]
    np.max(sfr[:20,other], axis=1)
    print weight[:20]
    print np.average(weight)
    print np.std(weight)

    ### Configure Axes ###
    xlabel = r'Snapshot [#]'
    ylabel = r'Star Formation Rate [Msol/yr]'

    pfunc.setAxis(ax, axis='x', fs=FS, c='black', label=xlabel, scale='linear', grid=GRID)
    pfunc.setAxis(ax, axis='y', fs=FS, c='black', label=ylabel, scale=SCALE, grid=GRID)

    inds = np.where( cols > 0.0 )
    norm = mpl.colors.LogNorm(vmin=np.min(cols[inds]), vmax=np.max(cols[inds]))

    ### Plot SFR ###

    for ii in xrange(nums):

        usecols = cols

        if( cols is None ):
            inds = np.where( sfr[ii] > 0.0 )[0]
            if( len(inds) < 2 ): continue
            l1, = ax.plot(snaps[inds], sfr[ii][inds], ls='-', lw=LW, c=COL_SFR, alpha=ALPHA)
        else:
            inds = np.where( (sfr[ii] > 0.0) & (cols[ii] > 0.0) )[0]
            if( len(inds) < 2 ): continue
            l1 = pfunc.colorline(snaps[inds], sfr[ii][inds], cols[ii][inds], norm=norm, lw=LW, alpha=ALPHA)


    aves = np.average(sfr, axis=0)
    stds = np.std(sfr, axis=0)

    l1, = ax.plot(snaps, aves, ls='--', lw=LW, c='black')
    l1.set_dashes(DASHES)
    for jj in [-1,+1]:
        l1, = ax.plot(snaps, aves+jj*stds, ls=':', lw=LW, c='black')
        l1.set_dashes(DOTS)

    return lines, names


