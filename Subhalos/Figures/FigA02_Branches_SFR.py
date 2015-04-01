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
LW2 = 2.0

PS = 20                                                                                              # Point-size
ALPHA = 0.4

SCALE = 'log'

GRID = False

DASHES = [8,4]
DOTS = [3,2]


NUM_BINS = 20

PERCENTILES = [10,50,90]

'''
NUM = 40


AXIS_BH      = 0
AXIS_STELLAR = 1
AXIS_TOTAL   = 2

AXIS_TYPES = [ AXIS_BH, AXIS_STELLAR, AXIS_TOTAL ]

COL_SFR_SP = 'red'
'''

COL_SFR = '0.3'

DEF_FNAME = '/n/home00/lkelley/illustris/EplusA/Subhalos/output/subhalos/fig-a02_ill-%d_branches_sfr.png'


def plotFigA02_Branches_SFR(run, branches, his, los, weights, fname=None, verbose=True):

    if( verbose ): print " - - FigA02_Branches_SFR.plotFigA02_Branches_SFR()"

    if( fname is None ): fname = DEF_FNAME % (run)

    sfr = branches[SH_SFR]
    numSubhalos = len(sfr)
    snaps = np.concatenate([his,los])
    div = np.average([his[-1],los[0]])

    '''
    cosmo = Cosmology()
    scales = cosmo.snapshotTimes(snaps)
    redz   = cosmo.redshift(scales)
    lbtime = cosmo.lookbackTime(scales)
    lbtime[np.where(lbtime < 0.0)] = 0.0
    '''

    ### Create Figure and Axes ###
    fig, ax = plt.subplots(figsize=FIG_SIZE, nrows=1, ncols=2, squeeze=False)
    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, hspace=HSPACE, top=TOP, wspace=WSPACE)


    inds = np.argsort(weights)
    #inds = np.random.choice(inds, size=1000)

    cfunc, cnorm, cmap = pfunc.cmapColors([np.min(weights), np.max(weights)])


    ## Plot Scatter SFR versus BH Mass
    lines, names = figa02_sfr_lines(ax[0,0], snaps, sfr[inds], cfunc(weights[inds]))
    ax[0,0].axvline(div, color='k', ls='--', lw=LW)


    ## Plot Scatter SFR versus BH Mass
    figa02_weights_hist(ax[0,1], weights[inds], cfunc)


    cbax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cb = mpl.colorbar.ColorbarBase(cbax, cmap=cmap, norm=cnorm, orientation='vertical')

    ticks = [np.min(weights), np.average(weights), np.max(weights)]
    tick_labels = [ "%.2f" % tt for tt in ticks ]

    cb.set_ticks(ticks)
    cb.set_ticklabels(tick_labels)

    '''
    # Add Legend
    fig.legend(dots, names, loc='center right', ncol=1, prop={'size':FS_LEGEND},
               bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0.98,0.5) );
    '''

    # Save Figure
    fig.savefig(fname)
    if( verbose ): print " - - - Saved Figure to '%s'" % (fname)

    return cb

# plotFigA01_Subfind_SFR()




def figa02_sfr_lines(ax, snaps, sfr, weights):

    nums = len(sfr)

    lines = []
    names = []

    ### Configure Axes ###
    xlabel = r'Snapshot [#]'
    ylabel = r'Star Formation Rate [Msol/yr]'

    pfunc.setAxis(ax, axis='x', fs=FS, c='black', label=xlabel, scale='linear', grid=GRID)
    pfunc.setAxis(ax, axis='y', fs=FS, c='black', label=ylabel, scale=SCALE, grid=GRID)

    ### Plot SFR Histories ###
    for ii in xrange(nums):
        l1, = ax.plot(snaps, sfr[ii], ls='-', lw=LW, c=weights[ii], alpha=ALPHA)


    aves = np.average(sfr, axis=0)
    stds = np.std(sfr, axis=0)

    l1, = ax.plot(snaps, aves, ls='--', lw=LW, c='black')
    l1.set_dashes(DASHES)
    for jj in [-1,+1]:
        l1, = ax.plot(snaps, aves+jj*stds, ls=':', lw=LW, c='black')
        l1.set_dashes(DOTS)


    return lines, names



def figa02_weights_hist(ax, weights, colFunc, num=NUM_BINS):

    lines = []
    names = []

    ### Configure Axes ###
    xlabel = r'Weight'
    ylabel = r'Count'
    pfunc.setAxis(ax, axis='x', fs=FS, c='black', label=xlabel, scale=SCALE)
    pfunc.setAxis(ax, axis='y', fs=FS, c='black', label=ylabel, scale=SCALE)

    stats = np.percentile(weights, PERCENTILES)

    # Create Bins
    extr = np.array([np.min(weights), np.max(weights)])
    bins = np.logspace( *np.log10(extr), num=num )

    # Plot
    nums, bins, patches = ax.hist(weights, bins=bins, orientation='vertical', log=True, color='blue', lw=LW2)

    binAves = np.array([ np.average([bins[ii],bins[ii+1]]) for ii in range(len(bins)-1) ])
    binCols = colFunc(binAves)

    for col, pat in zip(binCols,patches):
        pat.set_color(col)

    ax.set_xlim(extr)

    lens = len(stats)

    for ii,st in enumerate(stats):
        l1 = ax.axvline(st, ls='--', lw=LW2, color='black')
        if( (ii == 1 and lens == 3) or (ii == 2 and lens == 5) ): l1.set_dashes(DASHES)
        else:                                                     l1.set_dashes(DOTS)

    return
