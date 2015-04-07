import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import PlotFuncs as pfunc
import illpy
from illpy.Constants import *

from .. Constants import *

FIG_SIZE = [12, 10]
NROWS = 2
NCOLS = 2

LEFT     = 0.1
RIGHT    = 0.9
BOTTOM   = 0.1
TOP      = 0.95
HSPACE   = 0.4
WSPACE   = 0.4

FS = 14
FS_TITLE = 18
LW = 1.0

PS = 20                                                                                              # Point-size
ALPHA = 0.4

SCALE = 'log'

DASHES = [8,4]
DOTS = [3,2]

PERCENTILES = [16.0, 84.0]

#DEF_FNAME = '/n/home00/lkelley/illustris/EplusA/Subhalos/output/subhalos/fig-a06_ill-%d_eplusa_profiles.png'
DEF_FNAME = '/n/home00/lkelley/illustris/EplusA/Subhalos/output/subhalos/fig-a06%s_ill-%d_eplusa_profiles.png'

COL_EPAG = 'red'
#COL_NULL = '0.5'
COL_NULL = 'blue'

def plotFigA06_EplusA_Profiles(run, epags, nulls, fname=None, verbose=True):

    if( verbose ): print " - - FigA06_EplusA_Profiles.plotFigA06_EplusA_Profiles()"

    #if( fname is None ): fname = DEF_FNAME % (run)


    binEdges = epags[PROFILE_BIN_EDGES]
    binAves  = epags[PROFILE_BIN_AVES ]


    fname = DEF_FNAME % ('a', run)
    fig_a(binEdges, epags, nulls, fname)

    fname = DEF_FNAME % ('b', run)
    fig_b(binEdges, epags, nulls, fname)

    fname = DEF_FNAME % ('c', run)
    fig_c(binEdges, binAves, epags, nulls, fname)


    return




def fig_a(edges, epags, nulls, fname):

    ### Create Figure and Axes ###
    fig, ax = plt.subplots(figsize=FIG_SIZE, nrows=NROWS, ncols=NCOLS, squeeze=False)
    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, hspace=HSPACE, top=TOP, wspace=WSPACE)

    _draw_profile_lines(ax[0,0], edges, epags[PROFILE_GAS],   nulls[PROFILE_GAS],   'Gas Density [$M_\odot$/pc$^3$]' )
    _draw_profile_lines(ax[0,1], edges, epags[PROFILE_STARS], nulls[PROFILE_STARS], 'Star Density [$M_\odot$/pc$^3$]')
    _draw_profile_lines(ax[1,0], edges, epags[PROFILE_DM],    nulls[PROFILE_DM],    'DM Density [$M_\odot$/pc$^3$]'  )
    _draw_profile_lines(ax[1,1], edges, epags[PROFILE_COLS],  nulls[PROFILE_COLS],  'Color (g-r) [Magnitude]'        , yscale='linear')

    # Save Figure
    fig.savefig(fname)
    print " - - - Saved Figure to '%s'" % (fname)

    return



def fig_b(edges, epags, nulls, fname):

    ### Create Figure and Axes ###
    fig, ax = plt.subplots(figsize=FIG_SIZE, nrows=NROWS, ncols=NCOLS, squeeze=False)
    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, hspace=HSPACE, top=TOP, wspace=WSPACE)

    _draw_profile_lines_norm(ax[0,0], edges, epags[PROFILE_GAS],   nulls[PROFILE_GAS],   'Gas Density' )
    _draw_profile_lines_norm(ax[0,1], edges, epags[PROFILE_STARS], nulls[PROFILE_STARS], 'Star Density')
    _draw_profile_lines_norm(ax[1,0], edges, epags[PROFILE_DM],    nulls[PROFILE_DM],    'DM Density'  )
    _draw_profile_lines_norm(ax[1,1], edges, epags[PROFILE_COLS],  nulls[PROFILE_COLS],  'Color (g-r) [Magnitude]', yscale='linear')

    #ax.text(0.5, 0.95, "Normalized Profiles", size=FS_TITLE)

    # Save Figure
    fig.savefig(fname)
    print " - - - Saved Figure to '%s'" % (fname)

    return



def fig_c(edges, mids, epags, nulls, fname):

    ### Create Figure and Axes ###
    fig, ax = plt.subplots(figsize=FIG_SIZE, nrows=NROWS, ncols=NCOLS, squeeze=False)
    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, hspace=HSPACE, top=TOP, wspace=WSPACE)

    _draw_profile_lines_stats(ax[0,0], edges, mids, epags[PROFILE_GAS],   nulls[PROFILE_GAS],   'Gas Density' )
    _draw_profile_lines_stats(ax[0,1], edges, mids, epags[PROFILE_STARS], nulls[PROFILE_STARS], 'Star Density')
    _draw_profile_lines_stats(ax[1,0], edges, mids, epags[PROFILE_DM],    nulls[PROFILE_DM],    'DM Density'  )
    _draw_profile_lines_stats(ax[1,1], edges, mids, epags[PROFILE_COLS],  nulls[PROFILE_COLS],  'Color (g-r) [Magnitude]', yscale='linear')

    #ax[0,0].text(0.5, 0.95, "Normalized Profiles", size=FS_TITLE)

    # Save Figure
    fig.savefig(fname)
    print " - - - Saved Figure to '%s'" % (fname)

    return



def _draw_profile_lines(ax, bins, epags, nulls, name, yscale='log'):
    
    pfunc.setAxis(ax, axis='x', fs=FS, label="Distance [pc]", scale=SCALE)
    pfunc.setAxis(ax, axis='y', fs=FS, label=name, scale=yscale)

    ### Plot Nulls ###
    for nul in nulls:
        l1 = pfunc.plotHistLine(ax, bins, nul, lw=LW, color=COL_NULL, alpha=ALPHA)

    ### Plot E+A ###
    for epa in epags:
        l2 = pfunc.plotHistLine(ax, bins, epa, lw=LW, color=COL_EPAG, alpha=ALPHA)

        
    ### Plot Averages ###
    aveNull = np.median(nulls, axis=0)
    aveEpag = np.median(epags, axis=0)

    l3 = pfunc.plotHistLine(ax, bins, aveEpag, lw=3*LW, color='black', alpha=1.0)
    l3 = pfunc.plotHistLine(ax, bins, aveEpag, lw=LW, color=COL_EPAG, alpha=1.0)
    l4 = pfunc.plotHistLine(ax, bins, aveNull, lw=3*LW, color='black', alpha=1.0)
    l4 = pfunc.plotHistLine(ax, bins, aveNull, lw=LW, color=COL_NULL, alpha=1.0)
    
    return



def _draw_profile_lines_norm(ax, bins, epags, nulls, name, yscale='log'):
    
    pfunc.setAxis(ax, axis='x', fs=FS, label="Distance [pc]", scale=SCALE)
    pfunc.setAxis(ax, axis='y', fs=FS, label=name, scale=yscale)

    useEpags = np.array(epags)
    useNulls = np.array(nulls)

    ### Plot Nulls ###
    for ii in range(len(useNulls)):
        useNulls[ii] /= np.max(useNulls[ii])
        l1 = pfunc.plotHistLine(ax, bins, useNulls[ii], lw=LW, color=COL_NULL, alpha=ALPHA)


    ### Plot E+A ###
    for ii in range(len(useEpags)):
        useEpags[ii] /= np.max(useEpags[ii])
        l1 = pfunc.plotHistLine(ax, bins, useEpags[ii], lw=LW, color=COL_EPAG, alpha=ALPHA)

        
    ### Plot Averages ###
    aveNull = np.median(useNulls, axis=0)
    aveEpag = np.median(useEpags, axis=0)

    l3 = pfunc.plotHistLine(ax, bins, aveEpag, lw=3*LW, color='black', alpha=1.0)
    l3 = pfunc.plotHistLine(ax, bins, aveEpag, lw=LW, color=COL_EPAG, alpha=1.0)
    l4 = pfunc.plotHistLine(ax, bins, aveNull, lw=3*LW, color='black', alpha=1.0)
    l4 = pfunc.plotHistLine(ax, bins, aveNull, lw=LW, color=COL_NULL, alpha=1.0)

    return



def _draw_profile_lines_stats(ax, edges, mids, epags, nulls, name, yscale='log'):

    INTERP = True
    
    pfunc.setAxis(ax, axis='x', fs=FS, label="Distance [pc]", scale=SCALE)
    pfunc.setAxis(ax, axis='y', fs=FS, label=name, scale=yscale)

    aveEpags = np.average(epags, axis=0)
    medEpags = np.median (epags, axis=0)
    loEpags  = np.percentile(epags, PERCENTILES[0], axis=0)
    hiEpags  = np.percentile(epags, PERCENTILES[1], axis=0)


    aveNulls = np.average(nulls, axis=0)
    medNulls = np.median (nulls, axis=0)
    loNulls  = np.percentile(nulls, PERCENTILES[0], axis=0)
    hiNulls  = np.percentile(nulls, PERCENTILES[1], axis=0)


    ### Epags ###

    pfunc.plotHistLine(ax, edges, aveEpags, lw=LW, color=COL_EPAG, alpha=1.0, ls='-')
    pfunc.plotHistLine(ax, edges, medEpags, lw=LW, color=COL_EPAG, alpha=1.0, ls='--')
    pfunc.plotHistLine(ax, edges, loEpags,  lw=LW, color=COL_EPAG, alpha=1.0, ls='-')
    pfunc.plotHistLine(ax, edges, hiEpags,  lw=LW, color=COL_EPAG, alpha=1.0, ls='-')

    ylim = ax.get_ylim()

    xval, yval_lo = pfunc.histLine(edges, loEpags)
    xval, yval_hi = pfunc.histLine(edges, hiEpags)

    inds = np.where( yval_hi <= 0.0 )
    xval = np.delete(xval, inds)
    yval_lo = np.delete(yval_lo, inds)
    yval_hi = np.delete(yval_hi, inds)
    yval_lo[np.where(yval_lo <= 0.0)] = ylim[0]

    ax.fill_between(xval, yval_lo, yval_hi, facecolor=COL_EPAG, alpha=ALPHA, interpolate=INTERP)


    ### Nulls ###

    pfunc.plotHistLine(ax, edges, aveNulls, lw=LW, color=COL_NULL, alpha=1.0, ls='-')
    pfunc.plotHistLine(ax, edges, medNulls, lw=LW, color=COL_NULL, alpha=1.0, ls='--')
    pfunc.plotHistLine(ax, edges, loNulls,  lw=LW, color=COL_NULL, alpha=1.0, ls='-')
    pfunc.plotHistLine(ax, edges, hiNulls,  lw=LW, color=COL_NULL, alpha=1.0, ls='-')

    ylim = ax.get_ylim()

    xval, yval_lo = pfunc.histLine(edges, loNulls)
    xval, yval_hi = pfunc.histLine(edges, hiNulls)

    inds = np.where( yval_hi <= 0.0 )
    xval = np.delete(xval, inds)
    yval_lo = np.delete(yval_lo, inds)
    yval_hi = np.delete(yval_hi, inds)
    yval_lo[np.where(yval_lo <= 0.0)] = ylim[0]

    ax.fill_between(xval, yval_lo, yval_hi, facecolor=COL_NULL, alpha=ALPHA, interpolate=INTERP)


    return
