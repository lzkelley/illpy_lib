import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import PlotFuncs as pfunc
import illpy
from illpy.Constants import *

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
LW = 2.0

PS = 20                                                                                              # Point-size
ALPHA = 0.4

SCALE = 'log'

DASHES = [8,4]
DOTS = [3,2]

DEF_FNAME = '/n/home00/lkelley/illustris/EplusA/Subhalos/output/subhalos/fig-a05_run-%d_subhalo-%d.png'


def plotFigA05_Subhalo(run, num, data, fname=None, verbose=True):

    if( verbose ): print " - - FigA05_Subhalo.plotFigA05_Subhalo()"

    if( fname is None ): fname = DEF_FNAME % (run)


    numParts = data.npart_loaded
    cumNumParts = np.concatenate([[0],np.cumsum(numParts)])
    slices = [ slice( cumNumParts[ii], cumNumParts[ii+1] ) for ii in range(len(numParts)) ]

    print "numParts = ", numParts
    print "cumulative = ", cumNumParts
    print "slices = ", slices
    
    return

    
    ### Create Figure and Axes ###
    fig, ax = plt.subplots(figsize=FIG_SIZE, nrows=2, ncols=2, squeeze=False)
    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, hspace=HSPACE, top=TOP, wspace=WSPACE)


    ## Plot Scatter SFR versus BH Mass
    figa05_profles(ax[0,0], data)


    # Add Legend
    '''
    fig.legend(dots, names, loc='center right', ncol=1, prop={'size':FS_LEGEND},
               bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0.98,0.5) );
    '''

    # Save Figure
    fig.savefig(fname)
    print " - - - Saved Figure to '%s'" % (fname)

    return fig




def figa05_profiles(ax, args, sfr, sfr_sp=None, which=AXIS_BH, stats=None, stats_sp=None):

    ### Configure Axes ###
    xlabel = ""
    ylabel = ""
    pfunc.setAxis(ax, axis='x', fs=FS, label=xlabel, scale=SCALE)
    pfunc.setAxis(ax, axis='y', fs=FS, label=ylabel, scale=SCALE)


    ### Plot SFR ###

    s1 = ax.scatter(args, sfr, lw=LW, c=COL_SFR, marker='o', s=PS, alpha=ALPHA)
    dots.append(s1)
    names.append('SFR')
    

    ### Plot Specific SFR on Second Axis ###

    if( sfr_sp is not None ):
        y2label = r'Specific SFR [(Msol/yr)/Msol]'
        ax2 = pfunc.twinAxis(ax, twin='x', fs=FS, label=y2label, scale=SCALE, c=COL_SFR_SP, grid=False)
        s2 = ax2.scatter(args, sfr_sp, lw=LW, c=COL_SFR_SP, marker='o', s=PS, alpha=ALPHA)
        dots.append(s2)
        names.append('Specific SFR')

        if( stats_sp is not None ):
            for st in stats_sp:
                l1 = ax2.axhline(st, ls='-', lw=2*LW, color='0.5', alpha=0.5)
                l1 = ax2.axhline(st, ls='--', lw=LW, color=COL_SFR_SP)
                l1.set_dashes(DASHES)


    if( stats is not None ):
        for st in stats:
            l1 = ax.axhline(st, ls='-', lw=2*LW, color='0.5')
            l1 = ax.axhline(st, ls='--', lw=LW, color=COL_SFR)
            l1.set_dashes(DASHES)



    # Set xlimits to extrema
    xlim = [np.min(args), np.max(args)]
    ax.set_xlim(xlim)

    return dots, names


