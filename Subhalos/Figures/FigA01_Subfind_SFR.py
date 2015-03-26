import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy as sp

import PlotFuncs as pfunc
import illpy
from illpy.Constants import *

FIG_SIZE = [12, 8]
LEFT     = 0.1
RIGHT    = 0.9
BOTTOM   = 0.1
TOP      = 0.95
HSPACE   = 0.4

FS = 14
FS_TITLE = 16
FS_LEGEND = 10
LW = 2.0

PS = 20                                                                                              # Point-size
ALPHA = 0.4

SCALE = 'log'

DASHES = [8,4]
DOTS = [3,2]


AXIS_BH      = 0
AXIS_STELLAR = 1
AXIS_TOTAL   = 2

AXIS_TYPES = [ AXIS_BH, AXIS_STELLAR, AXIS_TOTAL ]

DEF_FNAME = '/n/home00/lkelley/illustris/EplusA/Subhalos/output/subhalos/fig-a01_run-%d_subfind_sfr.png'

COL_SFR = 'blue'
COL_SFR_SP = 'red'


def plotFigA01_Subfind_SFR(run, sfr, massTypes, massBH, fname=None, verbose=True):

    if( verbose ): print " - - FigA01_Subfind_SFR.plotFigA01_Subfind_SFR()"

    if( fname is None ): fname = DEF_FNAME % (run)

    massBH      = massBH*MASS_CONV
    massStellar = massTypes[:,PARTICLE_TYPE_STAR]*MASS_CONV
    massTotal   = np.sum(massTypes, axis=1)*MASS_CONV                                               # Sum over all particle types

    # Calculate Specific SFR
    sfr_sp = sfr/massStellar


    ### Create Figure and Axes ###
    fig, ax = plt.subplots(figsize=FIG_SIZE, nrows=2, ncols=1, squeeze=False)
    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, hspace=HSPACE, top=TOP)


    ## Plot versus BH Mass
    dots, names = figa01_panel(ax[0,0], massBH, sfr, sfr_sp, which=AXIS_BH)

    ## Plot versus Stellar Mass
    dots, names = figa01_panel(ax[1,0], massStellar, sfr, sfr_sp, which=AXIS_STELLAR)



    # Add Legend
    fig.legend(dots, names, loc='center right', ncol=1, prop={'size':FS_LEGEND},
               bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0.98,0.5) );

    # Save Figure
    fig.savefig(fname)
    if( verbose ): print " - - - Saved Figure to '%s'" % (fname)

    return fig




def figa01_panel(ax, args, sfr, sfr_sp=None, which=AXIS_BH):

    assert which in AXIS_TYPES, "``which`` must be in '%s' !!" % (AXIS_TYPES)


    ylabel = r'Star Formation Rate [Msol/yr]'

    if(   which is AXIS_BH      ): xlabel = r'Blackhole Mass [$M_\odot$]'
    elif( which is AXIS_STELLAR ): xlabel = r'Stellar Mass [$M_\odot$]'
    else:                          xlabel = r'Total Mass [$M_\odot$]'

    dots = []
    names = []

    pfunc.setAxis(ax, axis='x', fs=FS, c='black', label=xlabel, scale=SCALE)
    pfunc.setAxis(ax, axis='y', fs=FS, c='black', label=ylabel, scale=SCALE)

    s1 = ax.scatter(args, sfr, lw=LW, c=COL_SFR, marker='o', s=PS, alpha=ALPHA)
    dots.append(s1)
    names.append('SFR')

    if( sfr_sp is not None ):
        y2label = r'Specific SFR [(Msol/yr)/Msol]'
        ax2 = pfunc.twinAxis(ax, twin='x', fs=FS, label=y2label, scale=SCALE, c=COL_SFR_SP, grid=False)
        s2 = ax2.scatter(args, sfr_sp, lw=LW, c=COL_SFR_SP, marker='o', s=PS, alpha=ALPHA)
        dots.append(s2)
        names.append('Specific SFR')



    return dots, names
