import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from .. Constants import *
import PlotFuncs as pfunc
import illpy
from illpy.Constants import *

FIG_SIZE = [12, 10]
LEFT     = 0.1
RIGHT    = 0.9
BOT      = 0.1
TOP      = 0.95
HSPACE   = 0.4
WSPACE   = 0.4

NROWS    = 2
NCOLS    = 3

FS = 14
FS_TITLE = 16
FS_LEGEND = 10
LW = 2.0

PS = 20                                                                                              # Point-size
ALPHA = 0.4

SCALE = 'log'

DASHES = [8,4]
DOTS = [3,2]

#NUM = 40
#PERCENTILES = [10,50,90]

DEF_FNAME = '/n/home00/lkelley/illustris/EplusA/Subhalos/output/subhalos/fig-a03_ill-%d_eplusa_selected.png'

#COL_SFR = 'blue'
#COL_SFR_SP = 'red'


def plotFigA03_EplusA_Selected(run, cat, inds_epa, inds_oth, weights, others=None, fname=None, verbose=True):

    if( verbose ): print " - - FigA01_Subfind_SFR.plotFigA03_EplusA_Selected()"

    ### Set Parameters ###
    numSubhalos = len(weights)
    numOths     = len(inds_oth)
    numEpas     = len(inds_epa)

    if( fname  is None ): fname = DEF_FNAME % (run)
    if( others is None ): others = numOths
    if( others > len(inds_oth) ): others = numOths

    # Select all or a subset of ``non_inds`` to compare with ``inds``
    if( others < numOths ): others = np.random.choice( list(inds_oth), size=others)
    else:                   others = inds_oth
    others = np.array(others)
    
    # Calculate statistics
    #stats    = np.percentile(sfr,    PERCENTILES)
    
    print inds_epa

    mass_bh = [ cat[SH_BH_MASS][inds_epa], 
                cat[SH_BH_MASS][others]    ]
    mass_st = [ cat[SH_MASS_TYPE][inds_epa,PARTICLE_TYPE_STAR], cat[SH_MASS_TYPE][others,PARTICLE_TYPE_STAR] ]


    return


    ### Create Figure and Axes ###
    fig, ax = plt.subplots(figsize=FIG_SIZE, nrows=NROWS, ncols=NCOLS, squeeze=False)
    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOT, hspace=HSPACE, top=TOP, wspace=WSPACE)



    ## Plot Scatter SFR versus BH Mass
    dots, names = figa03_row(ax[0,1], massBH, sfr, sfr_sp, which=AXIS_BH, stats=stats)

    ## Plot Histogram SFR
    figa01_hist(ax[0,0], sfr, specific=False, stats=stats)




    # Add Legend
    '''
    fig.legend(dots, names, loc='center right', ncol=1, prop={'size':FS_LEGEND},
               bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0.98,0.5) );
    '''


    # Save Figure
    fig.savefig(fname)
    if( verbose ): print " - - - Saved Figure to '%s'" % (fname)

    return fig

# plotFigA03_EplusA_Selected()



def figa03_row(ax, xx, yy, verbose=True):

    if( verbose ): print "FigA03_EplusA_Selected.figa03_row()"

    dots = []
    names = []

    ### Configure Axes ###
    ylabel = r'Star Formation Rate [Msol/yr]'

    if(   which is AXIS_BH      ): xlabel = r'Blackhole Mass [$M_\odot$]'
    elif( which is AXIS_STELLAR ): xlabel = r'Stellar Mass [$M_\odot$]'
    else:                          xlabel = r'Total Mass [$M_\odot$]'

    pfunc.setAxis(ax, axis='x', fs=FS, c='black', label=xlabel, scale=SCALE)
    pfunc.setAxis(ax, axis='y', fs=FS, c=COL_SFR, label=ylabel, scale=SCALE)


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



'''
def figa01_hist(ax, sfr, num=NUM, specific=False, stats=None):

    # Specific --- lower right, use right-axis
    if( specific ): 
        ylabel = r'Specific SFR [(Msol/yr)/Msol]'
        col = COL_SFR_SP
        pos = 1.0
    # Normal   --- upper left,  use left-axis
    else:
        ylabel = r'Star Formation Rate [Msol/yr]'
        col = COL_SFR
        pos = 0.0

    # Setup Axes
    pfunc.setAxis(ax, axis='x', fs=FS, c='black', label='Count', scale=SCALE)
    pfunc.setAxis(ax, axis='y', fs=FS, c=col, label=ylabel, scale=SCALE, pos=pos)
    if( not specific ): ax.invert_xaxis()

    # Create Bins
    bins = np.array([np.min(sfr[np.nonzero(sfr)]), np.max(sfr)])
    bins = np.logspace( *np.log10(bins), num=num )

    # Plot
    ax.hist(sfr, bins=bins, orientation='horizontal', log=True, color=col)

    if( stats is not None ):
        for st in stats:
            l1 = ax.axhline(st, ls='-', lw=2*LW, color='0.5')
            l1 = ax.axhline(st, ls='--', lw=LW, color=col)
            l1.set_dashes(DASHES)

    return
'''
