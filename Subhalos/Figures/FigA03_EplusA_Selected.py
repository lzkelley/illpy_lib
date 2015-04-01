import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from .. Constants import *
import PlotFuncs as pfunc
import illpy
from illpy.Constants import *

VERBOSE = True

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
LW = 1.0

SCALE = 'log'

DASHES = [8,4]
DOTS = [3,2]

NUM_BINS = 20

#NUM = 40
#PERCENTILES = [10,50,90]

DEF_FNAME = '/n/home00/lkelley/illustris/EplusA/Subhalos/output/subhalos/fig-a03_ill-%d_eplusa_selected.png'


PS    = [ 10,     30,   ]
COL   = [ '0.5',  'red']
ALPHA = [ 0.5,    0.8   ]


def plotFigA03_EplusA_Selected(run, cat, inds_epa, inds_oth, weights, 
                               others=None, fname=None, verbose=VERBOSE):

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
    
    mass_bh = [ cat[SH_BH_MASS][others], cat[SH_BH_MASS][inds_epa] ]
    mass_st = [ cat[SH_MASS_TYPE][others,PARTICLE_TYPE_STAR], cat[SH_MASS_TYPE][inds_epa,PARTICLE_TYPE_STAR] ]
    #inds    = [ inds_oth, inds_epa ]



    ### Create Figure and Axes ###

    fig, ax = plt.subplots(figsize=FIG_SIZE, nrows=NROWS, ncols=NCOLS, squeeze=False)
    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOT, hspace=HSPACE, top=TOP, wspace=WSPACE)


    labels = [ "BH Mass", "Stellar Mass" ]
    figa03_row(ax[0,:], mass_bh, mass_st, labels, verbose=verbose)

    ## Plot Histogram SFR
    #figa01_hist(ax[0,0], sfr, specific=False, stats=stats)




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



def figa03_row(ax, xx, yy, labels, verbose=VERBOSE):

    if( verbose ): print " - - - FigA03_EplusA_Selected.figa03_row()"

    figa03_scatter(ax[0], xx, yy, labels, verbose=verbose)
    figa03_hist(ax[1], xx, labels[0], verbose=verbose)
    figa03_hist(ax[2], yy, labels[1], verbose=verbose)



    return



def figa03_scatter(ax, xx, yy, labels, verbose=VERBOSE):

    if( verbose ): print " - - - FigA03_EplusA_Selected.figa03_scatter()"

    pfunc.setAxis(ax, axis='x', fs=FS, c='black', label=labels[0], scale=SCALE)
    pfunc.setAxis(ax, axis='y', fs=FS, c='black', label=labels[1], scale=SCALE)

    for ii in range(2):
        ax.scatter(xx[ii], yy[ii], marker='o', s=PS[ii], c=COL[ii], alpha=ALPHA[ii])

    return



def figa03_hist(ax, data, label, verbose=VERBOSE):

    if( verbose ): print " - - - FigA03_EplusA_Selected.figa03_hist()"

    # Setup Axes
    pfunc.setAxis(ax, axis='x', fs=FS, label=label, scale=SCALE)
    pfunc.setAxis(ax, axis='y', fs=FS, label='Count', scale=SCALE)

    # Create Bins
    bins = np.array([np.min(np.concatenate(data)), np.max(np.concatenate(data))])
    bins = np.logspace( *np.log10(bins), num=NUM_BINS )

    # Plot Histogram (put EpA galaxies on bottom)
    print data[1]

    ax.hist(data[::-1], bins=bins, log=True, color=COL[::-1], histtype='barstacked', lw=LW)

    ylim = np.array(ax.get_ylim())
    ax.set_ylim(0.9, ylim[1])

    return

