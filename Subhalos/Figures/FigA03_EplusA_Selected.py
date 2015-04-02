import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from .. Constants import *
import PlotFuncs as pfunc
import illpy
from illpy.Constants import *
from illpy import AuxFuncs as aux


NUM_BINS = 20
VERBOSE = True


FIG_SIZE = [12, 14]
LEFT     = 0.1
RIGHT    = 0.9
BOT      = 0.1
TOP      = 0.95
HSPACE   = 0.4
WSPACE   = 0.4

NROWS    = 5
NCOLS    = 3

FS = 10
FS_TITLE = 16
LW = 1.0

SCALE = 'log'

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

    # Sort EpA subhalos by BH Mass
    mbh = cat[SH_BH_MASS][inds_epa]
    inds = np.argsort(mbh)
    inds_epa = inds_epa[inds]

    if( fname  is None ): fname = DEF_FNAME % (run)
    if( others is None ): others = numOths
    if( others > len(inds_oth) ): others = numOths

    # Select all or a subset of ``non_inds`` to compare with ``inds``
    if( others < numOths ): others = np.random.choice( list(inds_oth), size=others)
    else:                   others = inds_oth
    others = np.array(others)



    ### Get Target Parameters for 'other' (non-EpA) and 'epa' Galaxies ###

    # Radii
    rads_st   = [ cat[SH_RAD_TYPE][others,   PARTICLE_TYPE_STAR]*DIST_CONV,
                  cat[SH_RAD_TYPE][inds_epa, PARTICLE_TYPE_STAR]*DIST_CONV ]
    rads_dm   = [ cat[SH_RAD_TYPE][others,   PARTICLE_TYPE_DM]*DIST_CONV,
                  cat[SH_RAD_TYPE][inds_epa, PARTICLE_TYPE_DM]*DIST_CONV ]

    # Masses
    mass_bh   = [ cat[SH_BH_MASS][others  ]*MASS_CONV,
                  cat[SH_BH_MASS][inds_epa]*MASS_CONV ]
    mass_st   = [ cat[SH_MASS_TYPE][others,   PARTICLE_TYPE_STAR]*MASS_CONV,
                  cat[SH_MASS_TYPE][inds_epa, PARTICLE_TYPE_STAR]*MASS_CONV ]
    mass_gas  = [ cat[SH_MASS_TYPE][others,   PARTICLE_TYPE_GAS ]*MASS_CONV,
                  cat[SH_MASS_TYPE][inds_epa, PARTICLE_TYPE_GAS ]*MASS_CONV ]

    dens_st   = [ ms/np.power(rs,3.0) for ms,rs in zip(mass_st, rads_st) ]

    gas_frac  = [ mg/(ms+mg) for ms,mg in zip(mass_st, mass_gas) ]

    # Star Formation Rates
    sfr       = [ cat[SH_SFR][others], cat[SH_SFR][inds_epa] ]
    sfr_sp    = [ rat/mst for rat, mst in zip(sfr, mass_st) ]

    # Photometric Bands
    photo_gg  = [ cat[SH_PHOTO][others, PHOTO_g], cat[SH_PHOTO][inds_epa, PHOTO_g] ]
    photo_rr  = [ cat[SH_PHOTO][others, PHOTO_r], cat[SH_PHOTO][inds_epa, PHOTO_r] ]
    photo_ii  = [ cat[SH_PHOTO][others, PHOTO_i], cat[SH_PHOTO][inds_epa, PHOTO_i] ]
    photo_gmr = [ photo_gg[0]-photo_rr[0], photo_gg[1]-photo_rr[1] ]
    photo_gmi = [ photo_gg[0]-photo_ii[0], photo_gg[1]-photo_ii[1] ]


    # Get unique color for each EpA subhalo
    cols = pfunc.setColorCycle(numEpas)


    ### Create Figure and Axes ###

    fig, ax = plt.subplots(figsize=FIG_SIZE, nrows=NROWS, ncols=NCOLS, squeeze=False)
    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOT, hspace=HSPACE, top=TOP, wspace=WSPACE)


    # Plot Masses
    labels = [ "BH Mass [Msol]", "Stellar Mass [Msol]" ]
    figa03_row(ax[0,:], mass_bh, mass_st, labels,
               cols=cols, verbose=verbose)

    # Plot Star Formation Rates
    labels = [ "SFR [Msol/yr]", "S-SFR [Msol/yr/Msol]" ]
    figa03_row(ax[1,:], sfr, sfr_sp, labels,
               cols=cols, verbose=verbose)

    # Plot (g-r) vs. r
    labels = [ "r [Mag]", "g-r [Mag]" ]
    figa03_row(ax[2,:], photo_rr, photo_gmr, labels,
               xreverse=True, scale='linear', cols=cols, verbose=verbose)

    # Plot Radii
    labels = [ "Stellar HM Radii [kpc]", "DM HM Radii [kpc]" ]
    figa03_row(ax[3,:], rads_st, rads_dm, labels,
               cols=cols, verbose=verbose)

    # Plot Gas-Fraction and Stellar Density
    labels = [ "Gas Fraction (Mass)", "Stellar Density [Msol/kpc^3]" ]
    figa03_row(ax[4,:], gas_frac, dens_st, labels,
               cols=cols, verbose=verbose)


    # Add Title
    ax[0,0].set_title("E-plus-A Redshift Z = 0.0", size=FS_TITLE)
    

    # Save Figure
    fig.savefig(fname)
    print " - - - Saved Figure to '%s'" % (fname)

    return fig

# plotFigA03_EplusA_Selected()



def figa03_row(ax, xx, yy, labels, xreverse=False, scale=SCALE, cols=None, verbose=VERBOSE):

    if( verbose ): print " - - - FigA03_EplusA_Selected.figa03_row()"
    if( verbose ): print " - - - - '%s' and '%s'" % (labels[0], labels[1])

    figa03_scatter(ax[0], xx, yy, labels, xreverse=xreverse, scale=scale, cols=cols, verbose=verbose)
    figa03_hist(ax[1], xx, labels[0], scale=scale, verbose=verbose)
    figa03_hist(ax[2], yy, labels[1], scale=scale, verbose=verbose)


    return



def figa03_scatter(ax, xx, yy, labels, xreverse=False, scale=SCALE, cols=None, verbose=VERBOSE):

    if( verbose ): print " - - - FigA03_EplusA_Selected.figa03_scatter()"

    pfunc.setAxis(ax, axis='x', fs=FS, c='black', label=labels[0], scale=scale)
    pfunc.setAxis(ax, axis='y', fs=FS, c='black', label=labels[1], scale=scale)
    if( xreverse ): ax.invert_xaxis()

    if( cols is None ): useCols = COL
    else:               useCols = [ COL[0], cols ]

    for ii in range(2):
        ax.scatter(xx[ii], yy[ii], marker='o', s=PS[ii], c=useCols[ii], alpha=ALPHA[ii])

    ax.set_xlim(aux.extrema(np.concatenate(xx)))
    ax.set_ylim(aux.extrema(np.concatenate(yy)))

    return



def figa03_hist(ax, data, label, scale=SCALE, verbose=VERBOSE):

    if( verbose ): print " - - - FigA03_EplusA_Selected.figa03_hist()"

    # Setup Axes
    pfunc.setAxis(ax, axis='x', fs=FS, label=label, scale=scale)
    pfunc.setAxis(ax, axis='y', fs=FS, label='Count', scale='log')

    useLog = (scale == 'log')

    # Create Bins
    bins = np.array([np.min(np.concatenate(data)), np.max(np.concatenate(data))])
    bins = aux.space(bins, num=NUM_BINS, log=useLog)

    # Plot Histogram (put EpA galaxies on bottom)
    ax.hist(data[::-1], bins=bins, log=True, color=COL[::-1], histtype='barstacked', lw=LW)

    ylim = np.array(ax.get_ylim())
    ax.set_ylim(0.8, ylim[1])

    return

