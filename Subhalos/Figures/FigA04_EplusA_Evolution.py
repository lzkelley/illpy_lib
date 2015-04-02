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


FIG_SIZE = [12, 18]
LEFT     = 0.1
RIGHT    = 0.9
BOT      = 0.05
TOP      = 0.93
HSPACE   = 0.4
WSPACE   = 0.4

NROWS    = 7
NCOLS    = 3

FS = 10
FS_TITLE = 16
LW = 1.0

SCALE = 'log'

DEF_FNAME = '/n/home00/lkelley/illustris/EplusA/Subhalos/output/subhalos/fig-a04_ill-%d_eplusa_evolution.png'

PS    = [ 10,     30,   ]
COL   = [ '0.5',  'red']
ALPHA = [ 0.5,    0.8   ]


def plotFigA04_EplusA_Evolution(run, catOld, catNew, inds_old_epa, inds_old_oth, inds_new_epa, inds_new_oth, 
                                weightsEPA, others=None, fname=None, verbose=VERBOSE):

    if( verbose ): print " - - FigA04_EplusA_Evolution.plotFigA04_EplusA_Evolution()"

    assert len(inds_old_epa) == len(inds_new_epa), "'epa' inds old and new must match!"
    assert len(inds_old_oth) == len(inds_new_oth), "'oth' inds old and new must match!"

    ### Set Parameters ###
    numSubhalos = len(weightsEPA)
    numOths     = len(inds_old_oth)
    numEpas     = len(inds_old_epa)

    # Sort EpA subhalos by WeightsEPA (this should be the case already, but make sure
    inds = np.argsort(weightsEPA)
    old_epa = np.array(inds_old_epa[inds])
    new_epa = np.array(inds_new_epa[inds])

    if( fname  is None ): fname = DEF_FNAME % (run)
    if( others is None ): others = numOths

    if( others > len(inds_old_oth) ): others = numOths

    if( others < numOths ): 
        old_oth = np.random.choice( list(inds_old_oth), size=others)
        new_oth = np.random.choice( list(inds_new_oth), size=others)
    else:
        old_oth = inds_old_oth
        new_oth = inds_new_oth

    old_oth = np.array(old_oth)
    new_oth = np.array(new_oth)



    ### Get Target Parameters for 'other' (non-EpA) and 'epa' Galaxies ###

    # Radii
    rads_st_new   = [ catNew[SH_RAD_TYPE][new_oth, PARTICLE_TYPE_STAR]*DIST_CONV,
                      catNew[SH_RAD_TYPE][new_epa, PARTICLE_TYPE_STAR]*DIST_CONV ]
    rads_st_old   = [ catOld[SH_RAD_TYPE][old_oth, PARTICLE_TYPE_STAR]*DIST_CONV,
                      catOld[SH_RAD_TYPE][old_epa, PARTICLE_TYPE_STAR]*DIST_CONV ]

    rads_dm_new   = [ catNew[SH_RAD_TYPE][new_oth, PARTICLE_TYPE_DM]*DIST_CONV,
                      catNew[SH_RAD_TYPE][new_epa, PARTICLE_TYPE_DM]*DIST_CONV ]
    rads_dm_old   = [ catOld[SH_RAD_TYPE][old_oth, PARTICLE_TYPE_DM]*DIST_CONV,
                      catOld[SH_RAD_TYPE][old_epa, PARTICLE_TYPE_DM]*DIST_CONV ]
    
    # Masses
    mass_bh_new   = [ catNew[SH_BH_MASS][new_oth]*MASS_CONV,
                      catNew[SH_BH_MASS][new_epa]*MASS_CONV ]
    mass_bh_old   = [ catOld[SH_BH_MASS][old_oth]*MASS_CONV,
                      catOld[SH_BH_MASS][old_epa]*MASS_CONV ]

    mass_st_new   = [ catNew[SH_MASS_TYPE][new_oth, PARTICLE_TYPE_STAR]*MASS_CONV,
                      catNew[SH_MASS_TYPE][new_epa, PARTICLE_TYPE_STAR]*MASS_CONV ]
    mass_st_old   = [ catOld[SH_MASS_TYPE][old_oth, PARTICLE_TYPE_STAR]*MASS_CONV,
                      catOld[SH_MASS_TYPE][old_epa, PARTICLE_TYPE_STAR]*MASS_CONV ]

    mass_gas_new  = [ catNew[SH_MASS_TYPE][new_oth, PARTICLE_TYPE_GAS ]*MASS_CONV,
                      catNew[SH_MASS_TYPE][new_epa, PARTICLE_TYPE_GAS ]*MASS_CONV ]
    mass_gas_old  = [ catOld[SH_MASS_TYPE][old_oth, PARTICLE_TYPE_GAS ]*MASS_CONV,
                      catOld[SH_MASS_TYPE][old_epa, PARTICLE_TYPE_GAS ]*MASS_CONV ]


    dens_st_new   = [ ms/np.power(rs,3.0) for ms,rs in zip(mass_st_new, rads_st_new) ]
    dens_st_old   = [ ms/np.power(rs,3.0) for ms,rs in zip(mass_st_old, rads_st_old) ]

    gas_frac_new  = [ mg/(ms+mg) for ms,mg in zip(mass_st_new, mass_gas_new) ]
    gas_frac_old  = [ mg/(ms+mg) for ms,mg in zip(mass_st_old, mass_gas_old) ]


    # Star Formation Rates
    sfr_new       = [ catNew[SH_SFR][new_oth], catNew[SH_SFR][new_epa] ]
    sfr_old       = [ catOld[SH_SFR][old_oth], catOld[SH_SFR][old_epa] ]

    sfr_sp_new    = [ rat/mst for rat, mst in zip(sfr_new, mass_st_new) ]
    sfr_sp_old    = [ rat/mst for rat, mst in zip(sfr_old, mass_st_old) ]

    # Photometric Bands
    photo_gg_new  = [ catNew[SH_PHOTO][new_oth, PHOTO_g], catNew[SH_PHOTO][new_epa, PHOTO_g] ]
    photo_gg_old  = [ catOld[SH_PHOTO][old_oth, PHOTO_g], catOld[SH_PHOTO][old_epa, PHOTO_g] ]

    photo_rr_new  = [ catNew[SH_PHOTO][new_oth, PHOTO_r], catNew[SH_PHOTO][new_epa, PHOTO_r] ]
    photo_rr_old  = [ catOld[SH_PHOTO][old_oth, PHOTO_r], catOld[SH_PHOTO][old_epa, PHOTO_r] ]

    photo_ii_new  = [ catNew[SH_PHOTO][new_oth, PHOTO_i], catNew[SH_PHOTO][new_epa, PHOTO_i] ]
    photo_ii_old  = [ catOld[SH_PHOTO][old_oth, PHOTO_i], catOld[SH_PHOTO][old_epa, PHOTO_i] ]

    photo_gmr_new = [ photo_gg_new[0]-photo_rr_new[0], photo_gg_new[1]-photo_rr_new[1] ]
    photo_gmr_old = [ photo_gg_old[0]-photo_rr_old[0], photo_gg_old[1]-photo_rr_old[1] ]

    photo_gmi_new = [ photo_gg_new[0]-photo_ii_new[0], photo_gg_new[1]-photo_ii_new[1] ]
    photo_gmi_old = [ photo_gg_old[0]-photo_ii_old[0], photo_gg_old[1]-photo_ii_old[1] ]


    # Get unique color for each EpA subhalo
    cols = pfunc.setColorCycle(numEpas)


    ### Create Figure and Axes ###

    fig, ax = plt.subplots(figsize=FIG_SIZE, nrows=NROWS, ncols=NCOLS, squeeze=False)
    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOT, hspace=HSPACE, top=TOP, wspace=WSPACE)


    # Plot Stellar Mass --- Row 0
    labels = [ "Stellar Mass [Msol] Z=0.1", "Stellar Mass [Msol] Z=0.0" ]
    figa04_row(ax[0,:], mass_st_old, mass_st_new, labels,
               cols=cols, verbose=verbose)

    # Plot BH Mass --- Row 1
    name = "BH Mass [Msol]"
    labels = [ "%s Z=0.1" % name, "%s Z=0.0" % name ]
    figa04_row(ax[1,:], mass_bh_old, mass_bh_new, labels,
               cols=cols, verbose=verbose)

    # Plot SFR --- Row 2
    labels = [ "SFR [Msol/yr] Z=0.1", "SFR [Msol/yr] Z=0.0" ]
    figa04_row(ax[2,:], sfr_old, sfr_new, labels,
               cols=cols, verbose=verbose)

    # Plot Specific-SFR --- Row 3
    name = "S-SFR [Msol/yr/Msol]"
    labels = [ "%s Z=0.1" % name, "%s Z=0.0" % name ]
    figa04_row(ax[3,:], sfr_sp_old, sfr_sp_new, labels,
               cols=cols, verbose=verbose)

    # Plot Stellar Radius --- Row 4
    name = "Star HM Radius [kpc]"
    labels = [ "%s Z=0.1" % name, "%s Z=0.0" % name ]
    figa04_row(ax[4,:], rads_st_old, rads_st_new, labels,
               cols=cols, verbose=verbose)

    # Plot Gas Fraction --- Row 5
    name = "Gas Frac"
    labels = [ "%s Z=0.1" % name, "%s Z=0.0" % name ]
    figa04_row(ax[5,:], gas_frac_old, gas_frac_new, labels,
               cols=cols, verbose=verbose)


    # Plot Color --- Row 6
    name = "Color (g-r) [Mag]"
    labels = [ "%s Z=0.1" % name, "%s Z=0.0" % name ]
    figa04_row(ax[6,:], photo_gmr_old, photo_gmr_new, labels,
               scale='linear', xreverse=True, cols=cols, verbose=verbose)


    # Add Title
    textStr = "E-plus-A vs. Others\nRedshift Z = 0.1 vs. Z = 0.0"
    text = fig.text(0.5, 0.98, textStr, fontsize=FS_TITLE, transform=plt.gcf().transFigure,
                    verticalalignment='top', horizontalalignment='center')

    # Save Figure
    fig.savefig(fname)
    print " - - - Saved Figure to '%s'" % (fname)

    return fig

# plotFigA04_EplusA_Evolution()



def figa04_row(ax, xx, yy, labels, xreverse=False, scale=SCALE, cols=None, verbose=VERBOSE):

    figa04_scatter(ax[0], xx, yy, labels, xreverse=xreverse, scale=scale, cols=cols, verbose=verbose)
    figa04_hist(ax[1], xx, labels[0], scale=scale, verbose=verbose)
    figa04_hist(ax[2], yy, labels[1], scale=scale, verbose=verbose)

    return



def figa04_scatter(ax, xx, yy, labels, xreverse=False, scale=SCALE, cols=None, verbose=VERBOSE):

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



def figa04_hist(ax, data, label, scale=SCALE, verbose=VERBOSE):

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

