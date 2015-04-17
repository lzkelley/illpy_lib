import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from .. Constants import *
#from .. Analyzer import getSubhaloRadialProfiles
from .. Profiler import getSubhaloRadialProfiles

import PlotFuncs as pfunc
import illpy
from illpy.Constants import *
from illpy import AuxFuncs as aux


NUM_RAD_BINS = 40

FIG_SIZE = [10, 6]
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

PS = 40                                                                                              # Point-size
ALPHA = 0.4

SCALE = 'log'

DASHES = [8,4]
DOTS = [3,2]

DEF_FNAME = '/n/home00/lkelley/illustris/EplusA/Subhalos/output/subhalos/fig-a05_ill-%d_subhalo-%d.png'


PART_TYPES = [ PARTICLE_TYPE_GAS, PARTICLE_TYPE_DM, PARTICLE_TYPE_STAR, PARTICLE_TYPE_BH ]



distConv = DIST_CONV*1000/KPC                                                                       # Convert simulation units to [pc]
densConv = DENS_CONV*np.power(PC, 3.0)/MSOL                                                         # Convert simulation units to [msol/pc^3]


COL_GAS   = 'green'
COL_STAR  = 'red'
COL_DM    = '0.25'
COL_BH    = 'black'

MARK_GAS  = 'o'
MARK_STAR = '*'
MARK_DM   = '^'


CMAP = plt.cm.jet

def plotFigA05_Subhalo(run, num, data, fname=None, verbose=True):

    if( verbose ): print " - - FigA05_Subhalo.plotFigA05_Subhalo()"

    if( fname is None ): fname = DEF_FNAME % (run, num)


    '''
    #DM_MASS = data.masses[PARTICLE_TYPE_DM]

    posGas   = data[SNAPSHOT_POS][slices[PARTICLE_TYPE_GAS ]]
    posDM    = data[SNAPSHOT_POS][slices[PARTICLE_TYPE_DM  ]]
    posStars = data[SNAPSHOT_POS][slices[PARTICLE_TYPE_STAR]]
    posBH    = data[SNAPSHOT_POS][slices[PARTICLE_TYPE_BH  ]]
    '''
    
    ### Create Figure and Axes ###
    fig, ax = plt.subplots(figsize=FIG_SIZE, nrows=1, ncols=1, squeeze=False)
    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, hspace=HSPACE, top=TOP, wspace=WSPACE)

    figa05_profiles_radial(ax[0,0], data, slices)

    #figa05_project_particles(ax[1,1], data, slices, PARTICLE_TYPE_STAR)

    # Add Legend
    '''
    fig.legend(dots, names, loc='center right', ncol=1, prop={'size':FS_LEGEND},
               bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0.98,0.5) );
    '''

    # Save Figure
    fig.savefig(fname)
    print " - - - Saved Figure to '%s'" % (fname)

    return fig




def figa05_profiles_radial(ax0, data, slices):

    ### Configure Axes ###
    xlabel = "Radius [pc]"
    ylabel = r"Density [$M_\odot$/pc$^3$]"
    pfunc.setAxis(ax0, axis='x', fs=FS, label=xlabel, scale=SCALE)
    pfunc.setAxis(ax0, axis='y', fs=FS, label=ylabel, scale=SCALE)
    #ax1 = pfunc.twinAxis(ax0, scale='linear', grid=False)

    # Load Profiles
    rads, hg, hs, hd, hc = getSubhaloRadialProfiles(data)
    hc = np.array(hc)
    cfunc, cnorm, cmap = pfunc.cmapColors([0.3, np.max(hc)], scale='linear', cmap=CMAP)
    starCols = cfunc(hc)

    # Plot Profiles
    ax0.scatter(rads, hg, marker=MARK_GAS,  color=COL_GAS,  s=PS, alpha=ALPHA)
    #ax0.scatter(rads, hs, marker=MARK_STAR, color=COL_STAR, s=PS, alpha=ALPHA)
    ax0.scatter(rads, hs, marker=MARK_STAR, color=starCols, s=PS, alpha=ALPHA)
    ax0.scatter(rads, hd, marker=MARK_DM,   color=COL_DM,   s=PS, alpha=ALPHA)

    cbax = ax0.figure.add_axes([0.92, 0.2, 0.02, 0.6])
    cb = mpl.colorbar.ColorbarBase(cbax, cmap=cmap, norm=cnorm, orientation='vertical')

    return



'''
def figa05_project_particles(ax, data, slices, part):

    ### Configure Axes ###
    xlabel = "Radius [pc]"
    ylabel = ""
    pfunc.setAxis(ax, axis='x', fs=FS, label=xlabel, scale='linear')
    pfunc.setAxis(ax, axis='y', fs=FS, label=ylabel, scale='linear')

    ngas     = data[SNAPSHOT_NPART][PARTICLE_TYPE_GAS]
    nstars   = data[SNAPSHOT_NPART][PARTICLE_TYPE_STAR]

    # Get Positions of Particles
    posBH = data[SNAPSHOT_POS][slices[PARTICLE_TYPE_BH]]
    pos = (data[SNAPSHOT_POS][slices[part]] - posBH)*distConv
    nums = len(pos)
    maxPos = np.max( pos.flatten() )

    # Get weighting function
    if(   part == PARTICLE_TYPE_GAS  ): val = data[SNAPSHOT_DENS][slices[part]]*densConv
    elif( part == PARTICLE_TYPE_DM   ): val = [ data[SNAPSHOT_MASSES][part]*MASS_CONV ]*nums
    elif( part == PARTICLE_TYPE_STAR ): val = data[SNAPSHOT_MASS][ngas:(ngas+nstars)]*MASS_CONV
    elif( part == PARTICLE_TYPE_BH   ): val = [ data[SNAPSHOT_MASS][(ngas+nstars):] ]*nums
    else: raise RuntimeError("Unknown ``part`` = %s" % (str(part)))

    # Load effective sizes (smoothing lengths)
    size = data[SNAPSHOT_SUBFIND_HSML][slices[part]]*distConv/10.0
    
    minVal = np.min(val)
    maxVal = np.max(val)
    logScale = lambda xx: (np.log10(xx)-np.log10(minVal))/(np.log10(maxVal)-np.log10(minVal))

    if(   part == PARTICLE_TYPE_GAS  ): 
        scaleVal = logScale
        col = COL_GAS
    elif( part == PARTICLE_TYPE_DM   ): 
        scaleVal = lambda xx: 0.2
        col = COL_DM
    elif( part == PARTICLE_TYPE_STAR ): 
        scaleVal = logScale
        col = COL_STAR
    elif( part == PARTICLE_TYPE_BH   ): 
        scaleVal = lambda xx: 0.5
        col = COL_BH


    for xx, vv, ss in zip(pos, val, size):
        circ = plt.Circle((xx[0],xx[1]), ss, alpha=scaleVal(vv), color=col)
        ax.add_artist(circ)


    ax.set_xlim([-maxPos,maxPos])
    ax.set_ylim([-maxPos,maxPos])

    return
'''        



