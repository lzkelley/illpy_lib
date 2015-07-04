"""
Process radial profiles of Illustris subhalos.

Functions
---------
 - subhaloRadialProfiles() : construct binned, radial density profiles for all particle types


"""

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from illpy.Constants import GET_ILLUSTRIS_DM_MASS, PARTICLE, DTYPE, BOX_LENGTH

import Subhalo
import Constants
from Constants import SNAPSHOT, SUBHALO

import zcode.Math     as zmath
import zcode.Plotting as zplot
import zcode.InOut    as zio

VERBOSE = True

NUM_RAD_BINS = 100



def subhaloRadialProfiles(run, snapNum, subhalo, radBins=None, nbins=NUM_RAD_BINS, 
                          mostBound=None, verbose=VERBOSE):
    """
    Construct binned, radial profiles of density for each particle species.

    Profiles for the velocity dispersion and gravitational potential are also constructed for 
    all particle types together.

    Arguments
    ---------
       run       <int>    : illustris simulation run number {1,3}
       snapNum   <int>    : illustris simulation snapshot number {1,135}
       subhalo   <int>    : subhalo index number for target snapshot
       radBins   <flt>[N] : optional, right-edges of radial bins in simulation units
       nbins     <int>    : optional, numbers of bins to create if ``radBins`` is `None`
       mostBound <int>    : optional, ID number of the most-bound particle for this subhalo
       verbose   <bool>   : optional, print verbose output

    Returns
    -------
       radBins   <flt>[N]   : coordinates of right-edges of ``N`` radial bins
       posRef    <flt>[3]   : coordinates in simulation box of most-bound particle (used as C.O.M.)
       partTypes <int>[M]   : particle type numbers for ``M`` types, (``illpy.Constants.PARTICLE``)
       partNames <str>[M]   : particle type strings for each type
       numsBins  <int>[M,N] : binned number of particles for ``M`` particle types, ``N`` bins each
       massBins  <flt>[M,N] : binned radial mass profile 
       densBins  <flt>[M,N] : binned mass density profile
       potsBins  <flt>[N]   : binned gravitational potential energy profile for all particles
       dispBins  <flt>[N]   : binned velocity dispersion profile for all particles

    """


    if( verbose ): print " - - Profiler.subhaloRadialProfiles()"

    if( verbose ): print " - - - Loading subhalo partile data"
    # Redirect output during this call
    with zio.StreamCapture() as strCap:
        partData, partTypes = Subhalo.importSubhaloParticles(run, snapNum, subhalo, verbose=False)

    partNums = [ pd['count'] for pd in partData ]
    partNames = [ PARTICLE.NAMES(pt) for pt in partTypes ]
    numPartTypes = len(partNums)
    if( verbose ):
        print " - - - - Run %d, Snap %d, Subhalo %d : Loaded %s particles" % \
            (run, snapNum, subhalo, str(partNums))


    ## Find the most-bound Particle
    #  ----------------------------

    posRef = None

    # If no particle ID is given, find it
    if( mostBound is None ): 
        # Get group catalog
        mostBound = Subhalo.importGroupCatalogData(3, 135, subhalos=subhalo, \
                                                   fields=[SUBHALO.MOST_BOUND])

    if( mostBound is None ): raise RuntimeError("Could not find mostBound particle ID number!")

    # Find the most-bound particle, store its position
    for pdat,pname in zip(partData, partNames):
        # Skip, if no particles of this type
        if( pdat['count'] == 0 ): continue
        inds = np.where( pdat[SNAPSHOT.IDS] == mostBound )[0]
        if( len(inds) == 1 ): 
            if( verbose ): print " - - - Found Most Bound Particle in '%s'" % (pname)
            posRef = pdat[SNAPSHOT.POS][inds[0]]
            break

    # } pdat,pname

    if( posRef is None ): raise RuntimeError("Could not find most bound particle in snapshot!")


    mass = np.zeros(numPartTypes, dtype=object)
    rads = np.zeros(numPartTypes, dtype=object)
    pots = np.zeros(numPartTypes, dtype=object)
    disp = np.zeros(numPartTypes, dtype=object)
    radExtrema = None


    ## Iterate over all particle types and their data
    #  ==============================================
    
    if( verbose ): print " - - - Extracting and processing particle properties"
    for ii, (data, ptype) in enumerate(zip(partData, partTypes)):

        # Make sure the expected number of particles are found
        if( data['count'] != partNums[ii] ):
            errstr  = "Type '%s' count mismatch after loading!!\n" % (partNames[ii])
            errstr += "\tExpecting = %d" % (partNums[ii])
            errstr += "\tRetrieved = %d" % (data['count'])
            raise RuntimeError(errstr)


        # Skip if this particle type has no elements
        #    use empty lists so that call to ``np.concatenate`` below works (ignored)
        if( data['count'] == 0 ): 
            mass[ii] = []
            rads[ii] = []
            pots[ii] = []
            disp[ii] = []
            continue

        # Extract relevant data from dictionary
        posn   = reflectPos(data[SNAPSHOT.POS])

        # DarkMatter Particles all have the same mass, store that single value
        if( ptype == PARTICLE.DM ): mass[ii] = [ GET_ILLUSTRIS_DM_MASS(run) ]
        else:                       mass[ii] = data[SNAPSHOT.MASS]

        # Convert positions to radii from ``posRef`` (most-bound particle), and find radial extrema
        rads[ii] = zmath.dist(posn, posRef)
        pots[ii] = data[SNAPSHOT.POT]
        disp[ii] = data[SNAPSHOT.SUBF_VDISP]
        radExtrema = zmath.minmax(rads[ii], prev=radExtrema, nonzero=True)

    # } for data, ptype



    ## Create Radial Bins
    #  ------------------

    # Create radial bin spacings, these are the upper-bound radii
    if( radBins is None ): radBins = zmath.spacing(radExtrema, scale='log', num=nbins)

    # Find average bin positions, and radial bin (shell) volumes
    binVols = np.zeros(nbins)
    for ii in range(len(radBins)):
        if( ii == 0 ): binVols[ii] = np.power(radBins[ii],3.0)
        else:          binVols[ii] = np.power(radBins[ii],3.0) - np.power(radBins[ii-1],3.0)
    # } ii



    ## Bin Properties for all Particle Types
    #  -------------------------------------

    densBins = np.zeros([numPartTypes, nbins], dtype=DTYPE.SCALAR)    # Density
    massBins = np.zeros([numPartTypes, nbins], dtype=DTYPE.SCALAR)    # Mass
    numsBins = np.zeros([numPartTypes, nbins], dtype=DTYPE.INDEX )    # Count of particles

    # second dimension to store averages [0] and standard-deviations [1]
    potsBins = np.zeros([nbins, 2], dtype=DTYPE.SCALAR)               # Grav Potential Energy
    dispBins = np.zeros([nbins, 2], dtype=DTYPE.SCALAR)               # Velocity dispersion

    # Iterate over particle types
    if( verbose ): print " - - - Binning properties by radii"
    for ii, (data, ptype) in enumerate(zip(partData, partTypes)):

        # Skip if this particle type has no elements
        if( data['count'] == 0 ): continue

        # Get the total mass in each bin
        numsBins[ii,:], massBins[ii,:] = zmath.histogram(rads[ii], radBins, weights=mass[ii],
                                                         edges='right', func='sum', stdev=False)

        # Divide by volume to get density
        densBins[ii,:] = massBins[ii,:]/binVols

    # } for ii, pt1


    # Consistency check on numbers of particles
    for ii in xrange(numPartTypes):

        numExp = partNums[ii]
        numAct = np.sum(numsBins[ii])

        # Make sure the total number of particles are in bins
        if( numExp != numAct ):
            errstr  = "Type '%s' count mismatch after binning!!\n" % (partNames[ii])
            errstr += "\tExpecting = %d" % (numExp)
            errstr += "\tRetrieved = %d" % (numAct)
            raise RuntimeError(errstr)

    # } for ii

    # Convert list of arrays into 1D arrays of all elements
    rads = np.concatenate(rads)
    pots = np.concatenate(pots)
    disp = np.concatenate(disp)

    # Bin Grav Potentials
    counts, aves, stds = zmath.histogram(rads, radBins, weights=pots, 
                                         edges='right', func='ave', stdev=True)
    potsBins[:,0] = aves
    potsBins[:,1] = stds

    # Bin Velocity Dispersion
    counts, aves, stds = zmath.histogram(rads, radBins, weights=disp,
                                         edges='right', func='ave', stdev=True)
    dispBins[:,0] = aves
    dispBins[:,1] = stds


    return radBins, posRef, partTypes, partNames, numsBins, massBins, densBins, potsBins, dispBins

# subhaloRadialProfiles()




def plotSubhaloRadialProfiles(run, snapNum, subhalo, mostBound=None, verbose=VERBOSE):

    #plot1 = False
    plot1 = True
    plot2 = True

    if( verbose ): print " - - Profiler.plotSubhaloRadialProfiles()"

    if( verbose ): print " - - - Loading Profiles"
    radBins, partTypes, massBins, densBins, potsBins, dispBins = \
        subhaloRadialProfiles(run, snapNum, subhalo, mostBound=mostBound)

    partNames = [ PARTICLE.NAMES(pt) for pt in partTypes ]
    numParts = len(partNames)


    ## Figure 1
    #  --------
    if( plot1 ):
        fname = '1_%05d.png' % (subhalo)
        fig1 = plot_1(partNames, radBins, densBins, massBins)
        fig1.savefig(fname)
        plt.close(fig1)
        print fname


    ## Figure 2
    #  --------
    if( plot2 ):
        fname = '2_%05d.png' % (subhalo)
        fig2 = plot_2(radBins, potsBins, dispBins)
        fig2.savefig(fname)
        plt.close(fig2)
        print fname



    return

# plotSubhaloRadialProfiles()

def plot_1(partNames, radBins, densBins, massBins):

    numParts = len(partNames)
    fig, axes = zplot.subplots(figsize=[10,6])
    cols = zplot.setColorCycle(numParts)

    LW = 2.0
    ALPHA = 0.5

    plotBins = np.concatenate([ [zmath.extend(radBins)[0]], radBins] )
    
    for ii in range(numParts):
        zplot.plotHistLine(axes, plotBins, densBins[ii], ls='-',
                           c=cols[ii], lw=LW, alpha=ALPHA, nonzero=True, label=partNames[ii])


    axes.legend(loc='upper right', ncol=1, prop={'size':'small'}, 
                   bbox_transform=axes.transAxes, bbox_to_anchor=(0.99,0.99) )

    return fig

# plot_1()



def plot_2(radBins, potsBins, dispBins):

    FS = 12
    LW = 2.0
    ALPHA = 0.8


    fig, ax = plt.subplots(figsize=[10,6])
    zplot.setAxis(ax, axis='x', label='Distance', fs=FS, scale='log')
    zplot.setAxis(ax, axis='y', label='Dispersion', c='red', fs=FS)
    tw = zplot.twinAxis(ax, axis='x', label='Potential', c='blue', fs=FS)
    tw.set_yscale('linear')

    plotBins = np.concatenate([ [zmath.extend(radBins)[0]], radBins] )
    
    zplot.plotHistLine(ax, plotBins, dispBins[:,0], yerr=dispBins[:,1], ls='-',
                       c='red', lw=LW, alpha=ALPHA, nonzero=True)

    zplot.plotHistLine(tw, plotBins, potsBins[:,0], yerr=potsBins[:,1], ls='-',
                       c='blue', lw=LW, alpha=ALPHA, nonzero=True)

    return fig

# plot_2()



def reflectPos(pos, center=None):
    """
    Given a set of position vectors, reflect those which are on the wrong edge of the box.

    NOTE: Input positions ``pos`` MUST BE GIVEN IN illustris simulation units: [ckpc/h] !!!!
    If a particular ``center`` point is not given, the median position is used.
    
    Arguments
    ---------
    pos    : <float>[N,3], array of ``N`` vectors, MUST BE IN SIMULATION UNITS
    center : <float>[3],   (optional=None), center coordinates, defaults to median of ``pos``

    Returns
    -------
    fix    : <float>[N,3], array of 'fixed' positions with bad elements reflected

    """

    FULL = BOX_LENGTH
    HALF = 0.5*FULL

    # Create a copy of input positions
    fix = np.array(pos)

    # Use median position as center if not provided
    if( center is None ): center = np.median(fix, axis=0)
    else:                 center = ref

    # Find distances to center
    offsets = fix - center

    # Reflect positions which are more than half a box away
    fix[offsets >  HALF] -= FULL
    fix[offsets < -HALF] += FULL

    return fix

# reflectPos()



