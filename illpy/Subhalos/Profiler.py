"""


"""

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import illpy
from illpy.Constants import GET_ILLUSTRIS_DM_MASS, PARTICLE, DTYPE, BOX_LENGTH

import Subhalo
import Constants
from Constants import SNAPSHOT

import zcode
import zcode.Math as zmath
import zcode.Plotting as zplot

VERBOSE = True

NUM_RAD_BINS = 40
RAD_EXTREMA = [ 1.0, 1.0e6 ]                      # [parsecs]



### Subhalo Profiles ###
'''
PROFILE_BIN_EDGES = 'bin_edges'
PROFILE_BIN_AVES  = 'bin_aves'
PROFILE_GAS       = 'gas'
PROFILE_STARS     = 'stars'
PROFILE_DM        = 'dm'
PROFILE_COLS      = 'cols'
PROFILE_CREATED   = 'created'
PROFILE_VERSION   = 'version'
'''




def subhaloRadialProfiles(run, snapNum, subhalo, bins=None, nbins=NUM_RAD_BINS, 
                          refid=None, verbose=VERBOSE, debug=False):

    if( verbose ): print " - - Profiler.subhaloRadialProfiles()"

    if( verbose ): print " - - - Loading subhalo partile data"
    partData, partTypes = Subhalo.importSubhaloParticles(run, snapNum, subhalo, verbose=False)
    partNums = [ pd['count'] for pd in partData ]
    partNames = [ PARTICLE.NAMES(pt) for pt in partTypes ]
    numPartTypes = len(partNums)
    if( verbose ):
        print " - - - - Run %d, Snap %d, Subhalo %d : Loaded %s particles" % \
            (run, snapNum, subhalo, str(partNums))


    # Get the mass of each DM particle for this run (hard-coded in illpy.Constants)
    massPerDM = GET_ILLUSTRIS_DM_MASS(run)

    pos_com = np.zeros([numPartTypes, 3], dtype=DTYPE.SCALAR)
    
    pos = []; pot = []; vdisp = []; mass = []
    # Iterate over all particle types and their data
    if( debug ): print " - - - Extracting particle properties"
    for ii, (data, ptype) in enumerate(zip(partData, partTypes)):
        # ids_p = data[SNAPSHOT.IDS]
        pos_p   = reflectPos(data[SNAPSHOT.POS])
        pot_p   = data[SNAPSHOT.POT]
        vdisp_p = data[SNAPSHOT.SUBF_VDISP]
        # DarkMatter Particles
        if( ptype == PARTICLE.DM ): 
            mass_p = massPerDM
            for jj in range(3): pos_com[ii, jj] = np.average( pos_p[:,jj] )
        # Other Particles
        else:
            mass_p = data[SNAPSHOT.MASS]
            for jj in range(3): pos_com[ii, jj] = np.average( pos_p[:,jj], weights=mass_p )

        pos.append(pos_p)
        pot.append(pot_p)
        vdisp.append(vdisp_p)
        mass.append(mass_p)
        
    # } for data, ptype


    refCOM = None
    if( refid is not None ):
        for pdat,pname in zip(partData, partNames):
            inds = np.where( pdat[SNAPSHOT.IDS] == refid )[0]
            if( len(inds) == 1 ): 
                if( verbose ): print " - - - Found Most Bound Particle in '%s'" % (pname)
                refCOM = pdat[SNAPSHOT.POS][inds]
                break
    
        # } pdat,pname

    '''
    if( refCOM is not None ):
        print zmath.dist( pos_com, refCOM )

    return
    '''


    rads  = np.zeros([numPartTypes, numPartTypes], dtype=object)
    radExtrema = None

    potsAll = []
    radsAll = []
    dispAll = []

    # Iterate over COM of particles ``ii``
    if( debug ): print " - - - Finding Radii for each COM"
    for ii, pt1 in enumerate(partTypes):

        # Iterate over particles ``jj`` to find postitions relative to COM
        for jj, pt2 in enumerate(partTypes):

            # Calculate distances of particles ``jj`` from COM ``ii``
            # rads_p = np.sqrt( np.sum( np.square(pos[jj] - pos_com[ii]), axis=1) )
            rads_p = zmath.dist(pos[jj], pos_com[ii])
            rads[jj,ii] = rads_p

            # Find extrema for all combinations
            radExtrema = zmath.minmax(rads_p, prev=radExtrema, nonzero=True)

            if( jj == 0 ): 
                pot_all = np.array(pot[jj])
                rads_all = np.array(rads_p)
                vdisp_all = np.array(vdisp[jj])
            else:
                pot_all = np.concatenate([pot_all, pot[jj]])
                rads_all = np.concatenate([rads_all, rads_p])
                vdisp_all = np.concatenate([vdisp_all, vdisp[jj]])

        # } for jj, pt2

        potsAll.append(pot_all)
        radsAll.append(rads_all)
        dispAll.append(vdisp_all)

    # } for ii, pt1

    refRads = None
    refPots = None
    refDisp = None
    if( refCOM is not None ):
        # Iterate over particles ``jj`` to find postitions relative to COM
        for jj, pt2 in enumerate(partTypes):

            rads_p = zmath.dist(pos[jj], refCOM)

            if( jj == 0 ): 
                refPots = np.array(pot[jj])
                refRads = np.array(rads_p)
                refDisp = np.array(vdisp[jj])
            else:
                refPots = np.concatenate([refPots, pot[jj]])
                refRads = np.concatenate([refRads, rads_p])
                refDisp = np.concatenate([refDisp, vdisp[jj]])

        # } for jj, pt2


    # Create radial bin spacings, these will be the upper-bound radii
    radBins = zmath.spacing(radExtrema, scale='log', num=nbins)

    # Find average bin positions, and radial bin (shell) volumes
    binVols = np.zeros(nbins)
    for ii in range(len(radBins)):
        if( ii == 0 ): binVols[ii] = np.power(radBins[ii],3.0)
        else:          binVols[ii] = np.power(radBins[ii],3.0) - np.power(radBins[ii-1],3.0)
    # } ii

    densBins = np.zeros([numPartTypes, nbins, numPartTypes], dtype=DTYPE.SCALAR)
    massBins = np.zeros([numPartTypes, nbins, numPartTypes], dtype=DTYPE.SCALAR)

    # Iterate over COM of particles ``ii``
    if( debug ): print " - - - Binning properties by radii"
    for ii, pt1 in enumerate(partTypes):
        # Iterate over properties for particles ``jj``
        for jj, pt2 in enumerate(partTypes):

            # Get the total mass in each bin
            counts, massBins[jj,:,ii] = zmath.histogram(rads[jj,ii], radBins, weights=mass[jj],
                                                        edges='right', func='sum', stdev=False)
            
            # Divide by volume to get density
            densBins[jj,:,ii] = massBins[jj,:,ii]/binVols

        # } for jj, pt2
    # } for ii, pt1

    return radBins, massBins, densBins, radsAll, potsAll, dispAll, refRads, refPots, refDisp, partTypes

# subhaloRadialProfiles()



def plotSubhaloRadialProfiles(run, snapNum, subhalo, refid=None, verbose=VERBOSE):

    plot1 = False
    plot2 = True

    if( verbose ): print " - - Profiler.plotSubhaloRadialProfiles()"

    if( verbose ): print " - - - Loading Profiles"
    radBins, massBins, densBins, radsAll, potsAll, dispAll, refRads, refPots, refDisp, partTypes = \
        subhaloRadialProfiles(run, snapNum, subhalo, refid=refid)

    partNames = [ PARTICLE.NAMES(pt) for pt in partTypes ]
    numParts = len(partNames)


    ## Figure 1
    #  --------
    if( plot1 ):
        fname = '1_%05d.png' % (subhalo)
        fig1 = plot_1(numParts, partNames, radBins, densBins)
        fig1.savefig(fname)
        plt.close(fig1)
        print fname


    ## Figure 2
    #  --------
    if( plot2 ):
        fname = '2_%05d.png' % (subhalo)
        fig2 = plot_2(numParts, partNames, radsAll, potsAll, dispAll, refRads, refPots, refDisp)
        fig2.savefig(fname)
        plt.close(fig2)
        print fname



    return

# plotSubhaloRadialProfiles()

def plot_1(numParts, partNames, radBins, densBins):

    fig, axes = zplot.subplots(figsize=[10,14], nrows=4)
    
    cols = zplot.setColorCycle(numParts)
    # stys = zplot.setLineStyleCycle(numParts)
    
    for ii in range(numParts):

        axes[ii].set_title('COM %s' % partNames[ii])

        for jj in range(numParts):

            ll, = axes[ii].plot(radBins, densBins[jj,:,ii], ls='-', c=cols[jj], lw=2.0, alpha=0.5, 
                                label=partNames[jj])

    axes[0].legend(loc='lower left', ncol=1, prop={'size':'small'}, 
                   bbox_transform=fig.transFigure, bbox_to_anchor=(0.01,0.01) )


    return fig

# plot_1()


def plot_2(numParts, partNames, radsAll, potsAll, dispAll, refRads, refPots, refDisp):

    ALPHA = 0.05
    PS    = 5

    radsExtr = zmath.minmax(radsAll, nonzero=True)
    potsExtr = zmath.minmax(potsAll, nonzero=True)
    dispExtr = zmath.minmax(dispAll, nonzero=True)
    radsBins = zmath.spacing(radsExtr, num=NUM_RAD_BINS+1)[1:]
    plotBins = np.concatenate([[radsExtr[0]],radsBins])
    
    fig, axes = zplot.subplots(figsize=[10,14], nrows=numParts+1, top=0.95, bot=0.05, left=0.05, right=0.90,
                               xlim=radsExtr, ylim=dispExtr, twinylim=potsExtr)
    for ii,ax in enumerate(axes):

        if( ii < numParts ):
            plotRads = radsAll[ii]
            plotDisp = dispAll[ii]
            plotPots = potsAll[ii]
            plotName = partNames[ii]
        else:
            plotRads = refRads
            plotDisp = refDisp
            plotPots = refPots
            plotName = "Most Bound"

        ax.set_title(plotName)
        tw = ax.twinx()
        tw.set_yscale('linear')

        ax.scatter(plotRads, plotDisp, marker='o', s=PS, alpha=ALPHA, c='red',  label='VelDisp')
        tw.scatter(plotRads, plotPots, marker='o', s=PS, alpha=ALPHA, c='blue', label='Potential')

        count, dispHist, dispStd = zmath.histogram(plotRads, radsBins, weights=plotDisp, 
                                                   func='ave', stdev=True, edges='right')

        inds = np.where( dispStd > 0.0 )
        dispStd_ave = np.average( dispStd[inds], weights=count[inds] )
        dispStd_med = np.median( dispStd[inds] )

        zplot.plotHistLine(ax, plotBins, dispHist, yerr=dispStd, c='red', nonzero=True)

        count, potsHist, potsStd = zmath.histogram(plotRads, radsBins, weights=plotPots, 
                                                   func='ave', stdev=True, edges='right')

        inds = np.where( potsStd > 0.0 )
        potsStd_ave = np.average( potsStd[inds], weights=count[inds] )
        potsStd_med = np.median( potsStd[inds] )

        zplot.plotHistLine(tw, plotBins, potsHist, yerr=potsStd, c='blue', nonzero=True)

        print "%10s : %e %e   %e %e" % (plotName, dispStd_ave, dispStd_med, potsStd_ave, potsStd_med)

    # } ii, ax


    axes[0].legend(loc='lower left', ncol=1, prop={'size':'small'}, 
                   bbox_transform=fig.transFigure, bbox_to_anchor=(0.01,0.01) )

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



def powerLaw(rr,y0,r0,alpha):
    """ Single power law ``n(r) = y0*(r/r0)^alpha`` """
    return y0*np.power(rr/r0, alpha)

def powerLaw_ll(lr,ly0,lr0,alpha):
    """ log-log transform of ``n(r) = y0*(r/r0)^alpha`` """
    return ly0 + alpha*(lr - lr0)



def powerLaw_broken(rr,y0,r0,alpha,beta):
    """ Two power-laws linked together piece-wise at the scale radius ``r0`` """
    y1 = (powerLaw(rr,y0,r0,alpha))[rr<=r0]
    y2 = (powerLaw(rr,y0,r0,beta ))[rr> r0]
    yy = np.concatenate((y1,y2))
    return yy

def powerLaw_broken_ll(rr,y0,r0,alpha,beta):
    """ Log-log transform of a broken (piece-wise defined) power-law """
    y1 = (powerLaw_ll(rr,y0,r0,alpha))[rr<=r0]
    y2 = (powerLaw_ll(rr,y0,r0,beta ))[rr> r0]
    yy = np.concatenate((y1,y2))
    return yy




def fit_powerLaw(xx, yy, pars=None):
    """
    Fit the given data with a single power-law function
    
    Notes: the data is first transformed into log-log space, where a linear
           function is fit.  That is transformed back into linear-space and
           returned.
           
    Arguments
    ---------
    xx : <float>[N], independent variable given in normal (linear) space
    yy : <float>[N],   dependent variable given in normal (linear) space
    
    Returns
    -------
    func  : <callable>, fitting function with the fit parameters already plugged-in
    y0    : <float>   , normalization to the fitting function
    pars1 : <float>[2], fit parameters defining the power-law function.
    """
    
    # Transform to log-log space and scale towards unity
    y0 = np.max(yy)                                                                                
    
    lx = np.log10(xx)
    ly = np.log10(yy/y0)
    
    # Guess Power Law Parameters if they are not provided
    if( pars is None ): 
        pars0 = [1.0, -3.0]
    # Convert to log-space if they are provided
    else:                
        pars0 = np.array(pars)
        pars0[0] = np.log10(pars0[0])


    # Do not fit for normalization parameter ``y0``
    func = lambda rr,p0,p1: powerLaw_ll(rr, y0, p0, p1)
    pars1, covar = sp.optimize.curve_fit(func, lx, ly, p0=pars0)
    
    # Transform fit parameters from log-log space, back to normal
    pars1[0] = np.power(10.0, pars1[0])

    # Add global normalization ``y0`` back in
    pars1 = np.insert(pars1, 0, y0)

    # Create fitting function using the determined parameters
    func = lambda rr: powerLaw(rr, *pars1)
    
    # Return function and fitting parameters
    return func, pars1

# fit_powerLaw()



def fit_powerLaw_broken(xx0, yy0, inner=None, outer=None, xlo=None, xhi=None):
    """
    Fit a broken power law function to the given data, the inner slope can be fixed.
    """

    xx = np.array(xx0)
    yy = np.array(yy0)

    ## Select subsample of input arrays

    if( xlo is not None ):
        inds = np.where( xx >= xlo )
        xx = xx[inds]
        yy = yy[inds]

    if( xhi is not None ):
        inds = np.where( xx <= xhi )
        xx = xx[inds]
        yy = yy[inds]


    # Transform to log-log space and scale towards unity
    y0 = np.max(yy)
    lx = np.log10(xx)
    ly = np.log10(yy/y0)

        
    # Guess Power Law Parameters
    
    guess_x0 = np.average(lx)
    guess_x0 = np.power(10.0, guess_x0)

    # pars0 = [100.0*PC, -1.0, -4.0]
    pars0 = [guess_x0, -1.0, -4.0]
    pars0 = np.array(pars0)
    # Convert to log-space
    pars0[0] = np.log10(pars0[0])


    ## Fit all parameters  (``r0``, ``alpha`` and ``beta``)
    if( inner is None and outer is None ):
        func = lambda rr,r0,alp,bet: powerLaw_broken_ll(rr, y0, r0, alp, bet)
        pars1, covar = sp.optimize.curve_fit(func, lx, ly, p0=pars0)
    ## Fir outer profile If ``inner`` is specified
    elif( outer is None ):
        func = lambda rr,r0,bet: powerLaw_broken_ll(rr, y0, r0, inner, bet)
        # Remove inner-guess parameter
        pars0 = np.delete(pars0, 1)
        pars1, covar = sp.optimize.curve_fit(func, lx, ly, p0=pars0)
        # Replace inner parameter with given value
        pars1 = np.insert(pars1, 1, inner)
    ## Fit inner profile If ``outer`` is specified
    elif( inner is None ):
        func = lambda rr,r0,alp: powerLaw_broken_ll(rr, y0, r0, alp, outer)
        # Remove outer-guess parameter
        pars0 = np.delete(pars0, 2)
        pars1, covar = sp.optimize.curve_fit(func, lx, ly, p0=pars0)
        # Replace inner parameter with given value
        pars1 = np.insert(pars1, 2, outer)
    # Only fit break radius
    else:
        func = lambda rr,r0: powerLaw_broken_ll(rr, y0, r0, inner, outer)
        # Remove guess parameters for slopes
        pars0 = np.delete(pars0, 2)
        pars0 = np.delete(pars0, 1)
        pars1, covar = sp.optimize.curve_fit(func, lx, ly, p0=pars0)
        # Replace parameters with given values
        pars1 = np.insert(pars1, 1, inner)
        pars1 = np.insert(pars1, 2, outer)



    # Transform fit parameter ``r0`` from log-log space, back to normal space
    pars1[0] = np.power(10.0, pars1[0])

    # Add global normalization back into parameters
    pars1 = np.insert(pars1, 0, y0)

    # Create fitting function
    func = lambda rr: powerLaw_broken(rr, *pars1)
    
    # Return function and fitting parameters
    return func, pars1

# fit_powerLaw_broken()



