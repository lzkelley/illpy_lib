"""


"""


import sys
import os
from datetime import datetime

import numpy as np
import scipy as sp
from scipy import optimize

import illpy
from illpy import Cosmology
from illpy.Constants import *
from illpy import AuxFuncs as aux

from Constants import *

VERBOSE = True
_VERSION = 0.1


NUM_RAD_BINS = 40
RAD_EXTREMA = [ 1.0, 1.0e6 ]                      # [parsecs]


def getSubhaloRadialProfiles(data, bins=None, nbins=NUM_RAD_BINS, verbose=VERBOSE):

    if( verbose ): print " - - Profiler.getSubhaloRadialProfiles()"

    numParts = data[SNAPSHOT_NPART]
    cumNumParts = np.concatenate([[0],np.cumsum(numParts)])
    slices = [ slice( cumNumParts[ii], cumNumParts[ii+1] ) for ii in range(len(numParts)) ]

    # Get Number of Particles
    ngas      = data[SNAPSHOT_NPART][PARTICLE_TYPE_GAS]
    nstars    = data[SNAPSHOT_NPART][PARTICLE_TYPE_STAR]
    ndm       = data[SNAPSHOT_NPART][PARTICLE_TYPE_DM]

    if( verbose ): print " - - - Number of DM %d, Stars %d, Gas %d" % (ndm, nstars, ngas)


    ### Load Masses, Densities and Create Profiles ###
    rhoGas    = data[SNAPSHOT_DENS]*DENS_CONV      # Only gas has 'density'/'rho'
    massGas   = data[SNAPSHOT_MASS][:ngas]*MASS_CONV
    massStars = data[SNAPSHOT_MASS][ngas:(ngas+nstars)]*MASS_CONV
    massDM    = data[SNAPSHOT_MASSES][PARTICLE_TYPE_DM]*MASS_CONV


    posAll = data[SNAPSHOT_POS]
    posAll = reflectPos(posAll)


    ### Get Positions and Radii of Each Particle ###
    bhmass = data[SNAPSHOT_BH_MASS]
    posBH = posAll[slices[PARTICLE_TYPE_BH  ]]
    # If there are multiple BHs, use the most-massive as the CoM
    if( len(bhmass) > 1 ):
        bhind = np.argmax(bhmass)
        posBH = posBH[bhind]


    posGas    = posAll[slices[PARTICLE_TYPE_GAS ]]
    posDM     = posAll[slices[PARTICLE_TYPE_DM  ]]
    posStars  = posAll[slices[PARTICLE_TYPE_STAR]]



    # Select a center position
    # posCenter = posBH
    # posCenter = posAve
    posCenter = np.median(posStars, axis=0)

    # Subtract off BH position
    posGas   -= posCenter
    posDM    -= posCenter
    posStars -= posCenter

    radsGas   = np.array([ aux.getMagnitude(vec) for vec in posGas   ])*DIST_CONV
    radsDM    = np.array([ aux.getMagnitude(vec) for vec in posDM    ])*DIST_CONV
    radsStars = np.array([ aux.getMagnitude(vec) for vec in posStars ])*DIST_CONV

    # Create Radial Bins if they are not provided
    if( bins is None ):
        radsExtr  = np.concatenate([radsGas, radsDM, radsStars])
        radsExtr  = aux.extrema(radsExtr, nonzero=True)
        bins      = np.logspace( *np.log10(radsExtr), num=nbins+1 )


    # Find average bin positions, and radial bin (shell) volumes
    binAves = np.zeros(nbins)
    binVols = np.zeros(nbins)
    for ii in range(len(bins)-1):
        binAves[ii] = np.average([bins[ii], bins[ii+1]])
        binVols[ii] = np.power(bins[ii+1],3.0) - np.power(bins[ii],3.0)


    colsStars = data[SNAPSHOT_STELLAR_PHOTOS]     # Only stars have photometric entries
    gmr       = np.array([ cols[PHOTO_g] - cols[PHOTO_r] for cols in colsStars])


    ### Create Radial profiles ###

    # Average gas densities in each bin
    hg = aux.histogram(radsGas,   bins, weights=rhoGas,    ave=True )
    # Sum Stellar masses in each bin, divide by shell volumes
    hs = aux.histogram(radsStars, bins, weights=massStars, scale=1.0/binVols)
    # Sum DM Masses in each bin
    hd = aux.histogram(radsDM,    bins) 
    # Weight DM profile by masses and volumes
    hd = hd*massDM/binVols
    # Average colors in each bin
    hc = aux.histogram(radsStars, bins, weights=gmr, ave=True )



    return bins, binAves, hg, hs, hd, hc

# getSubhaloRadialProfiles()




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



def fit_powerLaw_broken(xx, yy, inner=None, outer=None):
    """
    Fit a broken power law function to the given data, the inner slope can be fixed.
    """

    # Transform to log-log space and scale towards unity
    y0 = np.max(yy)
    lx = np.log10(xx)
    ly = np.log10(yy/y0)

        
    # Guess Power Law Parameters
    pars0 = [100.0*PC, -1.0, -4.0]
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



