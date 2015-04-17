"""


"""


import sys
import os
from datetime import datetime

import illpy
from illpy import Cosmology
from illpy.Constants import *
from illpy import AuxFuncs as aux

from Constants import *
import Figures

VERBOSE = True
_VERSION = 0.1


NUM_RAD_BINS = 40
RAD_EXTREMA = [ 1.0, 1.0e6 ]                      # [parsecs]

distConv = DIST_CONV
densConv = DENS_CONV



def getSubhaloRadialProfiles(data, bins=None, verbose=VERBOSE):

    numParts = data[SNAPSHOT_NPART]
    cumNumParts = np.concatenate([[0],np.cumsum(numParts)])
    slices = [ slice( cumNumParts[ii], cumNumParts[ii+1] ) for ii in range(len(numParts)) ]

    # Get Number of Particles
    ngas      = data[SNAPSHOT_NPART][PARTICLE_TYPE_GAS]
    nstars    = data[SNAPSHOT_NPART][PARTICLE_TYPE_STAR]

    ### Get Positions and Radii of Each Particle ###
    bhmass = data[SNAPSHOT_BH_MASS]
    posBH = data[SNAPSHOT_POS][slices[PARTICLE_TYPE_BH  ]]
    # If there are multiple BHs, use the most-massive as the CoM
    if( len(bhmass) > 1 ):
        bhind = np.argmax(bhmass)
        posBH = posBH[bhind]

    # Subtract off BH position (i.e. define BH as center)
    posGas    = data[SNAPSHOT_POS][slices[PARTICLE_TYPE_GAS ]] - posBH
    posDM     = data[SNAPSHOT_POS][slices[PARTICLE_TYPE_DM  ]] - posBH
    posStars  = data[SNAPSHOT_POS][slices[PARTICLE_TYPE_STAR]] - posBH

    radsGas   = np.array([ aux.getMagnitude(vec) for vec in posGas   ])*distConv
    radsDM    = np.array([ aux.getMagnitude(vec) for vec in posDM    ])*distConv
    radsStars = np.array([ aux.getMagnitude(vec) for vec in posStars ])*distConv

    # Create Radial Bins if they are not provided
    if( bins is None ):
        radsExtr  = np.concatenate([radsGas, radsDM, radsStars])
        radsExtr  = aux.extrema(radsExtr, nonzero=True)
        bins  = np.logspace( *np.log10(radsExtr), num=NUM_RAD_BINS+1 )


    # Find average bin positions, and radial bin (shell) volumes
    binAves = np.zeros(NUM_RAD_BINS)
    binVols = np.zeros(NUM_RAD_BINS)
    for ii in range(len(bins)-1):
        binAves[ii] = np.average([bins[ii], bins[ii+1]])
        binVols[ii] = np.power(bins[ii+1],3.0) - np.power(bins[ii],3.0)


    ### Load Masses, Densities and Create Profiles ###
    rhoGas    = data[SNAPSHOT_DENS]*densConv      # Only gas has 'density'/'rho'
    massStars = data[SNAPSHOT_MASS][ngas:(ngas+nstars)]*MASS_CONV
    massDM    = data[SNAPSHOT_MASSES][PARTICLE_TYPE_DM]*MASS_CONV

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
