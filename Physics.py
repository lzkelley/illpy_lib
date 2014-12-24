# ==================================================================================================
# Physics.py
# ----------
#
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================


### Builtin Modules ###
import numpy as np
import scipy as sp
import random

import warnings
warnings.simplefilter('error')                                                                      # Throw Error on Warnings

from Constants import *



def calculateStrain(masses, dist, freq):
    """
    Calculate the Gravitational Wave strain from a binary.

    Use the expression from 'W&L 2003' to find the GW strain from a binary,
    averaged over orientations and polarizations and assuming a circular orbit.

    Wyithe & Loeb, 2003
    Low-Frequency Gravitational Waves from Massive Black Hole Binaries
    http://adsabs.harvard.edu/abs/2003ApJ...590..691W
      Equation 8; pg. 695

    Parameters
    ----------

    Returns
    -------

    """
    const = 8.0*np.sqrt(2.0/15.0)*np.power(NWTG, 5.0/3.0)/np.power(SPLC, 4.0)
    mterm = np.product(masses, axis=1)/np.power(np.sum(masses,axis=1), 1.0/3.0)
    hc = np.power(2.0*np.pi*freq, 2.0/3.0)*const*mterm/dist
    return hc




def criticalSeparation(m1, m2=None):
    """
    Find the critical BH separation when coalescence occurs.

    Find the orbital separation of 3 Schwarzschild radii: the ISCO.
    After this point, we will stop evolving the system and assume it coalesces.

    Parameters
    ----------
    m1 : scalar, (array[N])
        mass of first BH

    m2 : scalar, (array[N] same length as `m1`)
        mass of second BH

    Returns
    -------
    isco : scalar, (array : length of input)
        Last stable orbital separation for BH binary

    """

    if( m2 == None ): useMass = np.sum(m1, axis=1)
    else:             useMass = m1+m2

    const = 2.0*NWTG/np.power(SPLC,2.0)
    lso = 3.0*const*(useMass)

    return lso



def schwarzschildRadius(mass):
    const = 2.0*NWTG/np.square(SPLC)
    return const*mass


def getFrequencyFromSeparation(masses, aa):
    freqs = (1.0/(2.0*np.pi))*np.sqrt( NWTG*np.sum(masses,axis=1) )/np.power(aa,1.5)
    return freqs


def getSeparationFromFrequency(masses, ff):
    seps = np.power( NWTG*np.sum(masses,axis=1) / np.square(2.0*np.pi*ff) , 1.0/3.0 )
    return seps

