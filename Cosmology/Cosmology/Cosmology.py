# ==================================================================================================
# Cosmology.py
# ------------
#
#
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================

import numpy as np
import scipy as sp
from scipy import interpolate
import sys
import os

TIMES_FILE = "../sync-dir/illustris-snapshot-cosmology-data.npz"
BACK_TIMES_FILE = "./data/cosmology/illustris-snapshot-cosmology-data.npz"

INTERP = "quadratic"                                                                                # Type of interpolation for scipy

from Settings import LIB_PATHS
sys.path.append(*LIB_PATHS)
from Constants import BOX_LENGTH, HPAR, H0, SPLC, KPC




class Cosmology(object):
    """
    Class to access cosmological parameters over Illustris simulation times.

    Cosmology loads a preformatted data file which contains cosmological
    parameters at each snapshot.  This class handles accessing those parameters
    or interpolating between them.  The only exposed attributes are accessor
    functions to interface with scipy.interpolate.interp1d objects.


    Functions
    ---------
    redshift(sf)     : redshift [] (analytically)      for given scale-factor 
    lumDist(sf)      : luminosity distance [cm]        for given scale-factor
    comDist(sf)      : comoving distance [cm]          for given scale-factor
    angDist(sf)      : angular diameter distance [cm]  for given scale-factor
    arcSize(sf)      : arcsec size [cm]                for given scale-factor
    lookbackTime(sf) : lookback time [s]               for given scale-factor
    age(sf)          : age of the universe [s]         for given scale-factor
    distMod(sf)      : distance modulus []             for given scale-factor

    Additionally, `Cosmology` supports the `len()` function, which returns the
    number of snapshots.  All cosmological parameters can be accessed for an
    inidividual snapshot using the standard array accessor `[i]`


    Usage
    -----
    import Cosmology

    cosmo = Cosmology.Cosmology()

    # Get the number of snapshots
    numSnaps = len(cosmo)

    # Get all cosmological parameters for 10th to last snapshot
    print cosmo[numSnaps-10]

    # Find the luminosity distance at a scale factor of a=0.5
    lumDist = cosmo.lumDist(0.5)
    
    """


    __REDSHIFT  = 0
    __SCALEFACT = 1
    __HUB_CONST = 2
    __HUB_FUNC  = 3
    __COM_DIST  = 4
    __LUM_DIST  = 5
    __ANG_DIST  = 6
    __ARC_SIZE  = 7
    __LB_TIME   = 8
    __AGE       = 9
    __DIST_MOD  = 10
    
    __NPARS     = 11


    def __init__(self, fname=None): 

        # Load Cosmological Parameters from Data File
        if( fname == None ): fname = TIMES_FILE
        if( not os.path.exists(fname) ): fname = BACK_TIMES_FILE

        self.__loadCosmology(fname)

        # Initialize interpolation functions to None
        self.__redshift = None
        self.__hubbleConstant = None
        self.__hubbleFunction = None
        self.__lumDist  = None
        self.__comDist  = None
        self.__angDist  = None
        self.__arcSize  = None
        self.__lbTime   = None
        self.__age      = None
        self.__distMod  = None

        return

    
    def __getitem__(self, it):
        """ Get all cosmological parameters for snapshit `it` """
        return self.cosmo[it,:]

    def __len__(self): return self.__num
    

    def __initInterp(self, key):
        """ Initialize an interpolation function """
        return sp.interpolate.interp1d( self.cosmo[:, Cosmology.__SCALEFACT], self.cosmo[:, key], kind=INTERP )


    def redshift(self, sf):
        """ Calculate redshift analytically from given scalefactor """
        return (1.0/sf) - 1.0

    def hubbleConstant(self, sf):
        """ Interpolate Hubble Constant to the given scalefactor """
        if( self.__hubbleConstant == None ): self.__hubbleConstant = self.__initInterp(Cosmology.__HUB_CONST)
        return self.__hubbleConstant(sf)

    def hubbleFunction(self, sf):
        """ Interpolate Hubble Function [E(z)] to the given scalefactor """
        if( self.__hubbleFunction == None ): self.__hubbleFunction = self.__initInterp(Cosmology.__HUB_FUNC)
        return self.__hubbleFunction(sf)

    def lumDist(self, sf):
        """ Interpolate luminosity distance to the given scalefactor """
        if( self.__lumDist == None ): self.__lumDist = self.__initInterp(Cosmology.__LUM_DIST)
        return self.__lumDist(sf)

    def comDist(self, sf):
        """ Interpolate comoving distance to the given scalefactor """
        if( self.__comDist == None ): self.__comDist = self.__initInterp(Cosmology.__COM_DIST)
        return self.__comDist(sf)

    def angDist(self, sf):
        """ Interpolate angular-diameter distance to the given scalefactor """
        if( self.__angDist == None ): self.__angDist = self.__initInterp(Cosmology.__ANG_DIST)
        return self.__angDist(sf)

    def arcSize(self, sf):
        """ Interpolate arcsecond size to the given scalefactor """
        if( self.__arcSize == None ): self.__arcSize = self.__initInterp(Cosmology.__ARC_SIZE)
        return self.__arcSize(sf)

    def lookbackTime(self, sf):
        """ Interpolate lookback time to the given scalefactor """
        if( self.__lbTime == None ): self.__lbTime = self.__initInterp(Cosmology.__LB_TIME)
        return self.__lbTime(sf)

    def age(self, sf):
        """ Interpolate age of the universe to the given scalefactor """
        if( self.__age == None ): self.__age = self.__initInterp(Cosmology.__AGE)
        return self.__age(sf)

    def distMod(self, sf):
        """ Interpolate distance modulus to the given scalefactor """
        if( self.__distMod == None ): self.__distMod = self.__initInterp(Cosmology.__DIST_MOD)
        return self.__distMod(sf)


    def __loadCosmology(self, fname=TIMES_FILE):
        dat = np.load(fname)
        self.filename = fname
        self.__num = len(dat['num'])

        self.cosmo = np.zeros([self.__num, Cosmology.__NPARS], dtype=np.float32)

        self.cosmo[:, Cosmology.__REDSHIFT]  = dat['redshift']
        self.cosmo[:, Cosmology.__SCALEFACT] = dat['scale']
        self.cosmo[:, Cosmology.__HUB_CONST] = dat['H']
        self.cosmo[:, Cosmology.__HUB_FUNC]  = dat['E']
        self.cosmo[:, Cosmology.__COM_DIST]  = dat['comDist']
        self.cosmo[:, Cosmology.__LUM_DIST]  = dat['lumDist']
        self.cosmo[:, Cosmology.__ANG_DIST]  = dat['angDist']
        self.cosmo[:, Cosmology.__ARC_SIZE]  = dat['arcsec']
        self.cosmo[:, Cosmology.__LB_TIME]   = dat['lookTime']
        self.cosmo[:, Cosmology.__AGE]       = dat['age']
        self.cosmo[:, Cosmology.__DIST_MOD]  = dat['distMod']

        return


    @staticmethod
    def cosmologicalCorrection(base):
        """

        Simulations only sample a small volume, must compensate and extrapolate to
        actual cosmological event rates.
        For a number per comoving-volume 'R(z)'; the detected rate per redshift is
        dg(z) = R(z)*4*pi*D_H*[D_C^2/E(z)]*[dz/(1+z)]
            for a hubble distance D_H
            comoving distance D_C
            hubble function E(z) s.t. H(z) = H_0 E(z)

        Parameters
        ----------

        Returns
        -------

        """

        ### Get Cosmological Parameters ###

        # Get Comoving Distances
        # comDist = getComDist(cosmo, scales)
        comDist = base.comDist
        # Get redshifts
        # redz = cosmo.redshift(scales)                                                                   # It's okay if redshift is zero here!
        redz = base.redz
        # Get Hubble Function
        #hfunc = cosmo.hubbleFunction(scales)
        hfunc = base.hfunc

        # Get difference in redshift (0th doesn't matter because there are never
        #     mergers there; but assume it is same size as 1st)
        dz = np.ones(len(redz), dtype=np.dtype(redz[0]))
        dz[1:] = redz[:-1]-redz[1:]                                                                     # Remember redshifts are decreasing
        dz[0] = dz[1]

        # Calculate the spatial density of events/objects by dividing by simulation volume
        density = 1.0/np.power(BOX_LENGTH/HPAR, 3.0)

        # Calculate the cosmological volume, and time conversion parameters
        cosmoVolume = 4*np.pi*(SPLC/H0/KPC)*(np.square(comDist/KPC)/hfunc)
        cosmoVolume *= (dz/(1.0+redz))

        # Convert strains to cosmological strains
        cosmoFactor = density*cosmoVolume

        return cosmoFactor

