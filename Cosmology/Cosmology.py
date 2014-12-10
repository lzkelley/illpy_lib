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

from ..Constants import * #BOX_LENGTH, HPAR, H0, SPLC


#TIMES_FILE = "./data/illustris-snapshot-cosmology-data.npz"                                         # Contains cosmological values for each snapshot
TIMES_FILE = "/n/home00/lkelley/illustris/library-code/Cosmology/data/illustris-snapshot-cosmology-data.npz"
INTERP = "quadratic"                                                                                # Type of interpolation for scipy
FLT_TYPE = np.float32


class Cosmology(object):
    '''
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
    hubbleConstant(sf) : hubble constant [km/s/Mpc] 
    hubbleFunction(sf) : hubble function []

    Additionally, `Cosmology` supports the `len()` function, which returns the
    number of snapshots.  The scale-factor for an individual snapshot can be
    retrieved using the standard array accessor `[i]`


    Examples
    --------
    >> import Cosmology
    >> cosmo = Cosmology.Cosmology()

    >> # Get the number of snapshots
    >> numSnaps = len(cosmo)

    >> # Get the scale-factor for the 10th from last snapshot
    >> print cosmo[numSnaps-10]

    >> # Find the luminosity distance at a scale factor of a=0.5
    >> lumDist = cosmo.lumDist(0.5)
    
    '''

    __REDSHIFT  = 'redshift'
    __SCALEFACT = 'scale'
    __HUB_CONST = 'H'
    __HUB_FUNC  = 'E'
    __COM_DIST  = 'comDist'
    __LUM_DIST  = 'lumDist'
    __ANG_DIST  = 'angDist'
    __ARC_SIZE  = 'arcsec'
    __LB_TIME   = 'lookTime'
    __AGE       = 'age'
    __DIST_MOD  = 'distMod'

    __NUM       = 'num'


    def __init__(self, fname=None):

        # Load Cosmological Parameters from Data File
        if( fname == None ): fname = TIMES_FILE
        self.cosmo = np.load(fname)
        self.filename = fname
        self.__num = len(self.cosmo[self.__NUM])

        return

    
    def __getitem__(self, it):
        ''' Get scalefactor for a given snapshot number '''
        return self.cosmo[self.__SCALEFACT][it]


    def __len__(self): 
        ''' Return number of snapshots '''
        return self.__num
    

    def __initInterp(self, key):
        ''' Initialize an interpolation function '''
        return sp.interpolate.interp1d(self.cosmo[self.__SCALEFACT], self.cosmo[key], kind=INTERP)


    def __validSnap(self, snap):
        """
        Check if the given scalar is a valid snapshot number.

        If argument ``snap`` is an integer, 
        """

        # Make sure we can get the numpy dtype
        nsnap = np.array(snap)

        # If this is not an integer, false
        if( not np.issubdtype(nsnap.dtype, int) ):
            return False
        # If this is within number of snapshots, true
        elif( nsnap >= 0 and nsnap <= self.__num ):
            return True
        # outside range, false
        else:
            return False


    def redshift(self, sf):
        ''' Calculate redshift analytically from the given scalefactor '''
        return (1.0/sf) - 1.0


    def __parameter(self, sf, key):
        ''' 
        Retrieve a target parameter at a certain snapshot or scalefactor

        Arguments
        ---------
        sf : int or float
            If `int`, interpreted as a snapshot number
            otherwise interpreted as a (`float`) scalefactor
            
        key : str
            string key for ``self.cosmo`` dictionary of cosmological values


        Returns
        -------
        float, the value of the target parameter


        Raises
        ------
            ValueError: if ``sf`` is outside of the scalefactor range
            KeyError: if ``key`` is not in the ``self.cosmo`` dictionary

        '''

        # If this is a snapshot number, return value from that snapshot
        if( self.__validSnap(sf) ): 
            return self.cosmo[key][sf]
        # Otherwise, interpolate to given scale-factor
        else:
            # If interpolation function for this parameter hasn't been
            #     initialized, initialize it now
            if( not hasattr(self, key) ): 
                setattr(self, key, self.__initInterp(key))

            # Return interpolated value
            return getattr(self, key)(sf)


    def hubbleConstant(self, sf):
        ''' Get Hubble Constant for given snapshot or scalefactor '''
        return self.__parameter(sf, self.__HUB_CONST)

    def hubbleFunction(self, sf):
        ''' Get Hubble Function [E(z)] for given snapshot or scalefactor '''
        return self.__parameter(sf, self.__HUB_FUNC)

    def lumDist(self, sf):
        ''' Get luminosity distance for given snapshot or scalefactor '''
        return self.__parameter(sf, self.__LUM_DIST)

    def comDist(self, sf):
        ''' Get comoving distance for given snapshot or scalefactor '''
        return self.__parameter(sf, self.__COM_DIST)

    def angDist(self, sf):
        ''' Get angular-diameter distance for given snapshot or scalefactor '''
        return self.__parameter(sf, self.__ANG_DIST)

    def arcSize(self, sf):
        ''' Get arcsecond size for given snapshot or scalefactor '''
        return self.__parameter(sf, self.__ARC_SIZE)

    def lookbackTime(self, sf):
        ''' Get lookback time for given snapshot or scalefactor '''
        return self.__parameter(sf, self.__LB_TIME)

    def age(self, sf):
        ''' Get age of the universe for given snapshot or scalefactor '''
        return self.__parameter(sf, self.__AGE)

    def distMod(self, sf):
        ''' Get distance modulus for given snapshot or scalefactor '''
        return self.__parameter(sf, self.__DIST_MOD)



    def cosmologicalCorrection(self, sf):
        '''
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
        '''
        
        # Generalize argument to always be iterable
        if( not np.iterable(sf) ): sf = np.array([sf])

        ### Get Cosmological Parameters ###
        nums = len(sf)
        comDist = np.zeros(nums, dtype=FLT_TYPE)
        redz    = np.zeros(nums, dtype=FLT_TYPE)
        hfunc   = np.zeros(nums, dtype=FLT_TYPE)

        # For each input scale factor
        for ii,scale in enumerate(sf):
            # Get Comoving Distances
            comDist[ii] = self.comDist(scale)
            # Get redshifts
            redz[ii] = self.redshift(scale)
            # Get Hubble Function
            hfunc[ii] = self.hubbleFunction(scale)


        # Get difference in redshift (0th doesn't matter because there are never
        #     mergers there; but assume it is same size as 1st)
        dz = np.ones(len(redz), dtype=np.dtype(redz[0]))
        dz[1:] = redz[:-1]-redz[1:]                                                                 # Remember redshifts are decreasing in value
        dz[0] = dz[1]                                                                               

        # Calculate the spatial density of events/objects by dividing by simulation volume
        density = 1.0/np.power(BOX_LENGTH/HPAR, 3.0)

        # Calculate the cosmological volume, and time conversion parameters
        cosmoVolume = 4*np.pi*(SPLC/H0/KPC)*(np.square(comDist/KPC)/hfunc)
        cosmoVolume *= (dz/(1.0+redz))

        # Convert strains to cosmological strains
        cosmoFactor = density*cosmoVolume

        return cosmoFactor

    
