"""

"""

import os
import numpy as np
import scipy as sp
from scipy import interpolate
import astropy as ap
import astropy.cosmology

from ..Constants import BOX_LENGTH
from zcode.constants import HPAR, H0, SPLC, KPC

# Get local path, and data directory
_DATA_PATH = "%s/data/" % os.path.dirname(os.path.abspath(__file__))
# Contains cosmological values for each snapshot
_TIMES_FILENAME = "illustris-snapshot-cosmology-data.npz"

INTERP = "quadratic"                     # Type of interpolation for scipy
FLT_TYPE = np.float32
IMPOSE_FLOOR = True                # Enforce a minimum for certain parameters (e.g. comDist)
MAXIMUM_SCALE_FACTOR = 0.9999      # Scalefactor for minimum of certain parameters (e.g. comDist)


class Illustris_Cosmology(ap.cosmology.FlatLambdaCDM):
    """Astropy cosmology object with illustris parameters and additional functions and wrappers.

    Methods
    -------
    -   scale_to_age         - Convert from scale-factor to age of the universe [seconds].
    -   age_to_scale         - Convert from age of the universe [seconds] to scale-factor.

    """
    Omega0 = 0.2726
    OmegaLambda = 0.7274
    OmegaBaryon = 0.0456
    HubbleParam = 0.704
    H0 = HubbleParam * 100.0

    _Z_GRID = [10.0, 4.0, 2.0, 1.0, 0.5, 0.1]

    def __init__(self, max_scale=None, interp_grid=100):
        # Initialize parent class
        ap.cosmology.FlatLambdaCDM.__init__(
            self, H0=self.H0, Om0=self.Omega0, Ob0=self.OmegaBaryon)

        # Load illustris snapshot scale factors
        fname = os.path.join(_DATA_PATH, _TIMES_FILENAME)
        cosmo_data = np.load(fname)
        self.snapshot_scales = cosmo_data['scale']

        # Impose maximum scale-factor
        if max_scale:
            if max_scale < self.snapshot_scales[-2]:
                err_str = "``max_scale = {}`` must be greater than penultimate snapshot scale = {}"
                err_str = err_str.format(max_scale, self.snapshot_scales[-2])
                raise ValueError(err_str)

            self.snapshot_scales[-1] = np.min([max_scale, self.snapshot_scales[-1]])

        # Create grids for interpolations
        # -------------------------------
        #    Set the first point to be the highest redshift of the snapshots
        min_scale = np.min(self.snapshot_scales)
        max_redz = self._scale_to_z(min_scale)
        z_grid_pnts = np.append(max_redz, self._Z_GRID)
        #    Make sure grid is monotonically decreasing
        if not np.all(np.diff(z_grid_pnts) < 0.0):
            err_str = "Non-monotonic z_grid = {}".format(z_grid_pnts)
            err_str += "\nMin snapshot scale = {}, redshift = {}".format(min_scale, max_redz)
            raise ValueError(err_str)

        z0 = z_grid_pnts[0]
        zgrid = None
        for z1 in z_grid_pnts[1:]:
            temp = np.logspace(*np.log10([z0, z1]), num=interp_grid, endpoint=False)
            if zgrid is None:
                zgrid = temp
            else:
                zgrid = np.append(zgrid, temp)
            z0 = z1

        zgrid = np.append(zgrid, np.linspace(z0, 0.0, num=interp_grid))
        self._zgrid = np.sort(zgrid)
        # Ages in seconds
        self._age_grid = self.age(zgrid).cgs.value
        return

    @staticmethod
    def _scale_to_z(sf):
        """Convert from scale-factor to redshift.
        """
        sf = np.array(sf)
        return (1.0/sf) - 1.0

    def scale_to_age(self, sf):
        """Convert from scale-factor to age of the universe [seconds].
        """
        zz = self._scale_to_z(sf)
        return self.age(zz).cgs.value

    def age_to_scale(self, age):
        """Convert from age of the universe [seconds] to scale-factor.
        """
        zz = np.interp(age, self._age_grid, self._zgrid, left=np.nan, right=np.nan)
        return self.scale_factor(zz)


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
    hubbleConstant(sf) : hubble constant [km/s/Mpc]
    hubbleFunction(sf) : hubble function []

    scales(num)    : scale-factor of the given snapshot number



    Additionally, `Cosmology` supports the `len()` function, which returns the
    number of snapshots.  The scale-factor for an individual snapshot can be
    retrieved using the standard array accessor `[i]`.  This is equivalent to
    the `scales()` method.


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

    """

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


    def __init__(self):

        # Construct path to data file
        fname = os.path.join(_DATA_PATH, _TIMES_FILENAME)

        # Load Cosmological Parameters from Data File
        self.__cosmo = np.load(fname)
        self.filename = fname
        self.num = len(self.__cosmo[self.__NUM])

        return


    def __getitem__(self, it):
        """ Get scalefactor for a given snapshot number """
        return self.scales(it)


    def __len__(self):
        """ Return number of snapshots """
        return self.num


    def __initInterp(self, key):
        """ Initialize an interpolation function """
        return sp.interpolate.interp1d(self.__cosmo[self.__SCALEFACT], self.__cosmo[key], kind=INTERP)


    def __validSnap(self, snap):
        """
        Check if the given scalar is a valid snapshot number.

        If argument ``snap`` is an integer,
        """

        # Make sure we can get the numpy dtype
        nsnap = np.array(snap)

        # If this is not an integer, false
        if(not np.issubdtype(nsnap.dtype, int)):
            return False
        # If this is within number of snapshots, true
        elif(nsnap >= 0 and nsnap <= self.num):
            return True
        # outside range, false
        else:
            return False

    '''
    def snapshotTimes(self, num=None):
        """ Get scalefactor for all snapshots, or given snapshot number """
        if(num == None): return self.__cosmo[self.__SCALEFACT]
        else:              return self.__cosmo[self.__SCALEFACT][num]
    '''

    def scales(self, num=None):
        """ Get scalefactor for all snapshots, or given snapshot number """
        if(num is None): return self.__cosmo[self.__SCALEFACT]
        else:              return self.__cosmo[self.__SCALEFACT][num]

    def redshift(self, sf):
        """ Calculate redshift analytically from the given scalefactor """
        return (1.0/sf) - 1.0


    @staticmethod
    def zToA(redz): return 1.0/(1.0+redz)

    @staticmethod
    def aToZ(sf): return (1.0/sf) - 1.0



    def __parameter(self, sf, key):
        """
        Retrieve a target parameter at a certain snapshot or scalefactor

        Arguments
        ---------
        sf : int or float
            If `int`, interpreted as a snapshot number
            otherwise interpreted as a (`float`) scalefactor

        key : str
            string key for ``self.__cosmo`` dictionary of cosmological values


        Returns
        -------
        float, the value of the target parameter


        Raises
        ------
            ValueError: if ``sf`` is outside of the scalefactor range
            KeyError: if ``key`` is not in the ``self.__cosmo`` dictionary

        """

        # If this is a snapshot number, return value from that snapshot
        if(self.__validSnap(sf)):
            return self.__cosmo[key][sf]
        # Otherwise, interpolate to given scale-factor
        else:
            # If interpolation function for this parameter hasn't been
            #     initialized, initialize it now.
            #     Use uppercase attributes
            attrKey = "__" + key.upper()
            if(not hasattr(self, attrKey)):
                setattr(self, attrKey, self.__initInterp(key))

            # Return interpolated value
            return getattr(self, attrKey)(sf)


    def hubbleConstant(self, sf):
        """ Get Hubble Constant for given snapshot or scalefactor """
        return self.__parameter(sf, self.__HUB_CONST)

    def hubbleFunction(self, sf):
        """ Get Hubble Function [E(z)] for given snapshot or scalefactor """
        return self.__parameter(sf, self.__HUB_FUNC)

    def lumDist(self, sf):
        """ Get luminosity distance for given snapshot or scalefactor """
        return self.__parameter(sf, self.__LUM_DIST)

    def comDist(self, sf):
        """ Get comoving distance for given snapshot or scalefactor """
        return self.__parameter(sf, self.__COM_DIST)

    def angDist(self, sf):
        """ Get angular-diameter distance for given snapshot or scalefactor """
        return self.__parameter(sf, self.__ANG_DIST)

    def arcSize(self, sf):
        """ Get arcsecond size for given snapshot or scalefactor """
        return self.__parameter(sf, self.__ARC_SIZE)

    def lookbackTime(self, sf):
        """ Get lookback time for given snapshot or scalefactor """
        return self.__parameter(sf, self.__LB_TIME)

    def age(self, sf):
        """ Get age of the universe for given snapshot or scalefactor """
        return self.__parameter(sf, self.__AGE)

    def distMod(self, sf):
        """ Get distance modulus for given snapshot or scalefactor """
        return self.__parameter(sf, self.__DIST_MOD)



    def cosmologicalCorrection(self, sf):
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

        # Generalize argument to always be iterable
        if(not np.iterable(sf)): sf = np.array([sf])

        ### Get Cosmological Parameters ###
        nums = len(sf)
        comDist = np.zeros(nums, dtype=FLT_TYPE)
        redz    = np.zeros(nums, dtype=FLT_TYPE)
        hfunc   = np.zeros(nums, dtype=FLT_TYPE)

        # For each input scale factor
        for ii, scale in enumerate(sf):
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

    # cosmologicalCorrection()

# } class Cosmology
