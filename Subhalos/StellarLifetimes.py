
import numpy as np
import scipy as sp
import scipy.interpolate


class StellarLifetimes(object):
    """
    Class to convert between stellar masses and main-sequence lifetimes, and visa-versa.
    

    Attributes
    ----------
        MASSES : <float>[N], array of stellar masses [grams]
        LIFETIMES_ROT : <float>[N], array of main-sequence lifetimes for rotational (rot) stars
        LIFETIMES_STA : <float>[N], array of main-sequence lifetimes for stationary (sta) stars


    Functions
    ---------
        lifetime_rotational : <scalar>, 
                              Interpolate from mass to rotational main-sequence lifetime
        
        lifetime_stationary : <scalar>, 
                              Interpolate from mass to stationary (non-rotational) main-sequence lifetime

        lifetime : <scalar>, 
                   Interpolate from mass to averaged main-sequence lifetime

        mass_rotational : <scalar>,
                          Interpolate from rotational main-sequence age to stellar mass

        mass_stationary : <scalar> 
                          Interpolate from non-rotational main-sequence age to stellar mass

        mass : <scalar>,
               Interpolate from average main-sequence age to stellar mass


    Notes
    -----
        http://arxiv.org/abs/1110.5049
        Grids of stellar models with rotation - I. Models from 0.8 to 120 Msun at solar metallicity (Z = 0.014)
        Sylvia Ekstrom, Cyril Georgy, Patrick Eggenberger, et al.
        Table 2

    """
    
    _MASSES        = [ 0.8 , 0.9, 1.0, 1.1, 1.25, 
                       1.35, 1.5, 1.7, 2.0, 2.5 , 
                       3.0 , 4.0, 5.0, 7.0, 9.0  ]

    _LIFETIMES_ROT = [ 22448.55, 13954.45, 8788.403, 5591.467, 4680.58, 
                       3676.43,  2725.402, 2098.732, 1289.272, 673.098, 
                       405.366,  189.417,  109.207,  50.989,   31.211  ]

    _LIFETIMES_STA = [ 21552.716, 13461.100, 8540.320, 5464.714, 4352.893,
                       3221.584,  2241.796,  1633.828, 1008.831, 537.935,
                       320.585,   152.082,   88.193,   41.721,   26.261   ]

    _UNITS_MASS = 1.9891e+33                                                                        # Solar Mass [grams]
    _UNITS_TIME = 3.155690e+13                                                                      # Myr        [sec]



    def __init__(self, kind='quadratic', plot=True):
        
        self.MASSES        = np.array(self._MASSES)       *self._UNITS_MASS
        self.LIFETIMES_ROT = np.array(self._LIFETIMES_ROT)*self._UNITS_TIME
        self.LIFETIMES_STA = np.array(self._LIFETIMES_STA)*self._UNITS_TIME

        # Convert to Log-space
        logMass = np.log10(self.MASSES)
        logSta  = np.log10(self.LIFETIMES_STA)
        logRot  = np.log10(self.LIFETIMES_ROT)
        logAve  = np.average([logSta, logRot], axis=0)

        ### Create Interpolants ###
        self._lin_interp_time_rot = sp.interpolate.interp1d(logMass, logRot, kind=kind)
        self._lin_interp_time_sta = sp.interpolate.interp1d(logMass, logSta, kind=kind)
        self._lin_interp_time     = sp.interpolate.interp1d(logMass, logAve, kind=kind)

        # Arrays must be in order of independent variable, sort
        inds = np.argsort(logAve)
        self._lin_interp_mass_rot = sp.interpolate.interp1d(logRot[inds], logMass[inds], kind=kind)
        self._lin_interp_mass_sta = sp.interpolate.interp1d(logSta[inds], logMass[inds], kind=kind)
        self._lin_interp_mass     = sp.interpolate.interp1d(logAve[inds], logMass[inds], kind=kind)


    def lifetime_rotational(self, mass): 
        """Interpolate from mass to rotational main-sequence lifetime."""
        return np.power(10.0, self._lin_interp_time_rot(np.log10(mass)) )
        
    def lifetime_stationary(self, mass): 
        """Interpolate from mass to stationary (non-rotational) main-sequence lifetime."""
        return np.power(10.0, self._lin_interp_time_sta(np.log10(mass)) )

    def lifetime(self, mass): 
        """Interpolate from mass to averaged main-sequence lifetime."""
        return np.power(10.0, self._lin_interp_time(np.log10(mass)) )

    def mass_rotational(self, age): 
        """Interpolate from rotational main-sequence age to stellar mass"""
        return np.power(10.0, self._lin_interp_mass_rot(np.log10(age)) )

    def mass_stationary(self, age): 
        """Interpolate from non-rotational main-sequence age to stellar mass"""
        return np.power(10.0, self._lin_interp_mass_sta(np.log10(age)) )

    def mass(self, age): 
        """Interpolate from average main-sequence age to stellar mass"""
        return np.power(10.0, self._lin_interp_mass(np.log10(age)) )

                                             
