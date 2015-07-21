"""
Numerical constants pertaining to the Illustris Simulations and their data.

"""

import numpy as np
from zcode.Constants import MSOL, KPC, HPAR

### Physical Constants ###
'''
HPAR             = 0.704                          # Hubble parameter little h
PC               = 3.085678e+18                   # 1 pc  in cm
KPC              = 3.085678e+21                   # 1 kpc in cm
MSOL             = 1.989e+33                      # 1 M_sol in g
MPRT             = 1.673e-24                      # 1 proton mass in g
'''
MASS_CONV        = 1.0e10*MSOL/HPAR               # Convert from e10 Msol to [Msol]
MDOT_CONV        = 10.22                          # Multiply by this to get [Msol/yr]
DENS_CONV        = 6.77025e-22                    # (1e10 Msol/h)/(ckpc/h)^3 to g/cm^3 *COMOVING*
DIST_CONV        = KPC/HPAR                       # Convert from [ckpc/h] to [comoving cm]
CS_CONV          = 1.0                            # ??????? FIX


BOX_LENGTH       = 75000                          # [ckpc/h]

'''
FTPI             = 4.0*np.pi/3.0                  # (4.0/3.0)*Pi
NWTG             = 6.673840e-08                   # Newton's Gravitational  Constant
YR               = 3.156e+07                      # Year in seconds
SPLC             = 2.997925e+10                   # Speed of light [cm/s]
H0               = 2.268546e-18                   # Hubble constant at z=0.0   in [1/s]
SCHW             = 2*NWTG/(SPLC*SPLC)
RHO_CRIT         = 3.0*H0*H0/(4.0*np.pi*NWTG)     # Cosmological Critical Density [g/cm^3

# Derived Physical Constants
# YEAR  = YR
MYR              = 1.0e6*YR
GYR              = 1.0e9*YR
'''

### Numerical Constants ###

class DTYPE():
    ID     = np.uint64
    SCALAR = np.float64
    INDEX  = np.int64


### Illustris Constants ###

NUM_SNAPS              = 136

_DM_MASS = { 1: 4.408965e-04,
             2: 3.527172e-03,
             3: 2.821738e-02 }

def GET_ILLUSTRIS_DM_MASS(run):
    return _DM_MASS[run]


_BAD_SNAPS = { 1: [53, 55],
               2: [],
               3: [] }

def GET_BAD_SNAPS(run):
    return _BAD_SNAPS[run]


_ILLUSTRIS_RUN_NAMES   = { 1 : "L75n1820FP",
                           2 : "L75n910FP",
                           3 : "L75n455FP" }

def GET_ILLUSTRIS_RUN_NAMES(run): 
    return _ILLUSTRIS_RUN_NAMES[run]


_ILLUSTRIS_OUTPUT_DIR_BASE = "/n/ghernquist/Illustris/Runs/%s/output/"

def GET_ILLUSTRIS_OUTPUT_DIR(run): 
    return _ILLUSTRIS_OUTPUT_DIR_BASE % (_ILLUSTRIS_RUN_NAMES[run])




# Indices for Different Types of Particles
PARTICLE_GAS     = 0
PARTICLE_DM      = 1
PARTICLE_TRAC    = 3
PARTICLE_STAR    = 4
PARTICLE_BH      = 5
PARTICLE_TYPES   = [ PARTICLE_GAS,  PARTICLE_DM, PARTICLE_TRAC, PARTICLE_STAR, PARTICLE_BH ]
PARTICLE_NAMES   = [ "Gas" , "DM" , "-", "Tracer", "Star", "BH" ]
PARTICLE_NUM     = 6


class PARTICLE():
    GAS  = 0
    DM   = 1
    TRAC = 3
    STAR = 4
    BH   = 5

    _PARTICLE_NAMES   = [ "Gas" , "DM" , "-", "Tracer", "Star", "BH" ]
    
    @staticmethod
    def PROPERTIES(): 
        return [getattr(PARTICLE,it) for it in vars(PARTICLE) 
                if not it.startswith('_') and not callable(getattr(PARTICLE,it)) ]

    @classmethod
    def NAMES(cls, it):
        return cls._PARTICLE_NAMES[it]

# } class PARTICLE



# Indices for Different Photometric Bands
'''
PHOTO_U               = 0
PHOTO_B               = 1
PHOTO_V               = 2
PHOTO_K               = 3
PHOTO_g               = 4
PHOTO_r               = 5
PHOTO_i               = 6
PHOTO_z               = 7
'''


_PROCESSED_DIR = "/n/home00/lkelley/ghernquistfs1/illustris/data/%s/output/postprocessing/"

def GET_PROCESSED_DIR(run):
    return _PROCESSED_DIR % (_ILLUSTRIS_RUN_NAMES[run])

