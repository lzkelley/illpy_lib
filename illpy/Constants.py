"""
Numerical constants pertaining to the Illustris Simulations and their data.

"""

import numpy as np
from zcode.Constants import MSOL, PC, KPC, HPAR, YR
from enum import Enum

## Physical Constants
'''
MASS_CONV        = 1.0e10*MSOL/HPAR               # Convert from e10 Msol to [Msol]
MDOT_CONV        = 10.22                          # Multiply by this to get [Msol/yr]
DENS_CONV        = 6.77025e-22                    # (1e10 Msol/h)/(ckpc/h)^3 to g/cm^3 *COMOVING*
DIST_CONV        = KPC/HPAR                       # Convert from [ckpc/h] to [comoving cm]
CS_CONV          = 1.0                            # ??????? FIX
'''

class CONV_ILL_TO_CGS(Enum):
    """
    Convert from illustris units to physical [cgs] units (multiply).
    """
    MASS        = 1.0e10*MSOL/HPAR               # Convert from e10 Msol to [Msol]
    MDOT        = 10.22                          # Multiply by this to get [Msol/yr]
    DENS        = 6.77025e-22                    # (1e10 Msol/h)/(ckpc/h)^3 to g/cm^3 *COMOVING*
    DIST        = KPC/HPAR                       # Convert from [ckpc/h] to [comoving cm]
    CS          = 1.0                            # ??????? FIX


class CONV_CGS_TO_SOL(Enum):
    """
    Convert from cgs units to (standard) solar units, e.g. Msol, PC, etc, by multiplication
    """
    MASS        = 1.0/MSOL                       # [g] ==> Msol
    MDOT        = YR/MSOL                        # [g/s] ==> [Msol/yr]
    DENS        = np.power(PC,3.0)/MSOL          # [g/cm^3] ==> [Msol/pc^3]
    DIST        = 1.0/PC                         # [cm] ==> [pc]
    VEL         = 1.0e-5                         # [cm/s] ==> [km/s]


class CONV_ILL_TO_SOL(Enum):
    """
    Convert from illustris units to standard solar units (e.g. Msol, pc), by multiplication
    """
    MASS        = CONV_ILL_TO_CGS.MASS.value*CONV_CGS_TO_SOL.MASS.value  # e10 Msol to [Msol]
    MDOT        = CONV_ILL_TO_CGS.MDOT.value*CONV_CGS_TO_SOL.MDOT.value  #  to [Msol/yr]
    DENS        = CONV_ILL_TO_CGS.DENS.value*CONV_CGS_TO_SOL.DENS.value  #  to [Msol/pc^3]
    DIST        = CONV_ILL_TO_CGS.DIST.value*CONV_CGS_TO_SOL.DIST.value  #  to comoving-pc


BOX_LENGTH       = 75000                          # [ckpc/h]


## Numerical Constants

class DTYPE():
    ID     = np.uint64
    SCALAR = np.float64
    INDEX  = np.int64


## Illustris Constants

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
'''
PARTICLE_GAS     = 0
PARTICLE_DM      = 1
PARTICLE_TRAC    = 3
PARTICLE_STAR    = 4
PARTICLE_BH      = 5
PARTICLE_TYPES   = [ PARTICLE_GAS,  PARTICLE_DM, PARTICLE_TRAC, PARTICLE_STAR, PARTICLE_BH ]
PARTICLE_NAMES   = [ "Gas" , "DM" , "-", "Tracer", "Star", "BH" ]
PARTICLE_NUM     = 6
'''

class PARTICLE():
    GAS  = 0
    DM   = 1
    TRAC = 3
    STAR = 4
    BH   = 5

    _NAMES   = [ "Gas" , "DM" , "-", "Tracer", "Star", "BH" ]
    _NUM     = 6
    
    '''
    @staticmethod
    def PROPERTIES(): 
        return [getattr(PARTICLE,it) for it in vars(PARTICLE) 
                if not it.startswith('_') and not callable(getattr(PARTICLE,it)) ]

    @classmethod
    def NAMES(cls, it):
        return cls._PARTICLE_NAMES[it]
    '''

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

