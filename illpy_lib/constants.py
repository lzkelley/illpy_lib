"""
"""
import numpy as np

HPAR = 0.704
KPC = 3.085677581467192e+21   # kpc in cm
MSOL = 1.9884754153381438e+33   # Solar-mass in grams
YR = 31557600.0   # year in seconds

# Illustris Constants
NUM_SNAPS = 136
BOX_LENGTH = 75000                          # [ckpc/h]
BOX_VOLUME_MPC3 = np.power(BOX_LENGTH*1e-3/HPAR, 3.0)
BOX_VOLUME_CGS = np.power(BOX_LENGTH*KPC/HPAR, 3.0)    # comoving cm^3


_DM_MASS = {1: 4.408965e-04,
            2: 3.527172e-03,
            3: 2.821738e-02}

_BAD_SNAPS = {1: [53, 55],
              2: [],
              3: []}

_ILLUSTRIS_RUN_NAMES   = {1: "L75n1820FP",
                          2: "L75n910FP",
                          3: "L75n455FP"}

_ILLUSTRIS_OUTPUT_DIR_BASE = "/n/ghernquist/Illustris/Runs/%s/output/"

_PROCESSED_DIR = "/n/home00/lkelley/hernquistfs1/illustris/data/%s/output/postprocessing/"


# Physical Constants
class CONV_ILL_TO_CGS(object):
    """Convert from illustris units to physical [cgs] units (multiply).
    """
    MASS = 1.0e10*MSOL/HPAR               # Convert from e10 Msol to [g]
    MDOT = 10.22*MSOL/YR                  # Multiply by this to get [g/s]
    DENS = 6.77025e-22                    # (1e10 Msol/h)/(ckpc/h)^3 to g/cm^3 *COMOVING*
    DIST = KPC/HPAR                       # Convert from [ckpc/h] to [comoving cm]
    VEL  = 1.0e5                          # [km/s] to [cm/s]
    CS   = 1.0                            # ??????? FIX


class CONV_CGS_TO_SOL(object):
    """Convert from cgs units to (standard) solar units, e.g. Msol, PC, etc, by multiplication
    """
    MASS  = 1.0/MSOL                       # [g] ==> Msol
    MDOT  = YR/MSOL                        # [g/s] ==> [Msol/yr]
    DENS  = np.power(KPC/1000.0, 3.0)/MSOL          # [g/cm^3] ==> [Msol/pc^3]
    NDENS = np.power(KPC/1000.0, 3.0)               # [1/cm^3] ==> [1/pc^3]
    DIST  = 1000.0/KPC                         # [cm] ==> [pc]
    VEL   = 1.0e-5                         # [cm/s] ==> [km/s]
    ENER  = 1.0e-10                        # [erg/g] ==> [(km/s)^2]


class CONV_ILL_TO_SOL(object):
    """Convert from illustris units to standard solar units (e.g. Msol, pc), by multiplication
    """
    MASS = CONV_ILL_TO_CGS.MASS.value * CONV_CGS_TO_SOL.MASS.value  # e10 Msol to [Msol]
    MDOT = CONV_ILL_TO_CGS.MDOT.value * CONV_CGS_TO_SOL.MDOT.value  # to [Msol/yr]
    DENS = CONV_ILL_TO_CGS.DENS.value * CONV_CGS_TO_SOL.DENS.value  # to [Msol/pc^3]
    DIST = CONV_ILL_TO_CGS.DIST.value * CONV_CGS_TO_SOL.DIST.value  # to comoving-pc

    VEL = 1.0


# Indices for Different Types of Particles
class PARTICLE(object):
    GAS  = 0
    DM   = 1
    TRAC = 3
    STAR = 4
    BH   = 5

    # _NAMES = ["Gas", "DM", "-", "Tracer", "Star", "BH"]
    # _NUM  = 6


# Numerical Constants
class DTYPE(object):
    ID     = np.uint64
    SCALAR = np.float64
    INDEX  = np.int64


def GET_ILLUSTRIS_DM_MASS(run):
    return _DM_MASS[run]


def GET_BAD_SNAPS(run):
    return _BAD_SNAPS[run]


def GET_ILLUSTRIS_RUN_NAMES(run):
    return _ILLUSTRIS_RUN_NAMES[run]


def GET_ILLUSTRIS_OUTPUT_DIR(run):
    return _ILLUSTRIS_OUTPUT_DIR_BASE % (_ILLUSTRIS_RUN_NAMES[run])


def GET_PROCESSED_DIR(run):
    return _PROCESSED_DIR % (_ILLUSTRIS_RUN_NAMES[run])
