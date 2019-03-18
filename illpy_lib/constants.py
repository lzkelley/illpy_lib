"""
"""
import numpy as np

KPC = 3.085677581467192e+21   # kpc in cm
MSOL = 1.9884754153381438e+33   # Solar-mass in grams
YR = 31557600.0   # year in seconds


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


_ILLUSTRIS_RUN_NAMES   = {1: "L75n1820FP",
                          2: "L75n910FP",
                          3: "L75n455FP"}

_PROCESSED_DIR = "/n/home00/lkelley/hernquistfs1/illustris/data/%s/output/postprocessing/"
_ILLUSTRIS_OUTPUT_DIR_BASE = "/n/ghernquist/Illustris/Runs/%s/output/"

_DM_MASS = {1: 4.408965e-04,
            2: 3.527172e-03,
            3: 2.821738e-02}


def GET_ILLUSTRIS_DM_MASS(run):
    return _DM_MASS[run]


# def GET_BAD_SNAPS(run):
#     return _BAD_SNAPS[run]


def GET_ILLUSTRIS_RUN_NAMES(run):
    return _ILLUSTRIS_RUN_NAMES[run]


def GET_ILLUSTRIS_OUTPUT_DIR(run):
    return _ILLUSTRIS_OUTPUT_DIR_BASE % (_ILLUSTRIS_RUN_NAMES[run])


def GET_PROCESSED_DIR(run):
    return _PROCESSED_DIR % (_ILLUSTRIS_RUN_NAMES[run])
