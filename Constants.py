# ==================================================================================================
# Settings.py
# -----------
#
#
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================

import numpy as np


### Physical Constants ###

HPAR                  = 0.704                                                                       # Hubble parameter little h
MASS_CONV             = 1.0e10                                                                      # Convert from e10 Msol to [Msol]
MDOT_CONV             = 10.22                                                                       # Multiply by this to get [Msol/yr]
BOX_LENGTH            = 75000                                                                       # [ckpc/h]


### Numerical Constants ###
INT = np.int32
LONG = np.int64
FLT = np.float32
DBL = np.float64


### Illustris Files ###

NUM_SNAPS             = 136

# Indices for Different Types of Particles
PARTICLE_TYPE_GAS     = 0
PARTICLE_TYPE_DM      = 1
PARTICLE_TYPE_TRAC    = 3
PARTICLE_TYPE_STAR    = 4
PARTICLE_TYPE_BH      = 5

# File Names and Directories

RUN_DIRS = {
    1:'Illustris-1/' ,                                                                       # 3x1820^3 particles
    2:'Illustris-2/' ,                                                                       # 3x910 ^3    "
    3:'Illustris-3/'                                                                         # 3x455 ^3    "
    }

BH_MERGERS_FILENAMES  = 'output/blackhole_mergers/blackhole_mergers_*.txt'
BH_DETAILS_FILENAMES  = 'output/blackhole_details/blackhole_details_*.txt'
BH_SUMMARY_FILENAME   = 'output/blackholes.txt'

SNAPSHOT_DIRS         = 'output/snapdir_%03d/'
SNAPSHOT_FILENAMES    = 'snap_%03d'

GROUP_CAT_DIRS        = 'output/groups_%03d/'
GROUP_CAT_FILENAMES   = 'fof_subhalo_tab_%03d'


