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


### Plotting Parameters ###

LW1 = 2.0
LW2 = 0.5
LW3 = 3.0

GRY1 = '0.50'                                                                                       # Medium grey
GRY2 = '0.25'                                                                                       # Dark   grey
GRY3 = '0.75'                                                                                       # Light  grey

TYPE_COLS = [ 'r', 'b', GRY1, GRY2, 'g', 'k' ]

FIG_SIZE = [10,8]
AX_SIZE  = [ [0.84, 0.83] ]
AX_POS   = [ [0.08, 0.12] ]

LEG_POS  = [0.50, 0.01]

CB_SIZE  = []
CB_POS   = []
