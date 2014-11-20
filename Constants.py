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
KPC                   = 3.085678e+21                                                                # 1 kpc in cm
MSOL                  = 1.989e+33                                                                   # 1 M_sol in g
MPRT                  = 1.673e-24                                                                   # 1 proton mass in g
MASS_CONV             = 1.0e10/HPAR                                                                 # Convert from e10 Msol to [Msol]
MDOT_CONV             = 10.22                                                                       # Multiply by this to get [Msol/yr]
DENS_CONV             = 6.77025e-22                                                                 # (1e10 Msol/h)/(ckpc/h)^3 to g/cm^3 *COMOVING*
BOX_LENGTH            = 75000                                                                       # [ckpc/h]
FTPI                  = 4.0*np.pi/3.0                                                               # (4.0/3.0)*Pi
NWTG                  = 6.673840e-08                                                                # Newton's Gravitational  Constant
YEAR                  = 3.156e+07                                                                   # Year in seconds
SPLC                  = 2.997925e+10                                                                # Speed of light [cm/s]
H0                    = 2.268546e-18                                                                # Hubble constant at z=0.0   in [1/s]


### Numerical Constants ###
INT = np.int32
LONG = np.int64
FLT = np.float32
DBL = np.float64



### Post-Process (PP) Intermediate Files ###
PP_DIR = "/n/home00/lkelley/illustris/post-process/"
PP_TEMP_DIR = PP_DIR + "temp/"

PP_TIMES_FILENAME = lambda x: PP_DIR + "Illustris-%d/ill-%d_times.npz" % (x,x)
PP_DETAILS_FILENAME = lambda x,y: ( PP_DIR +
                                    "Illustris-%d/bh-details/ill-%d_bh-details_%d.dat" % (x,x,y) )
PP_MERGERS_FILENAME = lambda x: PP_DIR + "Illustris-%d/ill-%d_mergers.dat" % (x,x)
PP_MERGER_DETAILS_FILENAME = lambda x: PP_DIR + "Illustris-%d/bh-mergers/ill-%d_bh-mergers-details.npz" % (x,x)

PP_BH_LIFETIMES_FILENAME = lambda x: PP_DIR + "Illustris-%d/bh-lifetimes/ill-%d_bh-lifetimes.npz" % (x,x)

#PP_BH_DETAILS_DIR            = "bh-details_ill-%d/"
#BH_DETAILS_ASCII_FILENAME    = "ill-%d_details_ascii_%03d.dat"
#BH_DETAILS_OBJ_FILENAME      = "ill-%d_details_obj_%03d.dat"



### Illustris Files ###

NUM_SNAPS             = 136

# Indices for Different Types of Particles
PARTICLE_TYPE_GAS     = 0
PARTICLE_TYPE_DM      = 1
PARTICLE_TYPE_TRAC    = 3
PARTICLE_TYPE_STAR    = 4
PARTICLE_TYPE_BH      = 5
PARTICLE_NAMES        = [ "Gas" , "DM" , "-", "Tracer", "Star", "BH" ]

# File Names and Directories
BH_MERGERS_FILENAMES  = 'output/blackhole_mergers/blackhole_mergers_*.txt'
BH_DETAILS_FILENAMES  = 'output/blackhole_details/blackhole_details_*.txt'
BH_SUMMARY_FILENAME   = 'output/blackholes.txt'

SNAPSHOT_DIRS         = 'output/snapdir_%03d/'
SNAPSHOT_FILENAMES    = 'snap_%03d'

SNAPSHOT_NAMES = lambda x,y: '/n/hernquistfs1/Illustris/Runs/Illustris-%d/output/snapdir_%03d/snap_%03d' % (x,y,y)
GROUP_CAT_NAMES = lambda x,y: '/n/hernquistfs1/Illustris/Runs/Illustris-%d/output/groups_%03d/fof_subhalo_tab_%03d' % (x,y,y)

#GROUP_CAT_DIRS        = 'output/groups_%03d/'
#GROUP_CAT_FILENAMES   = 'fof_subhalo_tab_%03d'







### Plotting Parameters ###

LW1 = 2.0
LW2 = 0.5
LW3 = 3.0

GRY1 = '0.50'                                                                                       # Medium grey
GRY2 = '0.25'                                                                                       # Dark   grey
GRY3 = '0.75'                                                                                       # Light  grey

TYPE_COLS = [ 'r', 'b', GRY1, GRY2, 'g', 'k' ]

FIG_SIZE = [10,8]
AX_SIZE  = [ [0.84, 0.75] ]
AX_POS   = [ [0.08, 0.12] ]

LEG_POS  = [0.50, 0.01]

CB_SIZE  = []
CB_POS   = []
