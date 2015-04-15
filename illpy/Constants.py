# ==================================================================================================
# Constants.py
# ------------
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================

import numpy as np


### Physical Constants ###

HPAR                  = 0.704                                                                       # Hubble parameter little h
PC                    = 3.085678e+18                                                                # 1 pc  in cm
KPC                   = 3.085678e+21                                                                # 1 kpc in cm
MSOL                  = 1.989e+33                                                                   # 1 M_sol in g
MPRT                  = 1.673e-24                                                                   # 1 proton mass in g
MASS_CONV             = 1.0e10/HPAR                                                                 # Convert from e10 Msol to [Msol]
MDOT_CONV             = 10.22                                                                       # Multiply by this to get [Msol/yr]
DENS_CONV             = 6.77025e-22                                                                 # (1e10 Msol/h)/(ckpc/h)^3 to g/cm^3 *COMOVING*
DIST_CONV             = KPC/HPAR                                                                    # Convert from [ckpc/h] to [comoving cm]
CS_CONV               = 1.0                                                                         # ??????? FIX


BOX_LENGTH            = 75000                                                                       # [ckpc/h]
FTPI                  = 4.0*np.pi/3.0                                                               # (4.0/3.0)*Pi
NWTG                  = 6.673840e-08                                                                # Newton's Gravitational  Constant
YR                    = 3.156e+07                                                                   # Year in seconds
SPLC                  = 2.997925e+10                                                                # Speed of light [cm/s]
H0                    = 2.268546e-18                                                                # Hubble constant at z=0.0   in [1/s]
SCHW                  = 2*NWTG/(SPLC*SPLC)
RHO_CRIT              = 3.0*H0*H0/(4.0*np.pi*NWTG)            # Cosmological Critical Density [g/cm^3

# Derived Physical Constants
YEAR = YR
MYR                   = 1.0e6*YR
GYR                   = 1.0e9*YR


### Numerical Constants ###

INT  = np.int32
LNG  = np.int64
FLT  = np.float32
DBL  = np.float64
ULNG = np.uint64



### Illustris Constants ###

NUM_SNAPS             = 136

_ILLUSTRIS_RUN_NAMES   = { 1 : "L75n1820FP",
                           2 : "L75n910FP",
                           3 : "L75n455FP" }

_ILLUSTRIS_OUTPUT_DIR_BASE = "/n/ghernquist/Illustris/Runs/%s/output/"

def GET_ILLUSTRIS_OUTPUT_DIR(run): 
    return _ILLUSTRIS_OUTPUT_DIR_BASE % (_ILLUSTRIS_RUN_NAMES[run])


# GroupFind Files

_ILLUSTRIS_GROUP_PATH_BASE   = '/n/ghernquist/Illustris/Runs/%s/output/groups_%03d/'
_ILLUSTRIS_GROUP_FILENAME_BASE   = 'fof_subhalo_tab_%03d.0.hdf5'

def GET_ILLUSTRIS_GROUPS_FIRST_FILENAME(run,snap): 
 fname  = _ILLUSTRIS_GROUP_PATH_BASE % (_ILLUSTRIS_RUN_NAMES[run],snap)
 fname += _ILLUSTRIS_GROUP_FILENAME_BASE % (snap) 
 return fname


# Snapshot Files

_ILLUSTRIS_SNAPSHOT_PATH_BASE = '/n/ghernquist/Illustris/Runs/%s/output/snapdir_%03d/'
_ILLUSTRIS_SNAPSHOT_FILENAME_BASE   = 'snap_%03d.0.hdf5'

def GET_ILLUSTRIS_SNAPSHOT_FIRST_FILENAME(run,snap): 
    fname  = _ILLUSTRIS_SNAPSHOT_PATH_BASE % (_ILLUSTRIS_RUN_NAMES[run],snap)
    fname += _ILLUSTRIS_SNAPSHOT_FILENAME_BASE % (snap)
    return fname


# Indices for Different Types of Particles
PARTICLE_TYPE_GAS     = 0
PARTICLE_TYPE_DM      = 1
PARTICLE_TYPE_TRAC    = 3
PARTICLE_TYPE_STAR    = 4
PARTICLE_TYPE_BH      = 5
PARTICLE_NAMES        = [ "Gas" , "DM" , "-", "Tracer", "Star", "BH" ]
PARTICLE_TYPE_NUM     = 6

# Indices for Different Photometric Bands
PHOTO_U               = 0
PHOTO_B               = 1
PHOTO_V               = 2
PHOTO_K               = 3
PHOTO_g               = 4
PHOTO_r               = 5
PHOTO_i               = 6
PHOTO_z               = 7
