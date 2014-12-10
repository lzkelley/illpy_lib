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
YEAR                  = 3.156e+07                                                                   # Year in seconds
SPLC                  = 2.997925e+10                                                                # Speed of light [cm/s]
H0                    = 2.268546e-18                                                                # Hubble constant at z=0.0   in [1/s]


### Numerical Constants ###

INT = np.int32
LONG = np.int64
FLT = np.float32
DBL = np.float64


### Illustris Constants ###

NUM_SNAPS             = 136

# Indices for Different Types of Particles
PARTICLE_TYPE_GAS     = 0
PARTICLE_TYPE_DM      = 1
PARTICLE_TYPE_TRAC    = 3
PARTICLE_TYPE_STAR    = 4
PARTICLE_TYPE_BH      = 5
PARTICLE_NAMES        = [ "Gas" , "DM" , "-", "Tracer", "Star", "BH" ]

