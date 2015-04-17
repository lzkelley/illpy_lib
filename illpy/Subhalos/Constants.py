"""
Provide common constants and hard-settings.

"""

import numpy as np

import illpy
#from illpy.Constants import *
#from illpy.illbh.BHConstants import *

TYPE_ID     = np.uint64
TYPE_SCALAR = np.float64


### Subhalo Properties ###
SUBHALO_ID                        = "id"
SUBHALO_RUN                       = "run"
SUBHALO_SNAPSHOT                  = "snapshot"
SUBHALO_CREATED                   = "created"
SUBHALO_VERSION                   = "version"

SNAPSHOT_BH_MASS                  = "BH_Mass"
SNAPSHOT_BH_HSML                  = "BH_Hsml"
SNAPSHOT_BH_MDOT                  = "BH_Mdot"
SNAPSHOT_STELLAR_PHOTOS           = "GFM_StellarPhotometrics"
SNAPSHOT_FORM_TIME                = "GFM_StellarFormationTime"
SNAPSHOT_PARENT                   = "ParentID"
SNAPSHOT_SUBFIND_HSML             = "SubfindHsml"
SNAPSHOT_SUBFIND_VDISP            = "SubfindVelDisp"
SNAPSHOT_FILENAME                 = "filename"
SNAPSHOT_HSML                     = "hsml"
SNAPSHOT_MASS                     = "mass"
SNAPSHOT_MASSES                   = "masses"
SNAPSHOT_NPART                    = "npart_loaded"
SNAPSHOT_POS                      = "pos"
SNAPSHOT_POT                      = "pot"
SNAPSHOT_DENS                     = "rho"
SNAPSHOT_SFR                      = "sfr"
SNAPSHOT_VEL                      = "vel"
SNAPSHOT_EINT                     = "internalenergy"


SNAPSHOT_PROPERTIES = [ SNAPSHOT_BH_MASS, SNAPSHOT_BH_HSML, SNAPSHOT_BH_MDOT, SNAPSHOT_STELLAR_PHOTOS, 
                        SNAPSHOT_FORM_TIME, SNAPSHOT_PARENT, SNAPSHOT_SUBFIND_HSML, SNAPSHOT_SUBFIND_VDISP,
                        SNAPSHOT_FILENAME, SNAPSHOT_HSML, SNAPSHOT_MASS, SNAPSHOT_MASSES, 
                        SNAPSHOT_NPART, SNAPSHOT_POS, SNAPSHOT_POT, SNAPSHOT_DENS, 
                        SNAPSHOT_SFR, SNAPSHOT_VEL, SNAPSHOT_EINT ]

_SUBHALO_PARTICLES_PATH_BASE     = "subhalos/snap_%d/"
_SUBHALO_PARTICLES_FILENAME_BASE = "ill-%d_snap-%d_subhalo-%d.npz"


def GET_SUBHALO_PARTICLES_FILENAMES(run, snap, subhalo): 
    fileName  = GET_PROCESSED_DIR(run) + _SUBHALO_PARTICLES_PATH_BASE % (snap)
    fileName += _SUBHALO_PARTICLES_FILENAME_BASE % (run, snap, subhalo)
    return fileName


### Subfind Catalog Parameters ###

SH_BH_MASS                    = "SubhaloBHMass"
SH_RAD_TYPE                   = "SubhaloHalfmassRadType"
SH_MASS_TYPE                  = "SubhaloMassType"
SH_SFR                        = "SubhaloSFR"
SH_PHOTO                      = "SubhaloStellarPhotometrics"
SH_VEL_DISP                   = "SubhaloVelDisp"
SH_SPIN                       = "SubhaloSpin"
SH_LEN_TYPE                   = "SubhaloLenType"
SH_GROUP_NUM                  = "SubhaloGrNr"
SH_FILENAME                   = "filebase"
SH_PARENT                     = "SubhaloParent"


SUBFIND_PARAMETERS            = [ SH_BH_MASS,  SH_RAD_TYPE, SH_MASS_TYPE, SH_SFR,       SH_PHOTO, 
                                  SH_VEL_DISP, SH_LEN_TYPE, SH_SPIN,      SH_GROUP_NUM, SH_PARENT ]

SUBFIND_PARAMETER_TYPES       = [ TYPE_SCALAR, TYPE_SCALAR, TYPE_SCALAR,  TYPE_SCALAR,  TYPE_SCALAR, 
                                  TYPE_SCALAR, TYPE_ID,     TYPE_SCALAR,  TYPE_ID,      TYPE_ID    ]


SH_SNAPSHOT_NUM               = "snapshot"


### Subhalo Profiles ###

PROFILE_BIN_EDGES = 'bin_edges'
PROFILE_BIN_AVES  = 'bin_aves'
PROFILE_GAS       = 'gas'
PROFILE_STARS     = 'stars'
PROFILE_DM        = 'dm'
PROFILE_COLS      = 'cols'
PROFILE_CREATED   = 'created'
PROFILE_VERSION   = 'version'

