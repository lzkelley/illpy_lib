"""
Provide common constants and hard-settings.

"""

import numpy as np

import illpy
from illpy.Constants import *
from illpy.illbh.BHConstants import *

BASE_PATH  = "/n/home00/lkelley/illustris/EplusA/Subhalos/"

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



### General Illustris ###

NUM_SNAPS = 136

ILLUSTRIS_RUN_NAMES           = { 1: 'L75n1820FP',
                                  2: 'L75n910FP' ,
                                  3: 'L75n455FP'  }




### SubLink Merger Trees ###

ILLUSTRIS_TREE_PATH_BASE      = '/n/ghernquist/Illustris/Runs/%s/trees/SubLink_gal'
ILLUSTRIS_TREE_PATHS          = lambda run: ILLUSTRIS_TREE_PATH_BASE % (ILLUSTRIS_RUN_NAMES[run])

SL_SNAP_NUM                   = "SnapNum"
SL_SUBFIND_ID                 = "SubfindID"

SUBLINK_PARAMETERS            = [ SL_SNAP_NUM, SL_SUBFIND_ID ]
SUBLINK_PARAMETER_TYPES       = [ TYPE_ID,     TYPE_ID       ]

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

### Additional Data for Branches ###
BRANCH_RUN                    = "run"
BRANCH_INDS                   = "indices"
BRANCH_CREATED                = "created"
BRANCH_SNAPS                  = "snapshots"



### EplusA Snapshot Particle Data ###



### Subhalo Profiles ###

PROFILE_BIN_EDGES = 'bin_edges'
PROFILE_BIN_AVES  = 'bin_aves'
PROFILE_GAS       = 'gas'
PROFILE_STARS     = 'stars'
PROFILE_DM        = 'dm'
PROFILE_COLS      = 'cols'
PROFILE_CREATED   = 'created'
PROFILE_VERSION   = 'version'



### Intermediate Save Files ###

_SUBHALO_BRANCHES_SAVE_BASE = BASE_PATH + "data/%s/ill-%d_branches.npz"

_SUBHALO_FILENAMES_NUMBERS_SAVE_BASE = BASE_PATH + "data/%s/ill-%d_snap-%d_subhalos-names-numbers.npz"

_SUBHALO_RADIAL_PROFILES_FILENAME_BASE = BASE_PATH + "data/%s/ill-%d_snap-%d_subhalos-profiles.npz"



def SUBHALO_BRANCHES_FILENAMES(run):
    fileName = _SUBHALO_BRANCHES_SAVE_BASE % (ILLUSTRIS_RUN_NAMES[run], run)
    return fileName


def SUBHALO_FILENAMES_NUMBERS_FILENAMES(run, snap):
    fileName = _SUBHALO_FILENAMES_NUMBERS_SAVE_BASE % (ILLUSTRIS_RUN_NAMES[run], run, snap)
    return fileName


def GET_SUBHALO_RADIAL_PROFILES_FILENAMES(run, snap):
    fileName = _SUBHALO_RADIAL_PROFILES_FILENAME_BASE % (ILLUSTRIS_RUN_NAMES[run], run, snap)
    return fileName
