"""
Provide common constants and hard-settings.

"""

import numpy as np



BASE_PATH  = "/n/home00/lkelley/illustris/EplusA/Subhalos/"

TYPE_ID     = np.uint64
TYPE_SCALAR = np.float64


### General Illustris ###

NUM_SNAPS = 136

ILLUSTRIS_RUN_NAMES           = { 1: 'L75n1820FP',
                                  2: 'L75n910FP' ,
                                  3: 'L75n455FP'  }

ILLUSTRIS_OUTPUT_PATH_BASE    = '/n/hernquistfs1/Illustris/Runs/%s/output/'
ILLUSTRIS_OUTPUT_PATHS        = lambda run: ILLUSTRIS_OUTPUT_PATH_BASE % (ILLUSTRIS_RUN_NAMES[run])

ILLUSTRIS_OUTPUT_GROUP_PATH_BASE   = '/n/ghernquist/Illustris/Runs/%s/output/groups_%d/'
ILLUSTRIS_OUTPUT_GROUP_PATHS       = lambda run,snap: ILLUSTRIS_OUTPUT_GROUP_PATH_BASE % (ILLUSTRIS_RUN_NAMES[run],snap)
ILLUSTRIS_OUTPUT_GROUP_FILENAME_BASE   = 'fof_subhalo_tab_%d.0.hdf5'
ILLUSTRIS_OUTPUT_GROUP_FILENAMES       = lambda snap: ILLUSTRIS_OUTPUT_GROUP_FILENAME_BASE % (snap)
ILLUSTRIS_OUTPUT_GROUP_FIRST_FILENAME  = lambda run,snap: ILLUSTRIS_OUTPUT_GROUP_PATHS(run,snap) + ILLUSTRIS_OUTPUT_GROUP_FILENAMES(snap)

ILLUSTRIS_OUTPUT_SNAPSHOT_PATH_BASE = '/n/ghernquist/Illustris/Runs/%s/output/snapdir_%d/'
ILLUSTRIS_OUTPUT_SNAPSHOT_PATHS       = lambda run,snap: ILLUSTRIS_OUTPUT_SNAPSHOT_PATH_BASE % (ILLUSTRIS_RUN_NAMES[run],snap)
ILLUSTRIS_OUTPUT_SNAPSHOT_FILENAME_BASE   = 'snap_%d.0.hdf5'
ILLUSTRIS_OUTPUT_SNAPSHOT_FILENAMES       = lambda snap: ILLUSTRIS_OUTPUT_SNAPSHOT_FILENAME_BASE % (snap)
ILLUSTRIS_OUTPUT_SNAPSHOT_FIRST_FILENAME  = lambda run,snap: ILLUSTRIS_OUTPUT_SNAPSHOT_PATHS(run,snap) + ILLUSTRIS_OUTPUT_SNAPSHOT_FILENAMES(snap)



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
SUBHALO_ID                        = "id"
SUBHALO_RUN                       = "run"
SUBHALO_SNAPSHOT                  = "snapshot"
SUBHALO_CREATED                   = "created"

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


### Intermediate Save Files ###

_SUBHALO_PARTICLES_PATH_BASE     = BASE_PATH + "data/%s/subhalos/snap_%d/"
_SUBHALO_PARTICLES_FILENAME_BASE = "ill-%d_snap-%d_subhalo-%d.npz"

_SUBHALO_BRANCHES_SAVE_BASE = BASE_PATH + "data/%s/ill-%d_branches.npz"


def SUBHALO_PARTICLES_FILENAMES(run, snap, subhalo): 
    fileName  = _SUBHALO_PARTICLES_PATH_BASE % (ILLUSTRIS_RUN_NAMES[run], snap)
    fileName += _SUBHALO_PARTICLES_FILENAME_BASE % (run, snap, subhalo)
    return fileName


def SUBHALO_BRANCHES_FILENAMES(run):
    fileName = _SUBHALO_BRANCHES_SAVE_BASE % (ILLUSTRIS_RUN_NAMES[run], run)
    return fileName
