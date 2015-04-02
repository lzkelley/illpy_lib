"""
Provide common constants and hard-settings.

"""

import numpy as np

TYPE_ID     = np.uint64
TYPE_SCALAR = np.float64


### General Illustris ###

NUM_SNAPS = 136

ILLUSTRIS_RUN_NAMES           = { 1: 'L75n1820FP',
                                  2: 'L75n910FP' ,
                                  3: 'L75n455FP'  }

ILLUSTRIS_OUTPUT_PATH_BASE    = '/n/hernquistfs1/Illustris/Runs/%s/output/'
ILLUSTRIS_OUTPUT_PATHS        = lambda xx: ILLUSTRIS_OUTPUT_PATH_BASE % (ILLUSTRIS_RUN_NAMES[xx])



### SubLink Merger Trees ###

ILLUSTRIS_TREE_PATH_BASE      = '/n/ghernquist/Illustris/Runs/%s/trees/SubLink_gal'
ILLUSTRIS_TREE_PATHS          = lambda xx: ILLUSTRIS_TREE_PATH_BASE % (ILLUSTRIS_RUN_NAMES[xx])

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
