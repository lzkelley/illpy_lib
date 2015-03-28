"""
Provide common constants and hard-settings.

"""


### General Illustris ###

ILLUSTRIS_RUN_NAMES  = { 1: 'L75n1820FP',
                         2: 'L75n910FP' ,
                         3: 'L75n455FP'  }

ILLUSTRIS_OUTPUT_PATH_BASE = '/n/hernquistfs1/Illustris/Runs/%s/output/'
ILLUSTRIS_OUTPUT_PATHS     = lambda xx: ILLUSTRIS_OUTPUT_PATH_BASE % (ILLUSTRIS_RUN_NAMES[xx])



### SubLink Merger Trees ###

ILLUSTRIS_TREE_PATH_BASE   = '/n/ghernquist/Illustris/Runs/%s/trees/SubLink_gal'
ILLUSTRIS_TREE_PATHS       = lambda xx: ILLUSTRIS_TREE_PATH_BASE % (ILLUSTRIS_RUN_NAMES[xx])

SL_SNAP_NUM     = "SnapNum"



### Subfind Catalog Parameters ###

SH_BH_MASS      = "SubhaloBHMass"
SH_HALFMASS_RAD = "SubhaloHalfmassRad"
SH_MASS_TYPE    = "SubhaloMassType"
SH_SFR          = "SubhaloSFR"
SH_PHOTO        = "SubhaloStellarPhotometrics"
SH_VEL_DISP     = "SubhaloVelDisp"
SH_LEN_TYPE     = "SubhaloLenType"

SUBFIND_PARAMETERS = [ SH_BH_MASS, SH_HALFMASS_RAD, SH_MASS_TYPE, SH_SFR, SH_PHOTO, SH_VEL_DISP,
                       SH_LEN_TYPE ]


