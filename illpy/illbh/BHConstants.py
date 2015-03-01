
import os
import numpy as np

__all__ = [ 'MERGERS_TIMES', 'MERGERS_IDS', 'MERGERS_MASSES', 'MERGERS_DIR', 'MERGERS_RUN', 'MERGERS_NUM', \
            'MERGERS_CREATED', 'MERGERS_MAP_STOM', 'MERGERS_MAP_MTOS', 'MERGERS_MAP_ONTOP', \
            'IN_BH', 'OUT_BH', 'DATA_PATH', '_DOUBLE', '_LONG', \
            'DETAILS_IDS', 'DETAILS_TIMES', 'DETAILS_MASSES', 'DETAILS_MDOTS', 'DETAILS_RHOS', 'DETAILS_CS', \
            'DETAILS_RUN', 'DETAILS_SNAP', 'DETAILS_NUM', 'DETAILS_CREATED', 'DETAILS_BEFORE', 'DETAILS_AFTER', \
            'DETAILS_PHYSICAL_KEYS' ]

DATA_PATH = "%s/data/" % os.path.dirname(os.path.abspath(__file__))

_DOUBLE = np.float64
_LONG = np.int64


# Key Names for Mergers Dictionary
MERGERS_IDS       = 'ids'
MERGERS_TIMES     = 'times'
MERGERS_MASSES    = 'masses'
MERGERS_DIR       = 'dir'
MERGERS_RUN       = 'run'
MERGERS_CREATED   = 'created'
MERGERS_NUM       = 'num'
MERGERS_MAP_STOM  = 's2m'
MERGERS_MAP_MTOS  = 'm2s'
MERGERS_MAP_ONTOP = 'ontop'

# Index of [N,2] arrays corresponding to each BH
IN_BH  = 0
OUT_BH = 1


### Dictionary Keys for Details Parameters ###
DETAILS_IDS     = 'id'
DETAILS_TIMES   = 'times'
DETAILS_MASSES  = 'masses'
DETAILS_MDOTS   = 'mdots'
DETAILS_RHOS    = 'rhos'
DETAILS_CS      = 'cs'
DETAILS_RUN     = 'run'
DETAILS_SNAP    = 'snap'
DETAILS_NUM     = 'num'
DETAILS_CREATED = 'created'

DETAILS_BEFORE  = 0                                                                                  # Before merger time (MUST = 0!)
DETAILS_AFTER   = 1                                                                                  # After (or equal) merger time (MUST = 1!)

DETAILS_PHYSICAL_KEYS = [ DETAILS_IDS, DETAILS_TIMES, DETAILS_MASSES, DETAILS_MDOTS, DETAILS_RHOS, DETAILS_CS ]
