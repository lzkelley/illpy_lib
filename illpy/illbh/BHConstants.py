
import os
import numpy as np

__all__ = [ 'MERGERS_TIMES', 'MERGERS_IDS', 'MERGERS_MASSES', 'MERGERS_DIR', 'MERGERS_RUN', 'MERGERS_NUM', \
            'MERGERS_CREATED', 'MERGERS_MAP_STOM', 'MERGERS_MAP_MTOS', 'MERGERS_MAP_ONTOP', \
            'IN_BH', 'OUT_BH', 'DATA_PATH', '_DOUBLE', '_LONG', \
            'DETAIL_IDS', 'DETAIL_TIMES', 'DETAIL_MASSES', 'DETAIL_MDOTS', 'DETAIL_RHOS', 'DETAIL_CS', \
            'DETAIL_RUN', 'DETAIL_SNAP', 'DETAIL_NUM', 'DETAIL_CREATED', 'DETAIL_BEFORE', 'DETAIL_AFTER', \
            'DETAIL_PHYSICAL_KEYS' ]

DATA_PATH = "%s/data/" % os.path.dirname(os.path.abspath(__file__))

_DOUBLE = np.float64
_LONG = np.int64


# Key Names for Mergers Dictionary
MERGERS_TIMES     = 'times'
MERGERS_IDS       = 'ids'
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
DETAIL_IDS     = 'id'
DETAIL_TIMES   = 'times'
DETAIL_MASSES  = 'masses'
DETAIL_MDOTS   = 'mdots'
DETAIL_RHOS    = 'rhos'
DETAIL_CS      = 'cs'
DETAIL_RUN     = 'run'
DETAIL_SNAP    = 'snap'
DETAIL_NUM     = 'num'
DETAIL_CREATED = 'created'

DETAIL_BEFORE  = 0                                                                                  # Before merger time (MUST = 0!)
DETAIL_AFTER   = 1                                                                                  # After (or equal) merger time (MUST = 1!)

DETAIL_PHYSICAL_KEYS = [ DETAIL_IDS, DETAIL_TIMES, DETAIL_MASSES, DETAIL_MDOTS, DETAIL_RHOS, DETAIL_CS ]
