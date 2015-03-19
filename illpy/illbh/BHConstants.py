
import os
import numpy as np

__all__ = [ 'MERGERS_TIMES', 'MERGERS_IDS', 'MERGERS_MASSES', 'MERGERS_DIR', 'MERGERS_RUN', 'MERGERS_NUM', \
                'MERGERS_CREATED', 'MERGERS_MAP_STOM', 'MERGERS_MAP_MTOS', 'MERGERS_MAP_ONTOP', \
                'IN_BH', 'OUT_BH', 'DATA_PATH', '_DOUBLE', '_LONG', \
                'NUM_BH_TYPES', 'NUM_BH_TIMES', \
                'DETAILS_IDS', 'DETAILS_TIMES', 'DETAILS_MASSES', 'DETAILS_MDOTS', 'DETAILS_RHOS', 'DETAILS_CS', \
                'DETAILS_RUN', 'DETAILS_SNAP', 'DETAILS_NUM', 'DETAILS_CREATED', 'DETAILS_BEFORE', 'DETAILS_AFTER', \
                'DETAILS_FIRST', 'DETAILS_PHYSICAL_KEYS', \
                'TREE_LAST', 'TREE_NEXT', 'TREE_LAST_TIME', 'TREE_NEXT_TIME', 'TREE_CREATED', 'TREE_RUN', \
                'TREE_NUM_FUTURE', 'TREE_NUM_PAST', 'TREE_TIME_BETWEEN'
            ]

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

NUM_BH_TYPES = 2                                                                                    # There are 2 BHs, {IN_BH, OUT_BH}
NUM_BH_TIMES = 3                                                                                    # There are 3 times, {DETAIL_BEFORE, DETAIL_AFTER, DETAIL_FIRST}


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

# Types of matches between mergers and details
DETAILS_BEFORE  = 0                                                                                 # Before merger time (MUST = 0!)
DETAILS_AFTER   = 1                                                                                 # After (or equal) merger time (MUST = 1!)
DETAILS_FIRST   = 2                                                                                 # First matching details entry (MUST = 2!)

DETAILS_PHYSICAL_KEYS = [ DETAILS_IDS, DETAILS_TIMES, DETAILS_MASSES,
                          DETAILS_MDOTS, DETAILS_RHOS, DETAILS_CS ]


assert DETAILS_BEFORE == 0 and DETAILS_AFTER == 1 and DETAILS_FIRST == 2, \
    "``DETAILS_<BEFORE/AFTER/FIRST>`` MUST be in the proper order!"



### BH Merger Tree ###

TREE_LAST         = 'last'
TREE_NEXT         = 'next'
TREE_LAST_TIME    = 'lastTime'
TREE_NEXT_TIME    = 'nextTime'
TREE_CREATED      = 'created'
TREE_RUN          = 'run'
TREE_NUM_FUTURE   = 'numFuture'
TREE_NUM_PAST     = 'numPast'
TREE_TIME_BETWEEN = 'timeBetween'

