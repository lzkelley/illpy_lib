
import os
import numpy as np

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
