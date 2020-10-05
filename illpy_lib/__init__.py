"""Illustris simulation data manipulation routines and submodules.
"""

from . import constants  # noqa

import numpy as np


class DTYPE:
    ID     = np.uint64
    SCALAR = np.float64
    INDEX  = np.int64
