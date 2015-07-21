"""
Constants for Blackhole related functions and submodules.

Classes
-------
    MERGERS : enum-type class for BH-Merger dictionary keys.
              The list ``MERGERS_PHYSICAL_KEYS`` contains the keys which pertain to values taken
              from the BH Merger files themselves
    DETAILS : enum-type class for BH-Details entries dictionary keys.
              The list ``DETAILS_PHYSICAL_KEYS`` contains the keys corresponding to values taken
              from the BH Details files themselves
    BH_TYPE : enum-type class for tracking the two types {``IN``,``OUT``} of Merger BHs.
              The ``OUT`` BH is the one which persists after the merger, while the ``IN`` BH
              effectively dissappears.
    BH_TIME : enum-type class for the three stored, details times {``FIRST``,``BEFORE``,``AFTER``}.
    BH_TREE : enum-type class for BH merger tree dictionary keys.


Functions
---------



"""

import numpy as np
from enum import Enum
from glob import glob

from .. Constants import NUM_SNAPS, GET_ILLUSTRIS_RUN_NAMES, _PROCESSED_DIR, GET_PROCESSED_DIR


## Illustris Parameters
#  ====================

_ILLUSTRIS_MERGERS_FILENAME_REGEX = "blackhole_mergers_*.txt"
_ILLUSTRIS_DETAILS_FILENAME_REGEX = "blackhole_details_*.txt"

_ILLUSTRIS_MERGERS_DIRS = { 3 : "/n/ghernquist/Illustris/Runs/L75n455FP/output/blackhole_mergers/",
                            2 : "/n/ghernquist/Illustris/Runs/L75n910FP/combined_output/blackhole_mergers/",
                            1 : ["/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-curie/blackhole_mergers/",
                                 "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-supermuc/blackhole_mergers/",
                                 "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Aug8/blackhole_mergers/",
                                 "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Aug14/blackhole_mergers/",
                                 "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Sep25/blackhole_mergers/",
                                 "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Oct10/blackhole_mergers/" ]
                            }

_ILLUSTRIS_DETAILS_DIRS = { 3 : "/n/ghernquist/Illustris/Runs/L75n455FP/output/blackhole_details/",
                            2 : "/n/ghernquist/Illustris/Runs/L75n910FP/combined_output/blackhole_details/",
                            1 : ["/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-curie/blackhole_details/",
                                 "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-supermuc/blackhole_details/",
                                 "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Aug8/blackhole_details/",
                                 "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Aug14/blackhole_details/",
                                 "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Oct10/blackhole_details/",
                                 "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Sep25/blackhole_details/" ]
                            }


## Post-Processing Parameters
#  ==========================
_PROCESSED_MERGERS_DIR          = _PROCESSED_DIR + "blackhole_mergers/"
_PROCESSED_DETAILS_DIR          = _PROCESSED_DIR + "blackhole_details/"

_MERGERS_RAW_COMBINED_FILENAME  = "ill-%d_blackhole_mergers_combined.txt"
_MERGERS_RAW_MAPPED_FILENAME    = "ill-%d_blackhole_mergers_mapped_v%.2f.npz"
_MERGERS_FIXED_FILENAME         = "ill-%d_blackhole_mergers_fixed_v%.2f.npz"

_DETAILS_TEMP_FILENAME          = "ill-%d_blackhole_details_temp_snap-%d.txt"
_DETAILS_SAVE_FILENAME          = "ill-%d_blackhole_details_save_snap-%d_v%.2f.npz"

_MERGER_DETAILS_FILENAME        = 'ill-%d_blackhole_merger-details_v%.2f.npz'

_BLACKHOLE_TREE_FILENAME        = "ill-%d_bh-tree_v%.2f.npz"



class MERGERS(Enum):
    # Meta Data
    RUN       = 'run'
    CREATED   = 'created'
    NUM       = 'num'
    VERSION   = 'version'
    FILE      = 'filename'

    # Physical Parameters
    IDS       = 'ids'
    SCALES    = 'scales'
    MASSES    = 'masses'

    # Maps
    MAP_STOM  = 's2m'
    MAP_MTOS  = 'm2s'
    MAP_ONTOP = 'ontop'

# } MERGERS

MERGERS_PHYSICAL_KEYS = [ MERGERS.IDS, MERGERS.SCALES, MERGERS.MASSES ]


class DETAILS(Enum):
    RUN     = 'run'
    CREATED = 'created'
    VERSION = 'version'
    NUM     = 'num'
    SNAP    = 'snap'
    FILE    = 'filename'

    IDS     = 'id'
    SCALES  = 'scales'
    MASSES  = 'masses'
    MDOTS   = 'mdots'
    RHOS    = 'rhos'
    CS      = 'cs'

# } DETAILS

DETAILS_PHYSICAL_KEYS = [ DETAILS.IDS, DETAILS.SCALES, DETAILS.MASSES,  
                          DETAILS.MDOTS, DETAILS.RHOS, DETAILS.CS ]


class BH_TYPE(Enum):
    IN  = 0
    OUT = 1


class BH_TIME(Enum):
    BEFORE  = 0                                   # Before merger time (MUST = 0!)
    AFTER   = 1                                   # After (or equal) merger time (MUST = 1!)
    FIRST   = 2                                   # First matching details entry (MUST = 2!)


class BH_TREE(Enum):
    LAST         = 'last'
    NEXT         = 'next'
    LAST_TIME    = 'lastTime'
    NEXT_TIME    = 'nextTime'
    NUM_FUTURE   = 'numFuture'
    NUM_PAST     = 'numPast'
    TIME_BETWEEN = 'timeBetween'

    CREATED      = 'created'
    RUN          = 'run'
    VERSION      = 'version'




def GET_ILLUSTRIS_BH_MERGERS_FILENAMES(run, verbose=True):
    if( verbose ): print " - - BHConstants.GET_ILLUSTRIS_BH_MERGERS_FILENAMES()"
    filesDir = _ILLUSTRIS_MERGERS_DIRS[run]
    files = []
    if( type(filesDir) != list ): filesDir = [ filesDir ]

    for fdir in filesDir:
        filesNames = fdir + _ILLUSTRIS_MERGERS_FILENAME_REGEX
        someFiles = sorted( glob(filesNames) )
        if( verbose ): print " - - - '%s' : %d files" % (fdir, len(someFiles))
        files += someFiles

    if( verbose ): print " - - - %d Total Files" % (len(files))

    return files


def GET_ILLUSTRIS_BH_DETAILS_FILENAMES(run, verbose=True):

    if( verbose ): print " - - BHConstants.GET_ILLUSTRIS_BH_DETAILS_FILENAMES()"

    filesDir = _ILLUSTRIS_DETAILS_DIRS[run]
    files = []
    if( type(filesDir) != list ): filesDir = [ filesDir ]

    for fdir in filesDir:
        filesNames = fdir + _ILLUSTRIS_DETAILS_FILENAME_REGEX
        someFiles = sorted( glob(filesNames) )
        if( verbose ): print " - - - '%s' : %d files" % (fdir, len(someFiles))
        files += someFiles

    if( verbose ): print " - - - %d Total Files" % (len(files))

    return files



def GET_MERGERS_RAW_COMBINED_FILENAME(run):
    fname = _PROCESSED_MERGERS_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _MERGERS_RAW_COMBINED_FILENAME % (run)
    return fname


def GET_MERGERS_RAW_MAPPED_FILENAME(run, version):
    fname = _PROCESSED_MERGERS_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _MERGERS_RAW_MAPPED_FILENAME % (run, version)
    return fname


def GET_MERGERS_FIXED_FILENAME(run, version):
    fname = _PROCESSED_MERGERS_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _MERGERS_FIXED_FILENAME % (run, version)
    return fname


def GET_DETAILS_TEMP_FILENAME(run, snap):
    fname = _PROCESSED_DETAILS_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _DETAILS_TEMP_FILENAME % (run, snap)
    return fname


def GET_DETAILS_SAVE_FILENAME(run, snap, version):
    fname = _PROCESSED_DETAILS_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _DETAILS_SAVE_FILENAME % (run, snap, version)
    return fname


def GET_MERGER_DETAILS_FILENAME(run, version):
    fname = _PROCESSED_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _MERGER_DETAILS_FILENAME % (run, version)
    return fname


def GET_BLACKHOLE_TREE_FILENAME(run, version):
    fname = _PROCESSED_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _BLACKHOLE_TREE_FILENAME % (run, version)
    return fname


assert BH_TYPE.IN.value == 0 and BH_TYPE.OUT.value == 1, \
    "``BH_TYPE.{IN/OUT}`` MUST be in the proper order!"


assert BH_TIME.BEFORE.value == 0 and BH_TIME.AFTER.value == 1 and BH_TIME.FIRST.value == 2, \
    "``BH_TIME.{BEFORE/AFTER/FIRST}`` MUST be in the proper order!"

