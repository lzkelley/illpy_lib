"""
Constants for Blackhole related functions and submodules.


"""


import numpy as np
from glob import glob

from .. Constants import NUM_SNAPS, INT, LNG, FLT, DBL, ULNG, _ILLUSTRIS_RUN_NAMES

VERBOSE = True
#VERSION = 0.22


### Illustris Parameters ###
_ILLUSTRIS_MERGERS_FILENAME_REGEX = "blackhole_mergers_*.txt"
_ILLUSTRIS_DETAILS_FILENAME_REGEX = "blackhole_details_*.txt"

_ILLUSTRIS_MERGERS_DIRS         = { 3 : "/n/ghernquist/Illustris/Runs/L75n455FP/output/blackhole_mergers/",
                                    2 : "/n/ghernquist/Illustris/Runs/L75n910FP/combined_output/blackhole_mergers/",
                                    1 : ["/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-curie/blackhole_mergers/",
                                         "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-supermuc/blackhole_mergers/",
                                         "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Aug8/blackhole_mergers/",
                                         "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Aug14/blackhole_mergers/",
                                         "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Sep25/blackhole_mergers/",
                                         "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Oct10/blackhole_mergers/" ]
                                    }

_ILLUSTRIS_DETAILS_DIRS         = { 3 : "/n/ghernquist/Illustris/Runs/L75n455FP/output/blackhole_details/",
                                    2 : "/n/ghernquist/Illustris/Runs/L75n910FP/combined_output/blackhole_details/",
                                    1 : ["/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-curie/blackhole_details/",
                                         "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-supermuc/blackhole_details/",
                                         "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Aug8/blackhole_details/",
                                         "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Aug14/blackhole_details/",
                                         "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Oct10/blackhole_details/",
                                         "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Sep25/blackhole_details/" ]
                                    }


### Post-Processing Parameters ###
#_PROCESSED_DIR                  = "/n/home00/lkelley/illustris/data/%s/output/postprocessing/"
_PROCESSED_DIR                  = "/n/home00/lkelley/ghernquistfs1/illustris/data/%s/output/postprocessing/"
_PROCESSED_MERGERS_DIR          = _PROCESSED_DIR + "blackhole_mergers/"
_PROCESSED_DETAILS_DIR          = _PROCESSED_DIR + "blackhole_details/"

_MERGERS_RAW_COMBINED_FILENAME  = "ill-%d_blackhole_mergers_combined.txt"
_MERGERS_RAW_MAPPED_FILENAME    = "ill-%d_blackhole_mergers_mapped_v%.2f.npz"
_MERGERS_FIXED_FILENAME         = "ill-%d_blackhole_mergers_fixed_v%.2f.npz"

_DETAILS_TEMP_FILENAME          = "ill-%d_blackhole_details_temp_snap-%d.txt"
_DETAILS_SAVE_FILENAME          = "ill-%d_blackhole_details_save_snap-%d_v%.2f.npz"

_MERGER_DETAILS_FILENAME        = 'ill-%d_blackhole_merger-details_v%.2f.npz'

_BLACKHOLE_TREE_FILENAME        = "ill-%d_bh-tree_v%.2f.npz"





TYPE_ID      = ULNG

NUM_BH_TYPES = 2                                                                                    # There are 2 BHs, {BH_IN, BH_OUT}
NUM_BH_TIMES = 3                                                                                    # There are 3 times, {BH_BEFORE, BH_AFTER, BH_FIRST}


# Key Names for Mergers Dictionary
MERGERS_RUN       = 'run'
MERGERS_CREATED   = 'created'
MERGERS_NUM       = 'num'
MERGERS_VERSION   = 'version'
MERGERS_FILE      = 'filename'

MERGERS_IDS       = 'ids'
MERGERS_SCALES    = 'scales'
MERGERS_MASSES    = 'masses'

MERGERS_MAP_STOM  = 's2m'
MERGERS_MAP_MTOS  = 'm2s'
MERGERS_MAP_ONTOP = 'ontop'

MERGERS_PHYSICAL_KEYS = [ MERGERS_IDS, MERGERS_SCALES, MERGERS_MASSES ]


# Index of [N,2] arrays corresponding to each BH
BH_IN  = 0
BH_OUT = 1

assert BH_IN == 0 and BH_OUT == 1, \
    "``BH_{IN/OUT}`` MUST be in the proper order!"


# Types of matches between mergers and details
BH_BEFORE  = 0                                                                                 # Before merger time (MUST = 0!)
BH_AFTER   = 1                                                                                 # After (or equal) merger time (MUST = 1!)
BH_FIRST   = 2                                                                                 # First matching details entry (MUST = 2!)


assert BH_BEFORE == 0 and BH_AFTER == 1 and BH_FIRST == 2, \
    "``BH_{BEFORE/AFTER/FIRST}`` MUST be in the proper order!"


### Dictionary Keys for Details Parameters ###
DETAILS_RUN     = 'run'
DETAILS_CREATED = 'created'
DETAILS_VERSION = 'version'
DETAILS_NUM     = 'num'
DETAILS_SNAP    = 'snap'
DETAILS_FILE    = 'filename'

DETAILS_IDS     = 'id'
DETAILS_SCALES  = 'scales'
DETAILS_MASSES  = 'masses'
DETAILS_MDOTS   = 'mdots'
DETAILS_RHOS    = 'rhos'
DETAILS_CS      = 'cs'

DETAILS_PHYSICAL_KEYS = [ DETAILS_IDS, DETAILS_SCALES, DETAILS_MASSES,
                          DETAILS_MDOTS, DETAILS_RHOS, DETAILS_CS ]


### BH Merger Tree ###

TREE_LAST         = 'last'
TREE_NEXT         = 'next'
TREE_LAST_TIME    = 'lastTime'
TREE_NEXT_TIME    = 'nextTime'
TREE_NUM_FUTURE   = 'numFuture'
TREE_NUM_PAST     = 'numPast'
TREE_TIME_BETWEEN = 'timeBetween'

TREE_CREATED      = 'created'
TREE_RUN          = 'run'
TREE_VERSION      = 'version'




def GET_ILLUSTRIS_RUN_NAMES(run):
    """ Get canonical name of illustris simulation runs, e.g. 'L75n1820FP' """
    return _ILLUSTRIS_RUN_NAMES[run]


def GET_ILLUSTRIS_BH_MERGERS_FILENAMES(run, verbose=VERBOSE):

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


def GET_ILLUSTRIS_BH_DETAILS_FILENAMES(run, verbose=VERBOSE):

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


def GET_PROCESSED_DIR(run):
    return _PROCESSED_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))


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
