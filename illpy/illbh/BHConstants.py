
import os
import numpy as np
from glob import glob


VERBOSE = True
VERSION = 0.2


### Illustris Parameters ###
_ILLUSTRIS_RUN_NAMES  = { 1 : "L75n1820FP",
                          2 : "L75n910FP",
                          3 : "L75n455FP" }

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
                            1 : "" }


NUM_SNAPS = 135


### Post-Processing Parameters ###
_PROCESSED_DIR = "/n/home00/lkelley/illustris/data/%s/output/postprocessing/"
_PROCESSED_MERGERS_DIR = _PROCESSED_DIR + "blackhole_mergers/"
_PROCESSED_DETAILS_DIR = _PROCESSED_DIR + "blackhole_details/"

_MERGERS_RAW_COMBINED_FILENAME  = "blackhole_mergers_combined.txt"
_MERGERS_RAW_MAPPED_FILENAME = "blackhole_mergers_mapped.npz"
_MERGERS_FIXED_FILENAME     = "blackhole_mergers_fixed.npz"


INT = np.int32
FLT = np.float32
DBL = np.float64
LNG = np.int64



NUM_BH_TYPES = 2                                                                                    # There are 2 BHs, {BH_IN, BH_OUT}
NUM_BH_TIMES = 3                                                                                    # There are 3 times, {DETAIL_BEFORE, DETAIL_AFTER, DETAIL_F


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
TREE_CREATED      = 'created'
TREE_RUN          = 'run'
TREE_NUM_FUTURE   = 'numFuture'
TREE_NUM_PAST     = 'numPast'
TREE_TIME_BETWEEN = 'timeBetween'
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


def GET_MERGERS_RAW_COMBINED_FILENAME(run):
    fname = _PROCESSED_MERGERS_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _MERGERS_RAW_COMBINED_FILENAME
    return fname


def GET_MERGERS_RAW_MAPPED_FILENAME(run):
    fname = _PROCESSED_MERGERS_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _MERGERS_RAW_MAPPED_FILENAME
    return fname


GET_PROCESSED_DIR = lambda run: _PROCESSED_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
GET_MERGERS_PROCESSED_DIR = lambda run: _PROCESSED_MERGERS_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
GET_DETAILS_PROCESSED_DIR = lambda run: _PROCESSED_DETAILS_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))

