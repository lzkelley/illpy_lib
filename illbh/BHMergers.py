

import os, sys
from glob import glob
from datetime import datetime

import numpy as np

__all__ = [ 'MERGERS_TIMES', 'MERGERS_IDS', 'MERGERS_MASSES', 'MERGERS_DIR', 
            'MERGERS_RUN', 'MERGERS_CREATED', 'MERGERS_NUM', 
            'loadMergers', 'main' ]


# Default Runtime Parameters
RUN = 3
VERBOSE = True

# Internal Parameters
_VERSION = 0.1

_MERGERS_FILE_DIRS = { 3:'/n/ghernquist/Illustris/Runs/L75n455FP/output/blackhole_mergers/' }
_MERGERS_FILE_NAMES = "blackhole_mergers_*.txt"

_POST_PROCESS_PATH = "%s/post-process/" % os.path.dirname(os.path.abspath(__file__))
_MERGERS_SAVE_FILENAME = "ill-%d_mergers_v%.1f.npz"

_PRINT_INTERVAL = 200

# Key Names for Mergers Dictionary
MERGERS_TIMES = 'times'
MERGERS_IDS = 'ids'
MERGERS_MASSES = 'masses'
MERGERS_DIR = 'dir'
MERGERS_RUN = 'run'
MERGERS_CREATED = 'created'
MERGERS_NUM = 'num'

# Data Types
_DOUBLE = np.float64
_LONG = np.int64




def main(run=RUN, verbose=VERBOSE):
    """

    """

    if( verbose ): print "\nBHMergers.py\n"
    if( verbose ): print " - Run '%d'" % (run)

    if( verbose ): print " - Importing Mergers"
    mergers = __importMergers(run, verbose)
    if( verbose ): print " - Saving Mergers"
    __saveMergers(mergers, run, verbose)

    return



def loadMergers(run=RUN, verbose=VERBOSE):
    
    if( verbose ): print " - Loading mergers from save file"
    mergers = __loadMergersFromSave(run, verbose)

    if( mergers == None ): 
        if( verbose ): print " - Loading mergers directory from Illustris"
        mergers = __importMergers(run, verbose)
        if( verbose ): print " - Saving Mergers"
        __saveMergers(mergers, run, verbose)


    return mergers



def __loadMergersFromSave(run=RUN, verbose=VERBOSE):

    savefile = __getMergersSaveFilename(run)
    # Return None if no save-file exists
    if( not os.path.exists(savefile) ): 
        if( verbose ): print " - - No savefile '%s' exists!" % (savefile)
        return None

    # Load mergers and basic properties
    mergers = np.load(savefile)
    nums    = mergers[MERGERS_NUM]
    mergRun = mergers[MERGERS_RUN]
    created = mergers[MERGERS_CREATED]

    if( verbose ): 
        print " - - Loaded %d Mergers from '%s'" % (nums, savefile)
        print " - - - Run %d, saved at '%s'" % (mergRun, created)
    
    return mergers



def __importMergers(run=RUN, verbose=VERBOSE):

    ### Get Illustris Merger Filenames ###
    if( verbose ): print " - - Searching for merger Files"
    mergerDir = _MERGERS_FILE_DIRS[run]
    mergerFilenames = mergerDir + _MERGERS_FILE_NAMES
    mergerFiles = sorted(glob(mergerFilenames))
    numFiles = len(mergerFiles)
    if( verbose ): print " - - - Found %d merger Files" % (numFiles)


    ### Count Mergers and Prepare Storage for Data ###
    numLines = __countLines(mergerFiles)
    times = np.zeros(numLines, dtype=_DOUBLE)
    ids = np.zeros([numLines,2], dtype=_LONG)
    masses = np.zeros([numLines,2], dtype=_DOUBLE)


    ### Load Raw Data from Merger Files ###
    if( verbose ): print " - - Loading Merger Data"
    __loadMergersFromIllustris(times, ids, masses, mergerFiles, verbose=verbose)
    if( verbose ): print " - - - Loaded %d entries" % (numLines)


    ### Sort Data by Time ###
    if( verbose ): print " - - Sorting Data"
    # Find indices which sort by time
    inds = np.argsort(times)
    # Use indices to reorder arrays
    times[:] = times[inds]
    ids[:] = ids[inds]
    masses[:] = masses[inds]


    ### Store Sorted Data to Dictionary and Save ###
    if( verbose ): print " - - Storing Data to Dictionary"
    # Store to dictionary
    mergers = { MERGERS_DIR     : mergerDir,
                MERGERS_RUN     : run,
                MERGERS_NUM     : numLines,
                MERGERS_TIMES   : times,
                MERGERS_IDS     : ids,
                MERGERS_MASSES  : masses,
                MERGERS_CREATED : datetime.now().ctime()
                }

    return mergers



def __saveMergers(mergers, run=RUN, verbose=VERBOSE):
    savefile = __getMergersSaveFilename(run)
    np.savez(savefile, **mergers)
    if( verbose ): print " - - Saved Mergers dictionary to '%s'" % (savefile)
    return




def __getMergersSaveFilename(run=RUN):
    savefile = _POST_PROCESS_PATH
    if( savefile[-1] != '/' ): savefile += '/'
    if( not os.path.isdir(savefile) ): os.makedirs(savefile)
    savefile += _MERGERS_SAVE_FILENAME % (run, _VERSION)
    return savefile



def __loadMergersFromIllustris(times, ids, masses, files, verbose=VERBOSE):
    nums = len(times)

    count = 0
    for fil in files:
        for line in open(fil):
            # Get target elements from each line of file
            time, id0, mass0, id1, mass1 = __parseMergerLine(line)
            # Store values
            times[count] = time
            ids[count,0] = id0
            ids[count,1] = id1
            masses[count,0] = mass0
            masses[count,1] = mass1
            # Increment Counter
            count += 1

            # Print Progress
            if( verbose ):
                if( count % _PRINT_INTERVAL == 0 or count == nums):
                    sys.stdout.write('\r - - %.2f%% Complete' % (100.0*count/nums))

                if( count == nums ):
                    sys.stdout.write('\n')

                sys.stdout.flush()

    return



def __parseMergerLine(line):
    strs = line.split()
    return _DOUBLE(strs[1]), _LONG(strs[2]), _DOUBLE(strs[3]), _LONG(strs[2]), _DOUBLE(strs[3])


def __countLines(files):
    nums = 0
    # Iterate over each file
    for fil in files:
        # Count number of lines
        nums += sum(1 for line in open(fil))

    return nums





if __name__ == "__main__": main()

