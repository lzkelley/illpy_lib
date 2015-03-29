
import numpy as np
from datetime import datetime
import os

from .. import AuxFuncs as aux
from .. Constants import *
from .. import Cosmology
from BHConstants import *
import BHMergers

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
import BuildTree

RUN      = 3                                                                                        # Which illustris simulation to target
VERBOSE  = True
_VERSION = 0.1
LOAD     = False

MYR = (1.0e6)*YEAR

_TREE_SAVE_FILENAME = "ill-%d_bh-tree_v%.1f.npz"
bhTree_filename = lambda xx : DATA_PATH + (_TREE_SAVE_FILENAME % (xx, _VERSION))



###  ==============================================================  ###
###  ===========================  MAIN  ===========================  ###
###  ==============================================================  ###



def main(run=RUN, load=LOAD, verbose=VERBOSE):

    print " - BHTree.py"

    start_time  = datetime.now()

    ### Load Repeated Mergers ###
    tree = getTree(run, load=False, verbose=verbose)

    end_time    = datetime.now()
    durat       = end_time - start_time

    print " - - Done after %s\n\n" % (str(durat))

    return

# main()



def getTree(run, mergers=None, load=False, verbose=VERBOSE):
    """
    Load tree data from save file if possible, or recalculate directly.

    Arguments
    ---------
    run : <int>
        Illlustris run number {1,3}
    mergers : <dict>, (optional=None), BHMergers.py mergers
        Contains merger information.  If `None`, reloaded from BHMergers
    load : <bool>, (optional=False)
        Reload tree data directly from merger data
    verbose : <bool>, (optional=VERBOSE)
        Print verbose output

    Returns
    -------
    tree : <dict>
        container for tree data - see BHTree doc

    """

    if( verbose ): print " - - BHTree.getTree()"

    fname = bhTree_filename(run)

    recalc_flag = False
    if( load ):
        recalc_flag = load

    # If save file doesn't exist, must recalculate
    if( not os.path.exists(fname) ):
        if( verbose ): print " - - - No existing file '%s'" % ( fname )
        recalc_flag = True
    

    ### Recalculate Tree Data from Mergers ###
    if( recalc_flag ):

        if( mergers is None ):
            print " - - - No mergers provided, loading"
            mergers = BHMergers.loadMergers(run)

        print " - - - Finding Merger Tree from Merger Data"

        start = datetime.now()
        ### Build Tree ###
        tree = constructBHTree(run, mergers)
    
        ### Analyze Tree Data ###
        timeBetween, numPast, numFuture = analyzeTree(tree, verbose=verbose)

        # Save Tree data
        aux.saveDictNPZ(tree, fname, verbose=True)
        stop = datetime.now()
        if( verbose ): print " - - - - Done after %s" % (str(stop-start))

    ### Load Tree Data from Existing Save ###
    else:

        if( verbose ) : print " - - - Loading Merger Tree from '%s'" % (fname)
        start = datetime.now()
        dat   = np.load(fname)
        tree  = aux.npzToDict(dat)
        stop  = datetime.now()
        if( verbose ) : print " - - - - Loaded after %s" % (str(stop-start))


    return tree





def constructBHTree(run, mergers, verbose=VERBOSE):
    """
    Use merger data to find and connect BHs which merge multiple times.

    Arguments
    ---------
    run : <int>
        Illlustris run number {1,3}
    mergers : <dict>, BHMergers.py mergers
        Contains merger information
    verbose : <bool>, (optional=VERBOSE)
        Print verbose output

    Returns
    -------
    tree : <dict>
        container for tree data - see BHTree doc

    """

    if( verbose ): print " - - BHTree.constructBHTree()"

    cosmo = Cosmology()

    numMergers = mergers[MERGERS_NUM]
    last     = -1*  np.ones([numMergers,NUM_BH_TYPES], dtype=long)
    next     = -1*  np.ones([numMergers],              dtype=long)
    lastTime = -1.0*np.ones([numMergers,NUM_BH_TYPES], dtype=np.float64)
    nextTime = -1.0*np.ones([numMergers],              dtype=np.float64)

    # Convert merger scale factors to ages
    if( verbose ): print " - - - Converting merger times"
    start = datetime.now()
    scales = mergers[MERGERS_TIMES]
    times = np.array([ cosmo.age(sc) for sc in scales ], dtype=np.float64)
    stop = datetime.now()
    if( verbose ): print " - - - - Done after %s" % (str(stop-start))


    # Construct Merger Tree
    if( verbose ): print " - - - Builder BH Merger Tree"
    start = datetime.now()
    mids = mergers[MERGERS_IDS]
    BuildTree.buildTree(mids, times, last, next, lastTime, nextTime)
    stop = datetime.now()
    if( verbose ): print " - - - - Retrieved after %s" % (str(stop-start))

    inds = np.where( last < 0 )[0]
    print "MISSING LAST = ", len(inds)

    inds = np.where( next < 0 )[0]
    print "MISSING NEXT = ", len(inds)


    # Create dictionary to store data
    tree = { TREE_LAST      : last,
             TREE_NEXT      : next,
             TREE_LAST_TIME : lastTime,
             TREE_NEXT_TIME : nextTime,
             TREE_CREATED   : datetime.now().ctime(),
             TREE_RUN       : run }

    return tree




def analyzeTree(tree, verbose=VERBOSE):
    """
    Analyze the merger tree data to obtain typical number of repeats, etc.

    Arguments
    ---------
    tree : <dict>
        container for tree data - see BHTree doc
    verbose : <bool>, (optional=VERBOSE)
        Print verbose output

    Returns
    -------


    """

    if( verbose ): print " - - BHTree.analyzeTree()"
    
    last         = tree[TREE_LAST]
    next         = tree[TREE_NEXT]
    timeLast     = tree[TREE_LAST_TIME]
    timeNext     = tree[TREE_NEXT_TIME]

    numMergers   = len(next)

    aveFuture    = 0.0
    avePast      = 0.0
    aveFutureNum = 0
    avePastNum   = 0

    numPast      = np.zeros(numMergers, dtype=int  )
    numFuture    = np.zeros(numMergers, dtype=int  )

    if( verbose ): print " - - - %d Mergers" % (numMergers)

    # Find number of unique merger BHs (i.e. no previous mergers)
    inds = np.where( (last[:,IN_BH] < 0) & (last[:,OUT_BH] < 0) & (next[:] < 0) )
    numTwoIsolated = len(inds[0])
    inds = np.where( ((last[:,IN_BH] < 0) ^ (last[:,OUT_BH] < 0)) & (next[:] < 0) )                 # 'xor' comparison
    numOneIsolated = len(inds[0])
    
    if( verbose ): 
        print " - - - Mergers with neither  BH previously merged = %d" % (numTwoIsolated)
        print " - - - Mergers with only one BH previously merged = %d" % (numOneIsolated)


    ### Go back through All Mergers to Count Tree ###
    
    for ii in xrange(numMergers):

        ## Count Forward from First Mergers ##
        # If this is a first merger
        if( all(last[ii,:] < 0) ):

            # Count the number of mergers that the 'out' BH  from this merger, will later be in
            numFuture[ii] = countFutureMergers(next, ii)

            # Accumulate for averaging
            aveFuture += numFuture[ii]
            aveFutureNum += 1

                
        ## Count Backward from Last Mergers ##
        # If this is a final merger
        if( next[ii] < 0 ):

            # Count the number of mergers along the longest branch of past merger tree
            numPast[ii] = countPastMergers(last, ii)

            # Accumulate for averaging
            avePast += numPast[ii]
            avePastNum += 1

    # } ii

    # Calculate averages
    if( avePastNum   > 0 ): avePast   /= avePastNum
    if( aveFutureNum > 0 ): aveFuture /= aveFutureNum

    inds = np.where(next >= 0)[0]
    numRepeats = len(inds)
    fracRepeats = 1.0*numRepeats/numMergers
    if( verbose ): 
        print " - - - Number of repeated mergers = %d/%d = %.4f" % (numRepeats, numMergers, fracRepeats)
        print " - - - Average Number of Repeated mergers  past, future  =  %.3f, %.3f" % (avePast, aveFuture)


    indsInt = np.where( timeNext >= 0.0 )[0]
    timeStats = aux.avestd( timeNext[indsInt] )
    inds = np.where( timeNext == 0.0 )[0]

    if( verbose ): 
        print " - - - Number of merger intervals    = %d" % (len(indsInt))
        print " - - - - Time between = %.4e +- %.4e [Myr]" % (timeStats[0]/MYR, timeStats[1]/MYR)
        print " - - - Number of zero time intervals = %d" % (len(inds))


    timeBetween = timeNext[indsInt]

    tree[TREE_NUM_PAST] = numPast
    tree[TREE_NUM_FUTURE] = numFuture
    tree[TREE_TIME_BETWEEN] = timeBetween

    return timeBetween, numPast, numFuture



def countFutureMergers( next, ind ):
    count = 0
    ii = ind
    while( next[ii] >= 0 ):
        count += 1
        ii = next[ii]

    return count


def countPastMergers( last, ind, verbose=False ):
    
    last_in  = last[ind, IN_BH]
    last_out = last[ind, OUT_BH]
    
    num_in   = 0
    num_out  = 0

    if( last_in >= 0 ):
        num_in = countPastMergers( last, last_in )

    if( last_out >= 0 ):
        num_out = countPastMergers( last, last_out )

    if( verbose ): print "%d   <===   %d (%d)   %d (%d)" % (ind, last_in, num_in, last_out, num_out)

    return np.max([num_in, num_out])+1







if __name__ == "__main__": main()

