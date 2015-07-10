"""
Construct and/or load a Blackhole Merger tree from the illustris blackhole merger data.


"""


import numpy as np
from datetime import datetime
import os

from .. import AuxFuncs as aux
from .. Constants import *
from .. import Cosmology
from BHConstants import *
import BHMergers

import BuildTree


VERSION = 0.21



def loadTree(run, mergers=None, loadsave=True, verbose=True):
    """
    Load tree data from save file if possible, or recalculate directly.

    Arguments
    ---------
    run      : <int>, Illlustris run number {1,3}
    mergers  : <dict>, (optional=None), BHMerger data, reloaded if not provided
    loadsave : <bool>, (optional=True), try to load tree data from previous save
    verbose  : <bool>, (optional=True), Print verbose output

    Returns
    -------
    tree     : <dict>, container for tree data - see BHTree doc

    """

    if( verbose ): print " - - BHTree.loadTree()"

    fname = GET_BLACKHOLE_TREE_FILENAME(run, VERSION)


    ### Reload existing BH Merger Tree ###

    if( loadsave ):
        if( verbose ): print " - - - Loading save file '%s'" % (fname)

        if( os.path.exists(fname) ):
            tree  = aux.npzToDict(fname)
            loadVers = tree[TREE_VERSION]
            if( loadVers != VERSION ):
                loadsave = False
                print "BHTree.loadTree() : loaded  version '%s'" % (str(loadVers))
                print "BHTree.loadTree() : current version '%s'" % (str(VERSION ))
            else:
                if( verbose ): print " - - - Loaded."

        else:
            print "BHTree.loadTree() : save file does not exist!"
            loadsave = False

    

    ### Recreate BH Merger Tree ###

    if( not loadsave ):
        if( verbose ): print " - - - Reconstructing BH Merger Tree"

        # Load Mergers if needed
        if( mergers is None ):
            if( verbose ): print " - - - - No mergers provided, loading"
            mergers = BHMergers.loadFixedMergers(run)


        # Construct Tree 
        if( verbose ): print " - - - - Constructing Tree"
        tree = _constructBHTree(run, mergers, verbose=verbose)
    
        # Analyze Tree Data, store meta-data to tree dictionary
        timeBetween, numPast, numFuture = analyzeTree(tree, verbose=verbose)

        # Save Tree data
        aux.dictToNPZ(tree, fname, verbose=True)


    return tree

# loadTree()



def _constructBHTree(run, mergers, verbose=True):
    """
    Use merger data to find and connect BHs which merge multiple times.

    Arguments
    ---------
    run     : <int>, Illlustris run number {1,3}
    mergers : <dict>, BHMergers dictionary
    verbose : <bool>, (optional=True), Print verbose output

    Returns
    -------
    tree : <dict>
        container for tree data - see BHTree doc

    """

    if( verbose ): print " - - BHTree.constructBHTree()"

    cosmo = Cosmology()

    numMergers = mergers[MERGERS_NUM]
    last     = -1*  np.ones([numMergers,NUM_BH_TYPES], dtype=LNG)
    next     = -1*  np.ones([numMergers],              dtype=LNG)
    lastTime = -1.0*np.ones([numMergers,NUM_BH_TYPES], dtype=DBL)
    nextTime = -1.0*np.ones([numMergers],              dtype=DBL)

    # Convert merger scale factors to ages
    scales = mergers[MERGERS_SCALES]
    times = np.array([ cosmo.age(sc) for sc in scales ], dtype=DBL)

    # Construct Merger Tree from node IDs
    if( verbose ): print " - - - Building BH Merger Tree"
    start = datetime.now()
    mids = mergers[MERGERS_IDS]
    BuildTree.buildTree(mids, times, last, next, lastTime, nextTime)
    stop = datetime.now()
    if( verbose ): print " - - - - Built after %s" % (str(stop-start))

    inds = np.where( last < 0 )[0]
    if( verbose ): print " - - - %d Missing 'last'" % (len(inds))

    inds = np.where( next < 0 )[0]
    if( verbose ): print " - - - %d Missing 'next'" % (len(inds))


    # Create dictionary to store data
    tree = { TREE_LAST      : last,
             TREE_NEXT      : next,
             TREE_LAST_TIME : lastTime,
             TREE_NEXT_TIME : nextTime,

             TREE_CREATED   : datetime.now().ctime(),
             TREE_RUN       : run,
             TREE_VERSION   : VERSION
             }


    return tree

# _constructBHTree()



def analyzeTree(tree, verbose=True):
    """
    Analyze the merger tree data to obtain typical number of repeats, etc.

    Arguments
    ---------
    tree : <dict>
        container for tree data - see BHTree doc
    verbose : <bool>, (optional=True)
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
    inds = np.where( (last[:,BH_IN] < 0) & (last[:,BH_OUT] < 0) & (next[:] < 0) )
    numTwoIsolated = len(inds[0])
    # 
    inds = np.where( ((last[:,BH_IN] < 0) ^ (last[:,BH_OUT] < 0)) & (next[:] < 0) )                 # 'xor' comparison
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

    indsInt = np.where( timeNext >= 0.0 )[0]
    numInts = len(indsInt)
    timeStats = aux.avestd( timeNext[indsInt] )
    inds = np.where( timeNext == 0.0 )[0]
    numZeroInts = len(inds)

    if( verbose ): 
        print " - - - Repeated mergers = %d/%d = %.4f" % (numRepeats, numMergers, fracRepeats)
        print " - - - Average number past, future  =  %.3f, %.3f" % (avePast, aveFuture)
        print " - - - Number of merger intervals    = %d" % (numInts)
        print " - - - - Time between = %.4e +- %.4e [Myr]" % (timeStats[0]/MYR, timeStats[1]/MYR)
        print " - - - Number of zero time intervals = %d" % (numZeroInts)


    timeBetween = timeNext[indsInt]

    # Store data to tree dictionary
    tree[TREE_NUM_PAST] = numPast
    tree[TREE_NUM_FUTURE] = numFuture
    tree[TREE_TIME_BETWEEN] = timeBetween

    return timeBetween, numPast, numFuture

# analyzeTree()



def countFutureMergers( next, ind ):
    count = 0
    ii = ind
    while( next[ii] >= 0 ):
        count += 1
        ii = next[ii]

    return count

# countFutureMergers()



def countPastMergers( last, ind, verbose=False ):
    
    last_in  = last[ind, BH_IN]
    last_out = last[ind, BH_OUT]
    
    num_in   = 0
    num_out  = 0

    if( last_in >= 0 ):
        num_in = countPastMergers( last, last_in )

    if( last_out >= 0 ):
        num_out = countPastMergers( last, last_out )

    if( verbose ): print "%d   <===   %d (%d)   %d (%d)" % (ind, last_in, num_in, last_out, num_out)

    return np.max([num_in, num_out])+1

# countPastMergers()



