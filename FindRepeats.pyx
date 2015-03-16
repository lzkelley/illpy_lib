"""


"""


import numpy as np
cimport numpy as np

import illpy
from illpy.illbh.BHConstants import IN_BH, OUT_BH

def findRepeats(np.ndarray[long,   ndim=2] ids,   np.ndarray[double, ndim=1] times,
                np.ndarray[long,   ndim=2] last,  np.ndarray[long,   ndim=1] next,  
                np.ndarray[double, ndim=2] inter ):
    """

    Parameters
    ----------
    ids : <long>[N,2]
          array of merger BH indices (for both 'in' and 'out' BHs)

    times : <double>[N]
            array of merger times -- !! in units of age of the universe !!

    Notes
    -----
    - This does not account for any mergers for which *both* entering BHs were
      previous mergers.

    """

    cdef long outid, ss, ii, jj, tt
    cdef long numMergers = ids.shape[0]
    cdef int hunth = np.int(np.floor(0.01*numMergers))

    # Get indices to sort mergers by time
    cdef np.ndarray sort_t = np.argsort(times)

    ### Iterate Over Each Merger, In Order of Merger Time ###

    for ii in xrange(numMergers):

        if( ii > 0 and ii%hunth == 0 ): print "%5d/%d" % (ii, numMergers)

        # Conver to sorted merger index
        mind = sort_t[ii]

        # Get the output ID from this merger
        outid = ids[mind, OUT_BH]


        ## Iterate over all Later Mergers ##
        #  use a while loop so we can break out of it
        jj = ii+1
        while( jj < numMergers ):
            # Convert to sorted index
            tt = sort_t[jj]

            # If previous merger goes into this one; save relationships
            for BH in [IN_BH, OUT_BH]:

                if( ids[tt,BH] == outid ):
                    # Set index of next merger
                    next[ss] = tt
                    # Set index of last merger
                    last[tt,BH] = ss
                    # Set time between mergers
                    inter[ss,BH] = times[tt] - times[ss]

                    # Break back to highest for-loop over all mergers (ii)
                    jj = numMergers
                    break

            # Increment
            jj += 1

        # } jj

    # } ii

    return 
    #return last, next, inter






