# =================================================================================================
# FindRepeats.pyx
# ---------------
#
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# =================================================================================================




import numpy as np
cimport numpy as np




def findRepeats(np.ndarray[long, ndim=2] ids, np.ndarray[float, ndim=1] times):
    """

    Parameters
    ----------


    Returns
    -------


    Notes
    -----
    - This does not account for any mergers for which *both* entering BHs were
      previous mergers.

    """
    
    cdef long numMergers = ids.shape[0]
    cdef int hunth = np.int(np.floor(numMergers*0.01))

    cdef long out, ss, ii, jj

    # Get indices to sort mergers by time
    cdef np.ndarray sort = np.argsort(times)

    # Initialize arrays to store results; default to '-1'
    cdef np.ndarray next  = -1*np.ones(numMergers, dtype=long)                                      # Map to the next merger
    cdef np.ndarray last  = -1*np.ones(numMergers, dtype=long)                                      # Map to previous merger
    cdef np.ndarray inter = -1.0*np.ones(numMergers, dtype=float)                                  # Time to next merger


    ### Iterate Over Each Merger, In Order of Merger Time ###

    for ii in xrange(numMergers):

        if( ii > 0 and ii%hunth == 0 ): 
            print "%5d/%d" % (ii, numMergers)

        # Conver to sorted index
        ss = sort[ii]

        # Get the output ID from this merger
        out = ids[ss,1]

        ## Iterate over all Later Mergers ##

        for jj in xrange(ii+1, numMergers):
            
            # Convert to sorted index
            tt = sort[jj]
            
            # If previous merger goes into this one; save relationships
            if( ids[tt,0] == out or ids[tt,1] == out ):
                # Set index of next merger
                next[ss] = tt
                # Set index of last merger
                last[tt] = ss
                # Set time between mergers
                inter[ss] = times[tt] - times[ss]


        # } jj

    # } ii

    return inter, next, last






