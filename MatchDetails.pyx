

import numpy as np
cimport numpy as np






def getDetailIndicesForMergers(np.ndarray[long, ndim=1] active, np.ndarray[long, ndim=1] inid, np.ndarray[long, ndim=1] outid, np.ndarray[long, ndim=1] detid ):
    #                               np.ndarray[double, ndim=1] scales):

    cdef int numMergers = inid.shape[0]
    cdef int numDetails = detid.shape[0]
    cdef int numActive  = active.shape[0]

    # Find indices to sort ID arrays
    cdef np.ndarray sort_inid = np.argsort(inid)
    cdef np.ndarray sort_outid = np.argsort(outid)
    cdef np.ndarray sort_detid = np.argsort(detid)

    #cdef double temp = 0.0
    cdef int count = 0
    cdef int firstTry = 0
    cdef int missing = 0
    cdef int maxedOut = 0
    cdef int good = 0
    cdef int bad = 0
    
    cdef long ind, target, found
    cdef int MAX_COUNT = 100

    # Array to store results
    cdef np.ndarray retinds = -1*np.ones(numActive, dtype=long)
    cdef np.ndarray targetID = -1*np.ones(numActive, dtype=long)
    cdef np.ndarray foundID = -1*np.ones(numActive, dtype=long)

    ### Iterate over Each Merger Binary System ###
    for mm in range(numActive):

        target = outid[active[mm]]
        found = target
        targetID[mm] = target

        # Try to find binary (out bh) in details
        ind = np.searchsorted( detid, target, 'left', sorter=sort_detid )

        # Check if search succeeded
        if( detid[sort_detid[ind]] != found ): ind = -1
        else: firstTry += 1

        # If search failed; see if this 'out' bh merged again, update 'out' id to that
        count = 0
        while( ind < 0 ):
            
            # Check if we are stuck, if so, break
            if( count >= MAX_COUNT ):
                maxedOut += 1
                ind = -1
                break

            ### See if this 'out' BH merged again ###

            ind = np.searchsorted( inid, found, 'left', sorter=sort_inid )

            # If target is not an 'in bh' then it is missing, break
            if( inid[sort_inid[ind]] != found ):
                missing += 1
                ind = -1
                break

            ### Redo details search with new 'out id'

            # Set new 'out id' to match partner of 'in id'
            found = outid[sort_inid[ind]]
            
            # Redo search
            ind = np.searchsorted( detid, found, 'left', sorter=sort_detid )

            # Check if search succeeded
            if( detid[sort_detid[ind]] != found ): ind = -1

            # Increment counter to make sure not stuck
            count += 1

        # } while

        # If we have a match, store results
        if( ind >= 0 ): 
            # Store matching Details index
            retinds[mm] = sort_detid[ind]
            # Store the ID which was eventually matched
            foundID[mm] = found
            good += 1
        else:
            bad += 1

    # } mm

    return targetID, foundID, retinds









###  =============================================================  ###
###  ===============  WRAPPER FOR CYTHON FUNCTIONS  ==============  ###
###  =============================================================  ###



'''
def getDetailIndicesForMergers(np.ndarray[long, ndim=1] inid, np.ndarray[long, ndim=1] outid, np.ndarray[long, ndim=1] detid, 
                               np.ndarray[double, ndim=1] scales)
    """ Wrapper to call _timeMatch """

    # Call cython function
    inds = _getDetailIndicesForMergers(mid, mts, snap, did, time, life)

    return inds
'''








