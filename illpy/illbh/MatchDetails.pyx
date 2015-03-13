"""



"""

import numpy as np
cimport numpy as np

from BHConstants import IN_BH, OUT_BH, DETAILS_BEFORE, DETAILS_AFTER, DETAILS_FIRST


def getDetailIndicesForMergers(np.ndarray[long,  ndim=1] target, np.ndarray[long, ndim=2] mid, 
                               np.ndarray[float, ndim=1] mtime,  np.ndarray[long, ndim=3] lind,
                               np.ndarray[float, ndim=3] ltime,
                               np.ndarray[long,  ndim=1] detid,  np.ndarray[long, ndim=1] dtime  ):
    """
    Match merger BHs to details entries for a particular snapshot.

    This function takes as input the indices of active BH mergers as
    ``target``.  The in and out BH indices for *all* mergers are given in
    ``inid`` and ``outid``.  The ``outid``s are cross checked with each of
    IDs for the details entries given in ``detid``.  When matches are missing,
    it is assumed that the 'out' BH later merged into another system, in which
    case the resulting BH ID could have been different.  The initial 'out' BH
    which was targeted are returned in the resulting array ``targetID``,
    whereas the actual BH ID which was found (perhaps after numerous mergers
    inbetween) is returned in the array `foundID`.  Finally, the index of the
    Details entry where those IDs are finally found are returned in the array
    ``retinds`` which can then be used to extract the details information for
    each binary.

    The Details files have (up to) numerous entries for each BH.  Currently
    only one of them is selected and added to the ``retinds`` array.  The first
    match is returned --- which *may* be the earliest in time.


    Parameters
    ----------
    target : array, long
        Array of indices corresponding to 'target' (already formed) binary
        systems to be matched.  Length is the number of target mergers ``N``.
    inid : array, long
        Array of indices for Merger 'in' BHs.  Length is the total number
        of Mergers, same as ``outid``.
    outid : array, long
        Array of indices for Merger 'out' BHs.  Length is the total number
        of Mergers, same as ``inid``.
    detit : array, long
        Array of indices for each Details entry.  Length is the number of
        details entries.

    Returns
    -------
    3 arrays are returned, each of length equal to the total number of Mergers.
    Entries for *inactive* systems have values set to `-1`.  The values for
    target BHs are described below.

    targetID : array, long
        Indices of each merger 'out' BH which is the target.  Length is the
        total number of Mergers.
    foundID : array, long
        Indices of each merger which were actually matched in the Details
        entries.  ``targetID`` gives each mergers' 'out' BH, but many of those
        later merged again --- leading to a different ID number for the
        resulting system.  Those resulting IDs are given by ``foundID``.
    retinds : array, long
        Indices of the Details entries which correspond to each Merger.

    """
    
    
    # Get the lengths of all input arrays
    cdef int numMergers = mid.shape[0]
    cdef int numDetails = detid.shape[0]
    cdef int numTarget  = target.shape[0]

    # Sort first by ID then by time
    cdef np.ndarray s_det = np.lexsort( (dtime, detid) )

    # Declare variables
    cdef np.ndarray s_mrg
    cdef long s_ind, t_id, s_first_match, first_match, s_before_match, s_after_match
    cdef long d_ind_first, d_ind_before, d_ind_after
    cdef float t_time, d_first_time, d_before_time, d_after_time

    ### Iterate over Each Target Merger Binary System ###
    
    ### Iterate over Each Type of BH ###
    for BH in range([IN_BH, OUT_BH]):

        # Sort *target* mergers first by ID, then by Merger Time
        s_target = np.lexsort( (mtime[target], mid[target,BH]) )
        

        ### Iterate over Each Entry ###
        for ii in range(numTarget):
            s_ind  = target[s_target[ii]]                                                           # Sorted, target-merger index
            t_id   = mid[s_ind, BH]                                                                 # Target-merger BH ID
            t_time = mtime[s_ind]                                                                   # Target-merger BH Time

            ### Find First Match ###
            #   Find index in sorted 'det' arrays with first match to target ID `t_id`
            s_first_match = np.searchsorted( detid, t_id, 'left', sorter=s_det )

            # If this is not a match, no matching entries, continue to next BH
            if( detid[s_det[s_first_match]] != t_id ): continue
            
            ## Store first match
            d_ind_first = s_det[s_first_match]
            d_first_time  = dtime[d_ind_first]
            # if there are no previous matches or new match is *earlier*
            if( lind[s_ind, BH, DETAILS_FIRST] < 0 or d_first_time < ltime[s_ind, BH, DETAILS_FIRST] ):
                lind [s_ind, BH, DETAILS_FIRST] = d_ind_first                                       # Set link to details index
                ltime[s_ind, BH, DETAILS_FIRST] = d_first_time                                      # Set link to details time


            ### Find 'before' Match ###
            #   Find the *latest* detail ID match *before* the merger time
            s_before_match = s_first_match                                                          # Has to come after the 'first' match
            # Increment if the next entry is also an ID match and next time is still *before*
            while( detid[s_det[s_before_match+1]] == t_id and dtime[s_det[s_before_match+1]] < t_time ):
                s_before_match += 1


            ## Store Before Match 
            d_ind_before  = s_det[s_before_match]
            d_before_time = dtime[d_ind_before]
            # If we no longer match the ID, something is wrong
            if( detid[d_ind_before] != t_id ):
                raise RuntimeError("ID '%d' (merger %d) no longer matches 'before'!!" % (t_id, s_ind))

            # if there are no previous matches or new match is *later*
            if( lind[s_ind, BH, DETAILS_BEFORE] < 0 or d_before_time > ltime[s_ind, BH, DETAILS_BEFORE] ):
                lind [s_ind, BH, DETAILS_BEFORE] = d_ind_before                                     # Set link to details index
                ltime[s_ind, BH, DETAILS_BEFORE] = d_before_time                                    # Set link to details time
            

            ## Find 'after' Match
            #  Find the *earliest* detail ID match *after* the merger time
            #  Only exists if this is the 'out' BH
            if( BH == OUT_BH ):

                s_after_match = s_before_match                                                      # Has to come after the 'before' match
                # Increment if the next entry is also an ID match, but this time is still *before*
                while( detid[s_det[s_after_match+1]] == t_id and dtime[s_det[s_after_match]] < t_time ):
                    s_after_match += 1

                ## Store After Match 
                d_ind_after  = s_det[s_after_match]
                d_after_time = dtime[d_ind_after]
                # If we no longer match the ID, something is wrong
                if( detid[d_ind_before] != t_id ):
                    raise RuntimeError("ID '%d' (merger %d) no longer matches 'before'!!" % (t_id, s_ind))

                # if there are no previous matches or new match is *earlier*
                if( lind[s_ind, BH, DETAILS_AFTER] < 0 or d_after_time < ltime[s_ind, BH, DETAILS_AFTER] ):
                    lind [s_ind, BH, DETAILS_AFTER] = d_ind_after                                     # Set link to details index
                    ltime[s_ind, BH, DETAILS_AFTER] = d_after_time                                    # Set link to details time
            


    return 







