# =================================================================================================
# MatchDetails.pyx
# ----------------
#
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# =================================================================================================


import numpy as np
cimport numpy as np


def getDetailIndicesForMergers(np.ndarray[long, ndim=1] active, np.ndarray[long, ndim=1] inid,
                               np.ndarray[long, ndim=1] outid, np.ndarray[long, ndim=1] detid ):
    """
    Match merger BHs to details entries for a particular snapshot.

    'Details' are entries written by Illustris for each active BH at each
    timestep, which includes info like mass and mdot, etc.
    'Mergers' are pairs ('in' and 'out') of BHs which merge into a single 'out'
    BH.  This function finds the index (line number) of the first 'details'
    entry to match the merger-remnant (the surviving BH: 'out').

    The array ``detid`` gives the BH ID for each details entry in this snapshot
    (i.e. between the start and end times corresponding to this snapshot).
    The arrays ``inid`` and ``outid`` give the BH ID numbers for each BH in
    each merger event (all mergers, not just during this snapshot).
    The ``active`` array gives the Mergers which are 'active' in this snapshot,
    i.e. mergers which have occured before the time of this snapshot.


    This function takes as input the indices of active BH mergers as
    ``active``.  The in and out BH indices for *all* mergers are given in
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
    active : array, long
        Array of indices corresponding to 'active' (already formed) binary
        systems to be matched.  Length is the number of active mergers ``N``.
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
    active BHs are described below.

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
    cdef int numMergers = inid.shape[0]                                                             # Length of both 'inid' and 'outid'
    cdef int numDetails = detid.shape[0]
    cdef int numActive  = active.shape[0]

    # Find indices to sort ID arrays for faster searching
    cdef np.ndarray sort_inid = np.argsort(inid)
    cdef np.ndarray sort_outid = np.argsort(outid)
    cdef np.ndarray sort_detid = np.argsort(detid)

    cdef long ind, target, found
    cdef int MAX_COUNT = 100

    # Initialize arrays to store results; default to '-1'
    cdef np.ndarray retinds = -1*np.ones(numActive, dtype=long)
    cdef np.ndarray targetID = -1*np.ones(numActive, dtype=long)
    cdef np.ndarray foundID = -1*np.ones(numActive, dtype=long)

    ### Iterate over Each Active Merger Binary System ###
    for mm in range(numActive):

        target = outid[active[mm]]
        found = target
        # Store the target ID in the output array
        targetID[mm] = target

        # Try to find binary (out bh) in details
        ind = np.searchsorted( detid, target, 'left', sorter=sort_detid )

        # If search failed, set ind to invalid
        if( detid[sort_detid[ind]] != found ): ind = -1

        # If search failed; see if this 'out' bh merged again, update 'out' id to that
        count = 0
        while( ind < 0 ):

            # Check if we are stuck, if so, break
            if( count >= MAX_COUNT ):
                ind = -1
                break

            ### See if this 'out' BH merged again ###
            #   i.e. if it was the 'in' BH of a later merger

            ind = np.searchsorted( inid, found, 'left', sorter=sort_inid )

            # If target is not an 'in bh' then it is missing, break
            if( inid[sort_inid[ind]] != found ):
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

    # } mm

    return targetID, foundID, retinds












def detailsForBlackholes(np.ndarray[long, ndim=1] bhIDs, np.ndarray[double, ndim=1] bhTimes,
                         np.ndarray[long, ndim=1] detIDs, np.ndarray[double, ndim=1] detTimes):
    """
    Match merger BHs by ID to details entries just before and just after merger.

    Arguments
    ---------
    bhIDs    : array[N] long, merger BH ID numbers
    bhTimes  : array[N] float, merger-BH merger times (scalefactors)
    detIDs   : array[N] long, details entry ID numbers
    detTimes : array[N] float, detail entry times (scalefactors)

    Returns
    -------
    indsBef : array[N] long, index of bh-details file containing match before merger
    indsAft : array[N] long, index of bh-details file containing match after  merger

    """

    # Get the number of target BHs
    cdef int nums = bhIDs.shape[0]

    # Sort details first by ID, then by time in reverse (last in time comes first in list)
    cdef np.ndarray sort_det = np.lexsort( (-detTimes, detIDs) )

    # Sort target merger BH IDs
    cdef np.ndarray sort_bh = np.argsort(bhIDs)

    cdef int ii,jj

    # Initialize arrays to store results; default to '-1' for invalid entries
    cdef np.ndarray indsBef = -1*np.ones(nums, dtype=long)
    cdef np.ndarray indsAft = -1*np.ones(nums, dtype=long)

    ### Iterate over Each Active Merger Binary System ###
    jj = 0
    for ii in xrange(nums):

        # Get the sorted index
        ind = sort_bh[ii]
        # Get the sorted BH ID and time
        bh = bhIDs[ind]
        bhtime = bhTimes[ind]

        # Find Matching IDs
        while( detIDs[sort_det[jj]] < bh ): jj += 1
        # Once match is found; find times before merger
        while( detIDs[sort_det[jj]] == bh and detTimes[sort_det[jj]] >= bhtime ): jj += 1

        # If this is still a match, it is before the merger --- store index
        if( detIDs[sort_det[jj]] == bh ):
            # Store entry at sorted index (corresponding to current BH)
            indsBef[ind] = sort_det[jj]

        # If previous was a match, it was after merger --- store index
        if( jj > 0 and detIDs[sort_det[jj-1]] == bh ):
            indsAft[ind] = sort_det[jj-1]

    # } ii

    return indsBef, indsAft

# detailsForBlackholes()






