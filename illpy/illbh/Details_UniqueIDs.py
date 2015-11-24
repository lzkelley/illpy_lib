"""
"""

import os
import numpy as np
from mpi4py import MPI

import zcode.inout as zio

from BHConstants import DETAILS, _LOG_DIR, _distributeSnapshots, GET_DETAILS_UNIQUE_IDS_FILENAME, \
    _checkLoadSave


__version__ = '0.23'


def main(run=1, log=None):
    """
    """
    # Initialization
    # --------------
    #     MPI Parameters
    comm = MPI.COMM_WORLD
    rank = comm.rank
    name = __file__
    header = "\n%s\n%s\n%s" % (name, '='*len(name), str(datetime.now()))

    if(rank == 0):
        zio.checkPath(_LOG_DIR)
    comm.Barrier()

    # Initialize log
    log = illpy.illbh.BHConstants._loadLogger(
        __file__, debug=True, verbose=True, run=run, rank=rank, version=__version__)
    log.debug(header)
    if(rank == 0):
        print("Log filename = ", log.filename)



    return


def _uniqueIDs(run, comm, log):
    log.debug("_uniqueIDs()")
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    mySnaps = _distributeSnapshots(comm)

    log.info("Rank {:d}/{:d} with {:d} Snapshots [{:d} ... {:d}]".format(
        rank, size, mySnaps.size, mySnaps.min(), mySnaps.max()))

    nums, scales, masses, mdots, dens, csnds, ids = \
        _uniqueIDs_snaps(run, mySnaps, bhIDsUnique, _MAX_DETAILS_PER_SNAP, log)


    # Iterate over target snapshots
    # -----------------------------
    for snap in snapList:
        log.debug("snap = %03d" % (snap))



def loadUniqueIDs(run, snap, rank=None, loadsave=True, log=None):
    """
    """
    log.debug("loadUniqueIDs()")
    if log is None:
        # Initialize log
        log = illpy.illbh.BHConstants._loadLogger(
            __file__, debug=False, verbose=True, run=run, rank=rank, version=__version__)
        log.debug(header)
        if(rank == 0):
            print("Log filename = ", log.filename)
    

    fname = GET_DETAILS_UNIQUE_IDS_FILENAME(run, snap, __version__)
    data = _checkLoadSave(fname, loadsave, log)
    if data is None:
        # Load `BHDetails`
        dets = loadBHDetails(run, snap, verbose=False)
        numDets = dets[DETAILS.NUM]
        log.debug(" - Snap %d: %d Details" % (snap, numDets))
        detIDs = dets[DETAILS.IDS]
        if(np.size(detIDs) == 0): continue
        

        
        

    return data




if __name__ == "__main__":
    main()
