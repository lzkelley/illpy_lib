"""
Find subhalos based on target properties.


"""

from Constants import *
import readtreeHDF5
import illpy
from illpy.Constants import *


RUN = 3
VERBOSE = True

TARGET_LOOKBACK_TIME = 1.0e9                                                                        # [years]


def main(run=RUN, verbose=VERBOSE):

    if( verbose ): print "SubhaloFinder.py"

    treePath = ILLUSTRIS_TREE_PATHS[run]

    if( verbose ): print " - Run %d : '%s'" % (run, treePath)

    ### Load Cosmology ###
    if( verbose ): print " - Loading Cosmology"
    cosmo = illpy.Cosmology()


    ### Load Merger Tree ###
    if( verbose ): print " - Loading Merger Tree"
    tree = readtreeHDF5.TreeDB(treePath)


    # Determine Starting Snapshot
    if( verbose ): print " - Find Target Snapshot" 
    target = targetSnapshot(cosmo=cosmo, verbose=verbose)

    # Load target snapshot subhalo catalog
    


    return tree, cosmo



def targetSnapshot(lbtime=TARGET_LOOKBACK_TIME, cosmo=None, verbose=VERBOSE):
    """
    Find the snapshot nearest the given lookback time
    """


    if( verbose ): print " - SubhaloFinder.targetSnapshot()"
    if( cosmo is None ): cosmo = illpy.Cosmology()

    lbt = cosmo.lookbackTime(cosmo.snapshotTimes())
    lbtime *= YEAR

    # Find minimum difference between lookback times and target LB time
    if( verbose ): print " - - Target Lookback Time = %.2e [Gyr]" % (lbtime/GYR)
    diffs = np.fabs(lbt - lbtime)
    inds = np.argmin(diffs)

    if( verbose ): 
        print " - - Nearest snapshot = %d/%d, Time = %.2e [Gyr]" % (inds, len(lbt), lbt[inds]/GYR)
        scale = cosmo.snapshotTimes(inds)
        redz  = cosmo.redshift(scale)
        print " - - - Scalefactor = %.4f, Redshift = %.4f" % (scale, redz)

    return inds




if __name__ == '__main__': main()
