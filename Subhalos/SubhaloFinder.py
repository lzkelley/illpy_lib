"""
Find subhalos based on target properties.


"""


import readtreeHDF5
import readsubfHDF5

import illpy
from illpy.Constants import *
from Constants import *



RUN = 3
VERBOSE = True

TARGET_LOOKBACK_TIME = 1.0e9                                                                        # [years]

MIN_STAR_PARTICLES = 10
MIN_GAS_PARTICLES  = 20
MIN_BH_PARTICLES   = 1


def main(run=RUN, verbose=VERBOSE):

    if( verbose ): print "SubhaloFinder.py"

    treePath = ILLUSTRIS_TREE_PATHS(run)

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



    ### Load target snapshot subhalo catalog ###
    if( verbose ): print " - Loading Subhalo Catalog"
    cat = loadSubhaloCatalog(run, target, keys=SUBFIND_PARAMETERS, verbose=verbose)
    numSubhalos = len(cat[SH_SFR])
    if( verbose ): print " - - Loaded catalog with %d subhalos" % (numSubhalos)

    # Find valid subhalos
    if( verbose ): print " - Finding valid subhalos"
    inds = filterSubhalos(cat, verbose=verbose)
    if( verbose ): print " - - Found %d/%d valid subhalos" % (len(inds), numSubhalos)


    return cat, inds




def filterSubhalos(cat, ngas=MIN_GAS_PARTICLES, nstar=MIN_STAR_PARTICLES, nbh=MIN_BH_PARTICLES, 
                   verbose=VERBOSE):
    
    if( verbose ): print " - - SubhaloFinder.filterSubhalos()"

    filt = dict(cat)

    lenTypes = filt[SH_LEN_TYPE]
    nums = len(filt[SH_SFR])
    
    reqs = [ ngas, nstar, nbh ]
    types = [ PARTICLE_TYPE_GAS, PARTICLE_TYPE_STAR, PARTICLE_TYPE_BH ]
    names = [ 'Gas', 'Star', 'BH' ]
    inds = set(range(nums))


    ### Find Good Subhalos ###
    for num, typ, nam in zip(reqs, types, names):

        temp = np.where( lenTypes[:,typ] >= num )[0]
        #inds = inds.union(temp)
        inds = inds.intersection(temp)
        if( verbose ): 
            print " - - - Requiring at least %d '%s' (%d) Particles" % (num, nam, typ)
            print " - - - - %d/%d Valid (%d left)" % (len(temp), nums, len(inds))

            
    '''
    ### Remove Subhalos ###
    for key in filt.keys():
        print "\nOld shape for '%s' = %s" % (str(key), str(np.shape(filt[key])))

        if( len(np.shape(filt[key])) > 1 ): axis = 0
        else:                               axis = None

        filt[key] = np.delete(filt[key], list(inds), axis=axis )
        print "New shape for '%s' = %s" % (str(key), str(np.shape(filt[key])))

        return filt

    '''

    return np.array(list(inds))
    



def loadSubhaloCatalog(run, target, keys=SUBFIND_PARAMETERS, verbose=VERBOSE):
    """
    Load parameters from a particular snapshot's subfind catalog.
    """

    if( verbose ): print " - - SubhaloFinder.loadSubhaloCatalog()"

    # Get path to catalogs
    outputPath = ILLUSTRIS_OUTPUT_PATHS(run)

    # Load catalog
    if( verbose ): print " - Loading Subfind Catalog from '%s'" % (outputPath)
    subfcat = readsubfHDF5.subfind_catalog(outputPath, target, grpcat=False, subcat=True, keysel=keys)
    
    # Convert to dictionary
    cat = { akey : getattr(subfcat, akey) for akey in keys }

    return cat





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
