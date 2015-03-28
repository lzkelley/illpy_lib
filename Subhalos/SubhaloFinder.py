"""
Find subhalos based on target properties.


"""


import readtreeHDF5
import readsubfHDF5

import illpy
from illpy.Constants import *
from Constants import *

import Figures


RUN     = 3
SFR_CUT = 0.9
VERBOSE = True

TARGET_LOOKBACK_TIME = 1.0e9                                                                        # [years]

MIN_STAR_PARTICLES = 10
MIN_GAS_PARTICLES  = 20
MIN_BH_PARTICLES   = 1

MIN_NUM_SUBHALOS   = 10

def main(run=RUN, verbose=VERBOSE):

    if( verbose ): print "SubhaloFinder.py"

    treePath = ILLUSTRIS_TREE_PATHS(run)
    cut = SFR_CUT
    
    if( verbose ): print " - Run %d : '%s'" % (run, treePath)

    ### Load Cosmology ###
    if( verbose ): print " - Loading Cosmology"
    cosmo = illpy.Cosmology()



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

    # Plot initial subhalos
    Figures.figa01.plotFigA01_Subfind_SFR( run, cat[SH_SFR][inds], cat[SH_MASS_TYPE][inds], cat[SH_BH_MASS][inds] )


    ### Select Subhalos Based on Star Formation Rate ###
    sfrInds = sfrSubhalos(cat, inds, cut=cut, verbose=verbose)




    ### Load Merger Tree ###
    if( verbose ): print " - Loading Merger Tree"
    tree = readtreeHDF5.TreeDB(treePath)



    
    return cat, inds, sfrInds, tree



def getSubhaloBranches(inds, tree, target, verbose=VERBOSE):

    if( verbose ): print " - - SubhaloFinder.getSubhaloBranches()"
    numSubhalos = len(inds)

    assert numSubhalos > MIN_NUM_SUBHALOS, "There are only %d Subhalos!!!" % (numSubhalos)

    # Get snapshot numbers included in the branches
    branchSnaps = np.arange(target, NUM_SNAPS)[::-1]
    branchLens = NUM_SNAPS - target
    if( verbose ): print " - - - Branch Snaps = %d [%d,%d]" % (branchLens, branchSnaps[-1], branchSnaps[0])


    # Make sure branches are long enough to analyze
    assert branchLens > 2, "``branchLens`` < 2!!"
    

    ### Build Dictionary To Store Branches Data ###

    # Get sample branch to determine shapes of parameters
    desc = tree.get_future_branch(target, inds[0], keysel=SUBFIND_PARAMETERS)

    # Initialize dictionary with zeros arrays for each parameter
    epas = {}
    for key in SUBFIND_PARAMETERS:
        val = getattr(desc, key)
        shp = np.shape(val)

        if( len(shp) == 1 ): epas[key] = np.zeros( [numSubhalos, branchLens],         dtype=FLT )
        else:                epas[key] = np.zeros( [numSubhalos, branchLens, shp[1]], dtype=FLT )

    # } key

    
    ### Find Branches for Subhalos and Store Parameters ###
    numMissing = 0
    if( verbose ): print " - - - Finding brancehs for %d subhalos" % (len(inds))
    for ii,subh in enumerate(inds):
        # Get Branches
        desc = tree.get_future_branch(target, subh, keysel=np.append(SUBFIND_PARAMETERS, SL_SNAP_NUM))
        nums = getattr(desc, SL_SNAP_NUM)
        missing_flag = False
        if( len(nums) != branchLens ):
            if( verbose ): print "!!!! ii = %d, subhalo = %d, branch length = %d" % (ii, subh, len(nums))
            missing_flag = True
            numMissing += 1

        # Store Parameters
        for key in SUBFIND_PARAMETERS:
            val = getattr(desc, key)

            # If a snapshot is missing, insert each snapshots value manually
            if( missing_flag ): 
                for jj,sn in enumerate(nums):
                    kk = np.where( branchSnaps == sn )[0]
                    if( len(kk) == 1 ): epas[key][jj,kk] = val[jj]

            # If all snapshots exist, insert full array
            else:
                epas[key][ii] = getattr(desc, key)

    # } ii
                
    if( verbose ): print " - - - - Retrieved branches with %d unexpected lengths" % (numMissing)



    return epas



def sfrSubhalos(cat, inds, cut=SFR_CUT, specific=False, verbose=VERBOSE):

    if( verbose ): print " - - SubhaloFinder.sfrSubhalos()"

    sfr = cat[SH_SFR][inds]

    # Convert to Specific SFR
    if( specific ):
        vstr1 = " specific"
        vstr2 = "[(Msol/yr)/Msol]"
        stellarMass = cat[SH_MASS_TYPE][inds,PARTICLE_TYPE_STAR]*MASS_CONV
        sfr /= stellarMass
    else:
        vstr1 = ""
        vstr2 = "[Msol/yr]"


    setAve = np.average(sfr)

    # Find target percentile cut
    cutVal = np.percentile(sfr, cut*100.0)
    if( verbose ): print " - - - Cutting%s SFR above %.3f, %.3e %s" % (vstr1, cut, cutVal, vstr2)

    # Find indices Within ``inds`` above cut
    subInds = np.where( sfr >= cutVal )[0]
    subAve = np.average(sfr[subInds])
    # Conver to overall indices
    sfrInds = inds[subInds]

    if( verbose ): 
        print " - - - From %d to %d Subhalos (All %d)" % (len(inds), len(sfrInds), len(cat[SH_SFR]))
        print " - - - Ave before and after cut = %.2e, %.2e %s" % (setAve, subAve, vstr2)


    return sfrInds
    



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
