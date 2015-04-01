"""
Find subhalos based on target properties.


"""


import readtreeHDF5
import readsubfHDF5

import sys
import illpy
from illpy import Cosmology
from illpy.Constants import *
from Constants import *

import Figures
from StellarLifetimes import StellarLifetimes

RUN     = 3
SFR_CUT = 0.9
VERBOSE = True
PLOT    = True

TARGET_LOOKBACK_TIMES = [ 1.5e9, 1.5e8 ]                                                            # [years]
TARGET_STELLAR_MASSES = [ 2.0, 3.0 ]

MIN_STAR_PARTICLES = 10
MIN_GAS_PARTICLES  = 20
MIN_BH_PARTICLES   = 1

MIN_NUM_SUBHALOS   = 10



def main(run=RUN, verbose=VERBOSE, plot=PLOT):

    if( verbose ): print "SubhaloFinder.py"

    treePath = ILLUSTRIS_TREE_PATHS(run)
    cut = SFR_CUT
    
    if( verbose ): print " - Run %d : '%s'" % (run, treePath)

    ### Load Cosmology ###
    if( verbose ): print " - Loading Cosmology"
    cosmo = illpy.Cosmology()

    ### Load Merger Tree ###
    if( verbose ): print " - Loading Merger Tree"
    tree = readtreeHDF5.TreeDB(treePath)


    # Determine Starting Snapshot
    if( verbose ): print " - Find Target Snapshot" 
    hiSnaps, loSnaps = targetSnapshots(masses=TARGET_STELLAR_MASSES, cosmo=cosmo, verbose=verbose)
    firstSnap = np.min(np.concatenate([hiSnaps, loSnaps]))
    if( verbose ): print " - - Earliest snapshot = %d" % (firstSnap)

    ### Load target snapshot subhalo catalog ###
    if( verbose ): print " - Loading Subhalo Catalog"
    cat = loadSubhaloCatalog(run, firstSnap, keys=SUBFIND_PARAMETERS, verbose=verbose)
    numSubhalos = len(cat[SH_SFR])
    if( verbose ): print " - - Loaded catalog with %d subhalos" % (numSubhalos)


    # Find valid subhalos
    if( verbose ): print " - Finding valid subhalos"
    inds = filterSubhalos_minParticles(cat, verbose=verbose)
    if( verbose ): print " - - Found %d/%d valid subhalos" % (len(inds), numSubhalos)

    # Plot initial subhalos
    if( plot ): 
        cat_sfr = cat[SH_SFR][inds]
        cat_mass_type = cat[SH_MASS_TYPE][inds]
        cat_mass_bh = cat[SH_BH_MASS][inds]
        Figures.figa01.plotFigA01_Subfind_SFR( run, cat_sfr, cat_mass_type, cat_mass_bh )


    #inds = inds[:100]

    ### Get Branches of Target Halos ###
    if( verbose ): print " - Loading branches for %d selected subhalos" % (len(inds))
    epas, snaps, sfrInds = getSubhaloBranches(inds, tree, firstSnap, verbose=verbose)

    illpy.AuxFuncs.saveDictNPZ(epas, "epas.npz", verbose=verbose)


    ### Select Subhalos Based on Star Formation Rate ###
    #sfrInds, sfrs = sfrSubhalos(cat, inds, cut=cut, verbose=verbose)

    ### Get Weights for Quality of EplusA Galaxies ###
    weights = weightEplusA_sfr(epas, hiSnaps, loSnaps, verbose=verbose)


    if( plot ): Figures.figa02.plotFigA02_Branches_SFR(run, epas, hiSnaps, loSnaps, weights)


    return run, firstSnap, cat, inds, sfrInds, tree, epas, snaps, weights




def weightEplusA_sfr(epas, his, los, cosmo=None, verbose=VERBOSE):

    if( verbose ): print " - - SubhaloFinder.weightEplusA_sfr()"

    if( cosmo is None ): cosmo = Cosmology()

    snaps = np.concatenate([his,los])
    first = np.min(snaps)

    sfr = epas[SH_SFR]
    hiInds = his - first
    loInds = los - first

    ### Get Time Between Snapshots ###
    #   zeroth duration corresponds to time UP-TO zeroth snapshot
    durSnaps = np.concatenate([[first-1],snaps])
    scales  = cosmo.snapshotTimes(durSnaps)
    lbtimes = cosmo.age(scales)
    durs = np.diff(lbtimes)

    # Estimate Mass of Stars Formed
    starsFormed = sfr*durs/YEAR

    hiStars = np.sum(starsFormed[:,hiInds], axis=1)
    loStars = np.sum(starsFormed[:,loInds], axis=1)

    weights = hiStars/loStars
    
    return weights

# } weightEplusA_sfr()




def getSubhaloBranches(inds, tree, target, ngas=MIN_GAS_PARTICLES, nstar=MIN_STAR_PARTICLES, 
                       nbh=MIN_BH_PARTICLES, verbose=VERBOSE):

    if( verbose ): print " - - SubhaloFinder.getSubhaloBranches()"
    numSubhalos = len(inds)
    if( numSubhalos < 100 ): interval = 1
    else:                    interval = np.int( np.floor(numSubhalos/100.0) )


    #assert numSubhalos > MIN_NUM_SUBHALOS, "There are only %d Subhalos!!!" % (numSubhalos)

    # Get snapshot numbers included in the branches
    branchSnaps = np.arange(target, NUM_SNAPS)
    branchLens = NUM_SNAPS - target
    if( verbose ): print " - - - Branch Snaps = %d [%d,%d]" % (branchLens, branchSnaps[0], branchSnaps[-1])


    # Make sure branches are long enough to analyze
    assert branchLens > 2, "``branchLens`` < 2!!"

    # Also get snapshot numbers 
    useKeys = np.concatenate([SUBFIND_PARAMETERS, SUBLINK_PARAMETERS]).tolist()

    ### Build Dictionary To Store Branches Data ###

    # Get sample branch to determine shapes of parameters
    desc = tree.get_future_branch(target, inds[0], keysel=useKeys)


    # Initialize dictionary with zeros arrays for each parameter
    epas = {}
    for key in useKeys:
        val = getattr(desc, key)
        shp = np.shape(val)

        if( len(shp) == 1 ): epas[key] = np.zeros( [numSubhalos, branchLens],         dtype=FLT )
        else:                epas[key] = np.zeros( [numSubhalos, branchLens, shp[1]], dtype=FLT )

    # } key

    
    ### Find Branches for Subhalos and Store Parameters ###
    badSFR = -1

    numMissing = 0
    numNaked = 0
    numZeroSFR = 0
    badInds = []
    if( verbose ): print " - - - Finding branches for %d subhalos" % (len(inds))
    for ii,subh in enumerate(inds):
        # Get Branches
        desc = tree.get_future_branch(target, subh, keysel=useKeys)

        # Print progress
        if( verbose ):
            if( ii % interval == 0 or ii == numSubhalos-1):
                sys.stdout.write('\r - - - - %.2f%% Complete' % (100.0*ii/(numSubhalos-1)))

            if( ii == numSubhalos-1 ): sys.stdout.write('\n')
            sys.stdout.flush()


        # Make sure subhalo found in all future snapshots
        nums = getattr(desc, SL_SNAP_NUM)
        if( len(nums) != branchLens ):
            numMissing += 1
            badInds.append(ii)
            continue

        # Make sure subhalo has enough particles in all future snapshots
        lenTypes = getattr(desc, SH_LEN_TYPE)
        if( any(lenTypes[:, PARTICLE_TYPE_GAS ] < ngas ) or 
            any(lenTypes[:, PARTICLE_TYPE_STAR] < nstar) or
            any(lenTypes[:, PARTICLE_TYPE_BH  ] < nbh  )   ):
            numNaked += 1
            badInds.append(ii)
            continue

        # Make sure SFR is never uniformly Zero
        sfr = getattr(desc, SH_SFR)
        if( any(sfr <= 0.0) ):
            numZeroSFR += 1
            badInds.append(ii)
            if( badSFR < 0 ): badSFR = subh
            continue


        # Sort by snapshot number
        sort = np.argsort(nums)

        # Store Parameters
        for key in useKeys:
            epas[key][ii] = getattr(desc, key)[sort]                                                # In CHRONOLOGICAL order

    # } ii


    ### Cleanup bad Subhalos (no branches) ###
    inds = np.delete(inds, badInds)
    for key in useKeys:
        epas[key] = np.delete(epas[key], badInds, axis=0)

                
    if( verbose ): 
        print " - - - - Retrieved %d branches with (%d missing, %d lacking, %d zero SFR)" % \
            (len(epas[SH_SFR]), numMissing, numNaked, numZeroSFR)
        if( numZeroSFR > 0 ): print " - - - - - Zero SFR at Subhalo %d" % (badSFR)


    return epas, branchSnaps, inds

# } getSubhaloBranches()




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
        
    # Find target percentile cut (``cut`` given as fraction, convert to percentil)
    cutVal = np.percentile(sfr, cut*100.0)
    if( verbose ): print " - - - Cutting%s SFR above %.3f, %.3e %s" % (vstr1, cut, cutVal, vstr2)

    # Find indices Within ``inds`` above cut
    subInds = np.where( sfr >= cutVal )[0]
    subsfr = sfr[subInds]
    subAve = np.average(subsfr)
    # Conver to overall indices
    sfrInds = inds[subInds]

    if( verbose ): 
        print " - - - From %d to %d Subhalos (All %d)" % (len(inds), len(sfrInds), len(cat[SH_SFR]))
        print " - - -         Min,     Ave,     Max      %s" % (vstr2)
        print " - - - Before  %.2e %.2e %.2e" % (np.min(sfr), setAve, np.max(sfr))
        print " - - - After   %.2e %.2e %.2e" % (np.min(subsfr), subAve, np.max(subsfr))


    return sfrInds, subsfr

# sfrSubhalos()    



def filterSubhalos_minParticles(cat, ngas=MIN_GAS_PARTICLES, nstar=MIN_STAR_PARTICLES, 
                                nbh=MIN_BH_PARTICLES, verbose=VERBOSE):
    """
    
    Arguments
    ---------
        cat : <dict>, groupfind catalog containing subhalo properties
        ngas : <int>, 

    Notes
    -----
        [1] : Some subhalos seem to have zero star mass, but non-zero number of star
              particles.  Double check for this

    """
    
    
    if( verbose ): print " - - SubhaloFinder.filterSubhalos_minParticles()"

    lenTypes  = cat[SH_LEN_TYPE]
    massTypes = cat[SH_MASS_TYPE]
    nums = len(cat[SH_SFR])
    
    reqs  = [ ngas, nstar, nbh ]
    types = [ PARTICLE_TYPE_GAS, PARTICLE_TYPE_STAR, PARTICLE_TYPE_BH ]
    names = [ 'Gas', 'Star', 'BH' ]
    inds  = set(range(nums))

    ### Find Good Subhalos ###
    for num, typ, nam in zip(reqs, types, names):

        # Note[1]
        temp = np.where( (lenTypes[:,typ] >= num) & (massTypes[:,typ] > 0.0) )[0]
        inds = inds.intersection(temp)
        if( verbose ): 
            print " - - - Requiring at least %d '%s' (%d) Particles" % (num, nam, typ)
            print " - - - - %d/%d Valid (%d left)" % (len(temp), nums, len(inds))


    inds = np.array(list(inds))

    return inds

# filterSubhalos_minParticles()    



def loadSubhaloCatalog(run, target, keys=SUBFIND_PARAMETERS, verbose=VERBOSE):
    """
    Load parameters from a particular snapshot's subfind catalog.
    """

    if( verbose ): print " - - SubhaloFinder.loadSubhaloCatalog()"

    # Get path to catalogs
    outputPath = ILLUSTRIS_OUTPUT_PATHS(run)

    # Load catalog
    if( verbose ): print " - - - Loading Subfind Catalog from '%s'" % (outputPath)
    subfcat = readsubfHDF5.subfind_catalog(outputPath, target, grpcat=False, subcat=True, keysel=keys)
    
    # Convert to dictionary
    cat = { akey : getattr(subfcat, akey) for akey in keys }
    # Store snapshot number separately
    cat[SH_SNAPSHOT_NUM] = target
    cat[SH_FILENAME] = getattr(subfcat, SH_FILENAME)

    return cat





def targetSnapshots(times=TARGET_LOOKBACK_TIMES, masses=None, cosmo=None, verbose=VERBOSE):
    """
    Find the snapshots nearest the given lookback times
    """


    if( verbose ): print " - SubhaloFinder.targetSnapshot()"
    if( cosmo is None ): cosmo = illpy.Cosmology()

    # Get lookback times to all snapshots
    lbTimes = cosmo.lookbackTime(cosmo.snapshotTimes())
    redz    = cosmo.redshift(cosmo.snapshotTimes())

    starLife = StellarLifetimes()

    ### Default to Using Input Lookback Times ###

    ### If Masses are Provided, Calculate Corresponding Lookback Times ###
    if( masses is not None ):
        # Make sure input is iterable
        if( not np.iterable(masses) ): masses = [ masses ]*2
        # Convert from [msol] to [grams]
        stellarMasses = np.array(masses)*MSOL

        if( verbose ): 
            print " - - Using target masses %s to find typical ages" % (str(stellarMasses/MSOL))

        times = starLife.lifetime(stellarMasses)/YEAR


    # Make sure input is iterable
    if( not np.iterable(times) ): times = [ times,    times    ]
    elif( len(times) != 2 ):      times = [ times[0], times[0] ]
    # Convert from [yr] to [sec]
    targetLBTimes = np.array(times)*YEAR


    ### Find Snapshots Nearest 
    if( verbose ): print " - - Target Lookback Time = %s [Gyr]" % (str(targetLBTimes/GYR))
    indsOld = np.where( lbTimes >= targetLBTimes[0] )[0]
    indsYng = np.where( lbTimes <= targetLBTimes[1] )[0]
    indsOld = indsOld[-1]
    indsYng = indsYng[0]

    # Do not include latest snapshot (hi)
    hiSnaps = np.arange(indsOld, indsYng  )
    loSnaps = np.arange(indsYng, NUM_SNAPS)

    if( verbose ): 

        hiTimes = np.array(lbTimes[hiSnaps])
        loTimes = np.array(lbTimes[loSnaps])
        hiTimes[np.where(hiTimes < 0.0)] = 0.0
        loTimes[np.where(loTimes < 0.0)] = 0.0
        hiRedz  = redz[hiSnaps]
        loRedz  = redz[loSnaps]

        print " - - Nearest snapshots"
        print " - - - Numbers:  %4d, %4d" % (indsOld, indsYng)
        print " - - - LB Times: %.2f, %.2f [Gyr]" % (lbTimes[indsOld]/GYR, lbTimes[indsYng]/GYR)
        print " - - - Redshift: %.2f, %.2f" % (redz[indsOld], redz[indsYng])

        print " - - - High SFR Snapshots: [%4d, %4d] Number" % (hiSnaps[0], hiSnaps[-1])
        print "                           [%.2f, %.2f] Gyr" % (hiTimes[0]/GYR, hiTimes[-1]/GYR)
        print "                           [%.2f, %.2f] z" % (hiRedz[0], hiRedz[-1])

        print " - - - Low  SFR Snapshots: [%4d, %4d] Number" % (loSnaps[0], loSnaps[-1])
        print "                           [%.2f, %.2f] Gyr" % (loTimes[0]/GYR, loTimes[-1]/GYR)
        print "                           [%.2f, %.2f] z" % (loRedz[0], loRedz[-1])


    return hiSnaps, loSnaps





if __name__ == '__main__': main()
