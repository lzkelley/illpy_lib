"""
Find subhalos based on target properties.


"""


import sys
import os
from datetime import datetime

import readtreeHDF5
import readsubfHDF5

import illpy
from illpy import Cosmology
from illpy.Constants import *
from illpy import AuxFuncs as aux
from Constants import *


import Figures
from StellarLifetimes import StellarLifetimes

import arepo

RUN     = 3
SFR_CUT = 0.9
VERBOSE = True
PLOT    = True

SELECT_NUM  = 20
COMPARE_NUM = 20


TARGET_LOOKBACK_TIMES = [ 1.5e9, 1.5e8 ]                                                            # [years]
TARGET_STELLAR_MASSES = [ 2.0, 3.0 ]                                                                # [msol]
TARGET_SFR_AGE        = 8.0e8                                                                       # [years]
BH_MASS_CUT           = 1.0e6                                                                       # [msol]


MIN_STAR_PARTICLES = 10
MIN_GAS_PARTICLES  = 20
MIN_BH_PARTICLES   = 1

MIN_NUM_SUBHALOS   = 10



def main(run=RUN, loadsave=True, verbose=VERBOSE, plot=PLOT):

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
    snapFirst = np.min(np.concatenate([hiSnaps, loSnaps]))
    snapLast  = np.max(np.concatenate([hiSnaps, loSnaps]))
    if( verbose ): print " - - Earliest snapshot = %d; Latest = %d" % (snapFirst, snapLast)

    ### Load target snapshot subhalo catalog ###
    if( verbose ): print " - Loading Subhalo Catalog for Snapshot %d" % (snapFirst)
    catFirst = loadSubhaloCatalog(run, snapFirst, keys=SUBFIND_PARAMETERS, verbose=verbose)
    numSubhalos = len(catFirst[SH_SFR])
    if( verbose ): print " - - Loaded catalog with %d subhalos" % (numSubhalos)


    # Find valid subhalos
    if( verbose ): print " - Finding valid subhalos"
    inds = filterSubhalos_minParticles(catFirst, verbose=verbose)
    if( verbose ): print " - - Found %d/%d valid subhalos" % (len(inds), numSubhalos)

    # Plot initial subhalos
    if( plot ): 
        cat_sfr = catFirst[SH_SFR][inds]
        cat_mass_type = catFirst[SH_MASS_TYPE][inds]
        cat_mass_bh = catFirst[SH_BH_MASS][inds]
        Figures.figa01.plotFigA01_Subfind_SFR( run, cat_sfr, cat_mass_type, cat_mass_bh )


    if( verbose ): print " - Loading Subhalo Branches for %d subhalos" % (len(inds))
    branches = loadSubhaloBranches(run, inds, tree, snapFirst, loadsave=loadsave, verbose=verbose)


    ### Select Subhalos Based on Star Formation Rate ###
    #sfrInds, sfrs = sfrSubhalos(catFirst, inds, cut=cut, verbose=verbose)


    ### Get Weights for Quality of EplusA Galaxies ###
    weightsSFR = weight_sfrChange(branches, hiSnaps, loSnaps, cosmo=cosmo, verbose=verbose)


    if( plot ): Figures.figa02.plotFigA02_Branches_SFR(run, branches, hiSnaps, loSnaps, weightsSFR)

    # Get the indices of the SFR-selected EplusA 'epa' Subhalos
    inds_epa = selectTop(weightsSFR, num=SELECT_NUM, verbose=verbose)
    # Get the indices of the other 'oth' Subhalos
    inds_oth = list(set(range(len(weightsSFR))).difference(inds_epa))

    weightsEPA = weightsSFR[inds_epa]

    ### Get Subfind ID numbers for Subhalos ###
    
    # first snapshot
    old_epa = np.array(branches[SL_SUBFIND_ID][inds_epa, 0])
    old_oth = np.array(branches[SL_SUBFIND_ID][inds_oth, 0])

    # final snapshot
    new_epa = np.array(branches[SL_SUBFIND_ID][inds_epa,-1])
    new_oth = np.array(branches[SL_SUBFIND_ID][inds_oth,-1])


    ### Load Final snapshot subhalo catalog ###
    if( verbose ): print " - Loading Subhalo Catalog for Snapshot %d" % (snapFirst)
    catLast = loadSubhaloCatalog(run, snapLast, keys=SUBFIND_PARAMETERS, verbose=verbose)
    numSubhalos = len(catLast[SH_SFR])
    if( verbose ): print " - - Loaded catalog with %d subhalos" % (numSubhalos)

    ### Plot EplusA ###
    if( plot ): 
        # Plot EplusA Galaxies at Redshift z = 0.0
        Figures.figa03.plotFigA03_EplusA_Selected(run, catLast, new_epa, new_oth, weightsEPA)
        # Plot EplusA Galaxies Change in properties over Redshift
        Figures.figa04.plotFigA04_EplusA_Evolution(run, catFirst, catLast, old_epa, old_oth, 
                                                   new_epa, new_oth, weightsEPA)



    ### Load EplusA Galaxies' Particles from Redshift 0.0 ###
    if( verbose ): print " - Loading EplusA Subhalo Particle data from Snapshot %d" % (snapLast)
    epas = loadSubhaloParticles(run, new_epa, snapLast, verbose=verbose)


    ### Select Some Other Non-EplusA 'Null' Subhalos ###
    if( verbose ): print " - Selecting Non-EplusA 'null' comparison Halos"
    new_null = np.random.choice(new_oth, size=COMPARE_NUM, replace=False)
    nulls = loadSubhaloParticles(run, new_null, snapLast, verbose=verbose)



    '''
    if( plot ):
        for ii,epaID in enumerate(new_epa):
            Figures.figa05.plotFigA05_Subhalo(run, epaID, epas[ii])
    '''



    return run, snapFirst, snapLast, catFirst, catLast, inds, subhaloInds, tree, branches, snaps, hiSnaps, loSnaps, weightsSFR, weightsEPA, old_epa, old_oth, new_epa, new_oth, new_null



def loadSubhaloParticles(run, snapNum, subhaloInds, loadsave=True, verbose=VERBOSE):

    if( verbose ): print " - - SubhaloFinder.loadSubhaloParticles()"

    if( verbose ): print " - - - Loading %d subhalos' particle data" % (len(subhaloInds))

    groupCat = None
    
    subhalos = []
    for ii,shind in enumerate(subhaloInds):
        fileName = SUBHALO_PARTICLES_FILENAMES(run, snapNum, shind)
        if( verbose ): print " - - - - %d : Subhalo %d - '%s'" % (ii, shind, fileName)

        if( loadsave ):
            if( os.path.exists(fileName) ):
                if( verbose ): print " - - - - - Loading from previous save"
                subhaloData = aux.npzToDict(fileName)
            else:
                print "``loadsave`` file '%s' does not exist!" % (fileName)
                loadsave = False


        if( not loadsave ):
            if( verbose ): print " - - - - - Reloading EplusA Particles from snapshot"
            subhaloData, groupCat = _getSubhaloParticles(run, snapNum, shind, groupCat=groupCat, verbose=verbose)
            aux.saveDictNPZ(subhaloData, fileName, verbose=True)


        subhalos.append(subhaloData)

    # } ii


    subhalos = np.array(subhalos)
    return subhalos



def _getSubhaloParticles(run, snapNum, subhaloInd, groupCat=None, verbose=VERBOSE):

    if( verbose ): print " - - SubhaloFinder._getSubhaloParticles()"

    # Get snapshot file path and filename
    snapshotPath = ILLUSTRIS_OUTPUT_SNAPSHOT_FIRST_FILENAME(run, snapNum)

    ### If Group Catalog is not Provided, load it ###
    if( groupCat is None ):

        # Get group catalog file path and filename
        groupPath = ILLUSTRIS_OUTPUT_GROUP_FIRST_FILENAME(run, snapNum)

        # Load subfind catalog
        if( verbose ): print " - - - Loading subfind catalog from '%s'" % (groupPath)
        groupCat = arepo.Subfind(groupPath, combineFiles=True)



    ### Load Snapshot Data ###

    # Create filter for target subhalo
    filter = [ arepo.filter.Halo(groupCat, subhalo=subhaloInd) ]
    # Load snapshot
    if( verbose ): print " - - - Loading snapshot data"
    data = arepo.Snapshot(snapshotPath, filter=filter, fields=SNAPSHOT_PROPERTIES,
                          combineFiles=True, verbose=False )


    ### Convert Snapshot Data into Dictionary ###

    dataDict = {}
    for snapKey in SNAPSHOT_PROPERTIES:
        dataDict[snapKey] = getattr(data, snapKey)

    dataDict[SUBHALO_ID]       = subhaloInd
    dataDict[SUBHALO_RUN]      = run
    dataDict[SUBHALO_SNAPSHOT] = snapNum
    dataDict[SUBHALO_CREATED]  = datetime.now().ctime()

    return dataDict, groupCat



def selectTop(args, num=10, verbose=VERBOSE):
    
    if( verbose ): print " - - SubhaloFinder.selectTop()"

    # Make sure requested size is less than full array length
    assert num <= len(args), "``num`` must be less than length of array!"
    # Sort args from lowest to highest
    sort = np.argsort(args)
    # Reverse to sort from highest to lowest
    sort = sort[::-1]
    # select the top/first ``num`` elements
    tops = sort[:num]

    return tops

# selectTop()




def weight_sfrChange(branches, his, los, cosmo=None, verbose=VERBOSE):

    if( verbose ): print " - - SubhaloFinder.weight_sfrChange()"

    if( cosmo is None ): cosmo = Cosmology()

    snaps = np.concatenate([his,los])
    first = np.min(snaps)

    sfr = branches[SH_SFR]
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

# } weight_sfrChange()



def loadSubhaloBranches(run, inds, tree, target, loadsave=True, verbose=VERBOSE):

    if( verbose ): print " - - SubhaloFinder.loadSubhaloBranches()"

    saveFile = SUBHALO_BRANCHES_FILENAMES(run)

    if( loadsave ):
        if( verbose ): print " - - - Loading branches from save '%s'" % (saveFile)
        if( not os.path.exists(saveFile) ):
            print "``loadsave`` file '%s' does not exist!" % (saveFile)
            loadsave = False
        else:
            branches = illpy.AuxFuncs.npzToDict(saveFile)
            #snaps = branches[BRANCH_SNAPS]
            #subhaloInds = branches[BRANCH_INDS]


    if( not loadsave ):
        if( verbose ): print " - - Reloading subhalo branches from merger tree"
        #branches, snaps, subhaloInds = getSubhaloBranches(run, inds, tree, snapFirst, verbose=verbose)
        branches = _getSubhaloBranches(run, inds, tree, target, verbose=verbose)
        illpy.AuxFuncs.saveDictNPZ(branches, saveFile, verbose=verbose)


    return branches

# loadSubhaloBranches()




def _getSubhaloBranches(run, inds, tree, target, ngas=MIN_GAS_PARTICLES, nstar=MIN_STAR_PARTICLES, 
                        nbh=MIN_BH_PARTICLES, verbose=VERBOSE):

    if( verbose ): print " - - SubhaloFinder.getSubhaloBranches()"
    numSubhalos = len(inds)
    if( numSubhalos < 100 ): interval = 1
    else:                    interval = np.int( np.floor(numSubhalos/200.0) )


    #assert numSubhalos > MIN_NUM_SUBHALOS, "There are only %d Subhalos!!!" % (numSubhalos)

    # Get snapshot numbers included in the branches
    branchSnaps = np.arange(target, NUM_SNAPS)
    branchLens = NUM_SNAPS - target
    if( verbose ): print " - - - Branch Snaps = %d [%d,%d]" % (branchLens, branchSnaps[0], branchSnaps[-1])


    # Make sure branches are long enough to analyze
    assert branchLens > 2, "``branchLens`` < 2!!"

    # Also get snapshot numbers 
    useKeys = np.concatenate([SUBFIND_PARAMETERS, SUBLINK_PARAMETERS]).tolist()
    useTypes = np.concatenate([SUBFIND_PARAMETER_TYPES, SUBLINK_PARAMETER_TYPES]).tolist()

    ### Build Dictionary To Store Branches Data ###

    # Get sample branch to determine shapes of parameters
    desc = tree.get_future_branch(target, inds[0], keysel=useKeys)

    # Initialize dictionary with zeros arrays for each parameter
    branches = {}
    keyStr = ""
    count = 0
    for key,typ in zip(useKeys, useTypes):
        val = getattr(desc, key)
        shp = np.shape(val)

        if( len(shp) == 1 ): branches[key] = np.zeros( [numSubhalos, branchLens],         dtype=typ )
        else:                branches[key] = np.zeros( [numSubhalos, branchLens, shp[1]], dtype=typ )

        if( count%5 == 0 ): keyStr += " - - - "
        keyStr += "'%s', " % (key)
        count += 1

    # } key


    if( verbose ): 
        print " - - Retrieving branch parameters for :"
        print keyStr

    
    ### Find Branches for Subhalos and Store Parameters ###
    badSFR = -1

    numMissing = 0
    numNaked = 0
    numMassless = 0
    numZeroSFR = 0
    numCutBH = 0
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

        # Make sure subhalo has mass in particles in all future snapshots
        massTypes = getattr(desc, SH_MASS_TYPE)
        if( any(massTypes[:, PARTICLE_TYPE_GAS ] <= 0.0 ) or 
            any(massTypes[:, PARTICLE_TYPE_STAR] <= 0.0 ) or
            any(massTypes[:, PARTICLE_TYPE_BH  ] <= 0.0 )   ):
            numMassless += 1
            badInds.append(ii)
            continue

        # Make sure SFR is never uniformly Zero
        sfr = getattr(desc, SH_SFR)
        if( any(sfr <= 0.0) ):
            numZeroSFR += 1
            badInds.append(ii)
            if( badSFR < 0 ): badSFR = subh
            continue

        # Make sure BH is above cutoff mass at all snapshots (really just the first matters)
        if( any(getattr(desc, SH_BH_MASS)*MASS_CONV < BH_MASS_CUT) ):
            numCutBH += 1
            badInds.append(ii)
            continue

        # Sort by snapshot number
        sort = np.argsort(nums)

        # Store Parameters
        for key in useKeys:
            param = getattr(desc, key)
            branches[key][ii] = param[sort]                                                         # In CHRONOLOGICAL order

    # } ii


    ### Remove Mergers ###
    numMergers = 0
    # Get subhalo ID numbers for last snapshot
    endInds = branches[SL_SUBFIND_ID][:,-1]
    # Remove IDs which appear more than once
    for ii, ind in enumerate(endInds):
        # Only search beyond test entry, so that last case of duplicate entry remains
        found = np.where( endInds[ii:] == ind )[0]
        if( len(found) > 1 ): 
            badInds.append(ii)
            numMergers += 1


    ### Cleanup bad Subhalos (no branches) ###
    goodInds = np.delete(inds, badInds)
    for key in useKeys:
        branches[key] = np.delete(branches[key], badInds, axis=0)

        
    ### Add Meta-Data ###
    branches[BRANCH_RUN]     = run
    branches[BRANCH_INDS]    = goodInds
    branches[BRANCH_CREATED] = datetime.now().ctime()
    branches[BRANCH_SNAPS]   = branchSnaps
                
    if( verbose ): 
        print " - - - - Retrieved %d subhalo branches" % (len(branches[SH_SFR]))
        print " - - - - - Missing  : %d" % (numMissing)
        print " - - - - - Lacking  : %d" % (numNaked)
        out = " - - - - - Zero SFR : %d" % (numZeroSFR)
        if( numZeroSFR > 0 ): out += "; e.g. Subhalo %d" % (badSFR)
        print out
        print " - - - - - Mergers  : %d" % (numMergers)
        print " - - - - - Small BH : %d  (Below %.2e [Msol])" % (numCutBH, BH_MASS_CUT)
        print " - - - - - Massless : %d" % (numMassless)


    # return branches, branchSnaps, goodInds
    return branches

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
