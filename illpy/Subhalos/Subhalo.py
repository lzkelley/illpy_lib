
from datetime import datetime
import numpy as np

import illpy
from illpy.Constants import GET_ILLUSTRIS_OUTPUT_DIR, PARTICLE

import illustris_python as ill

VERBOSE = True

LOAD_PARTICLES = [PARTICLE.GAS, PARTICLE.DM, PARTICLE.STAR, PARTICLE.BH]



def importSubhaloParticles(run, snapNum, subhalo, partTypes=None, verbose=VERBOSE):
    """
    Import particle data for a given Subhalo from the illustris snapshot files.

    The target particle types are the standard, e.g. 0-gas, 1-dm, etc described by the constants in
    ``illpy.Constants.PARTICLE``.  Each particle type has a different set of parameters returned in
    the resulting dictionaries in the ``data`` output.

    Arguments
    ---------
       run       <int>      : Illustris simulation number {1,3}
       snapNum   <int>      : Illustris snapshot number {1,135}
       subhalo   <int>      : Subhalo index for this snapshot
       partTypes <int>([N]) : optional, Target particle types; if `None`, all are loaded
       verbose   <bool>     : optional, print verbose output

    Returns
    -------
       data      <dict>([N]) : dictionary of target particle data
                               If a single ``partType`` is given, a single dictionary is returned.
                               Otherwise a list of dictionaries, one for each ``partType`` is
                               returned in the same order as provided.

    Additional Parameters
    ---------------------
       LOAD_PARTICLES <int>[N] : Default list of particle types to load if ``partType == None``.

    """

    if( verbose ): print " - - Subhalos._importSubhaloParticles()"

    ## Prepare Particle Types to Import
    #  --------------------------------

    # Set particle types to Default
    if( partTypes is None ): partTypes = LOAD_PARTICLES

    # Make sure ``partTypes`` is iterable
    if( not np.iterable(partTypes) ): partTypes = [ partTypes ]

    # Get names of particle types
    partNames = [ PARTICLE.NAMES(ptype) for ptype in partTypes ]

    # Get snapshot file path and filename
    outputPath = GET_ILLUSTRIS_OUTPUT_DIR(run)


    ## Load Snapshot data for target Particles
    #  ---------------------------------------

    if( verbose ): print " - - - Loading snapshot data"
    data = []
    if( verbose ): start_all = datetime.now()
    # Iterate over particle types
    for ptype,pname in zip(partTypes,partNames):
        if( verbose ): start = datetime.now()
        partData = ill.snapshot.loadSubhalo(outputPath, snapNum, subhalo, ptype )
        data.append(partData)
        if( verbose ): stop = datetime.now()
        numParts = partData['count']
        numParams = len(partData.keys())-1
        if( verbose ): 
            print "         %8d %6s, %2d pars, after %s" % \
                (numParts, pname, numParams, str(stop-start))


    if( verbose ): stop_all = datetime.now()
    if( verbose ): print " - - - - All After %s" % (str(stop_all-start_all))

    # If single particle, don't both with list
    if( len(data) == 1 ): data = data[0]

    return data

# importSubhaloParticles()
