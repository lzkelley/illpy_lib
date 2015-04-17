"""


"""


import sys
import os
from datetime import datetime

import illpy
from illpy import Cosmology
from illpy.Constants import *
from illpy import AuxFuncs as aux

from Constants import *
import Figures
import Profiler


RUN     = 2
SNAP    = 135
VERBOSE = True
PLOT    = True

_VERSION = 0.1

#NUM_RAD_BINS = 40

#RAD_EXTREMA = [ 1.0, 1.0e6 ]                     # [parsec]


NUMS_EPAGS = { 2: np.array([ 61628,  53861,  56400,  62551, 36906,  
                             66419,  65997,  74870,  55591, 56132, 
                             32836,  47564,  15070,  71370, 83554,  
                             36227,  24929,  59653,  74329, 55481 ]) }


NUMS_NULLS = { 2: np.array([ 98645,  42640,  104561, 103399, 101025, 
                             75064,  85549,  84796,  55151,  106001, 
                             56933,  49007,  32557,  63844,  27716,
                             78516,  3026,   72601,  91679,  89625,
                             82511,  3034,   92484,  86035,  104021, 
                             66022,  78603,  104659, 78803,  98268, 
                             115876, 97312,  101598, 94810,  99410,  
                             84564,  98779,  94954,  84278,  84121 ]) }


distConv = DIST_CONV*1000/KPC                                                                       # Convert simulation units to [pc]                  
densConv = DENS_CONV*np.power(PC, 3.0)/MSOL                                                         # Convert simulation units to [msol/pc^3] 



def main(run=RUN, snap=SNAP, loadsave=True, verbose=VERBOSE, plot=PLOT):

    if( verbose ): print "Analyzer.py"
    startMain = datetime.now()

    numsEpags = NUMS_EPAGS[run]
    numsNulls = NUMS_NULLS[run]


    ### Load E+A Subhalo Profiles ###

    if( verbose ): print " - Loading EplusA Profiles"
    start = datetime.now()
    profsEpags = Profiler.loadProfiles(run, snap, numsEpags, loadsave=loadsave, verbose=verbose)
    stop = datetime.now()
    if( verbose ): print " - - Loaded after %s" % (str(stop-start))


    ### Load Null Subhalo Profiles ###

    if( verbose ): print " - Loading EplusA Profiles"
    start = datetime.now()
    profsNulls = Profiler.loadProfiles(run, snap, numsNulls, loadsave=loadsave, verbose=verbose)
    stop = datetime.now()
    if( verbose ): print " - - Loaded after %s" % (str(stop-start))


    ### Plot ###
    Figures.figa06.plotFigA06_EplusA_Profiles(run, profsEpags, profsNulls)


    stopMain = datetime.now()
    if( verbose ): print " - Done after %s" % (str(stopMain-startMain))
    
    return profsEpags, profsNulls

# main()








if __name__ == '__main__': main()
