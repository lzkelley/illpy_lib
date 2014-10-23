# ==================================================================================================
# ObjMerger.py
# ------------
# 
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================

import numpy as np
import readsnapHDF5 as rs
from glob import glob
from datetime import datetime

from BlackholeMergerSettings import *



class Mergers(object):
    ''' Class to store many mergers with components as simple arrays.

    Usage:
      mergers = ManyMergers(NUM_MERGERS)
      mergers[i] = [ TIME, OUT_ID, OUT_MASS, IN_ID, IN_MASS ]

    This class is just a wrapper for 5 numpy arrays storing the time,
    IDs and masses of both blackholes ('in' the accreted, and 'out'
    the accretor).  Each constituent array can be accessed as if this
    object was a dict, e.g. 
      times = mergers[MERGER_TIME]
    or the values for a single merger can be accessed, e.g.
      merger100 = mergers[100]

    ManyMergers supports deletion by element, or series of elements,
    e.g.
      del mergers[100]
      mergers.delete([100,200,221])
    
    Individual mergers can be added using either the ManyMergers.add()
    method, or by accessing the last+1 memeber, e.g. if len(mergers) = N
      mergers[N+1] = [ TIME, IN_ID, ... ]
    will also work.

    '''

    MERGER_TIME     = 'time'
    MERGER_OUT_ID   = 'out_id'
    MERGER_OUT_MASS = 'out_mass'
    MERGER_IN_ID    = 'in_id'
    MERGER_IN_MASS  = 'in_mass'

    
    def __init__(self, nums):
        ''' Initialize object with empty arrays for 'num' entries '''
        self.time     = np.zeros(nums, dtype=FLT )
        self.out_id   = np.zeros(nums, dtype=UINT)
        self.out_mass = np.zeros(nums, dtype=FLT )
        self.in_id    = np.zeros(nums, dtype=UINT)
        self.in_mass  = np.zeros(nums, dtype=FLT )
        self.__len    = nums


    def __len__(self): return self.__len


    def __getitem__(self, key):
        '''
        Return target arrays.

        For a particular merger, key should be that index.
        For a particular array, key is appropriate string argument.
            e.g. MERGER_TIME = 'time' for the time array
        '''

        if(   type(key) == int ): 
            return [ self.time[key], self.in_id[key], self.in_mass[key], 
                     self.out_id[key], self.out_mass[key] ]
        elif( key == MERGER_TIME     ): return self.time
        elif( key == MERGER_OUT_ID   ): return self.out_id
        elif( key == MERGER_OUT_MASS ): return self.out_mass
        elif( key == MERGER_IN_ID    ): return self.in_id
        elif( key == MERGER_IN_MASS  ): return self.in_mass
        else: raise KeyError("Unrecozgnized key '%s'!" % (str(key)) )


    def __setitem__(self, key, vals):
        '''
        Set target array.

        For a particular merger, key should be that index.
        For a particular array, key is appropriate string argument.
            e.g. MERGER_TIME = 'time' for the time array
        '''
        if(   type(key) == int ): 
            if( key == self.__len ): self.add(vals)
            else: 
                self.time[key]     = vals[0]
                self.out_id[key]   = vals[1]
                self.out_mass[key] = vals[2]
                self.in_id[key]    = vals[3]
                self.in_mass[key]  = vals[4]
        elif( key == MERGER_TIME     ): 
            self.time = vals
        elif( key == MERGER_OUT_ID   ): 
            self.out_id = vals
        elif( key == MERGER_OUT_MASS ): 
            self.out_mass = vals
        elif( key == MERGER_IN_ID    ): 
            self.in_id = vals
        elif( key == MERGER_IN_MASS  ): 
            self.in_mass = vals
        else: raise KeyError("Unrecozgnized key '%s'!" % (str(key)) )


    def __delitem__(self, key):
        ''' Delete the merger array at the target index '''

        self.time     = np.delete(self.time,     key)
        self.out_id   = np.delete(self.out_id,   key)
        self.out_mass = np.delete(self.out_mass, key)
        self.in_id    = np.delete(self.in_id,    key)
        self.in_mass  = np.delete(self.in_mass,  key)
        self.__len    = len(self.time)


    def delete(self, keys):
        ''' Delete the merger(s) at 'keys' - an integer (list) '''

        self.time     = np.delete(self.time,     keys)
        self.out_id   = np.delete(self.out_id,   keys)
        self.out_mass = np.delete(self.out_mass, keys)
        self.in_id    = np.delete(self.in_id,    keys)
        self.in_mass  = np.delete(self.in_mass,  keys)
        self.__len    = len(self.time)

        
    def add(self, vals):
        ''' Append the given merger information as a new last element '''

        self.time     = np.append(self.time,     vals[0])
        self.out_id   = np.append(self.out_id,   vals[3])
        self.out_mass = np.append(self.out_mass, vals[4])
        self.in_id    = np.append(self.in_id,    vals[1])
        self.in_mass  = np.append(self.in_mass,  vals[2])
        self.__len    = len(self.time)

    
