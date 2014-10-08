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


class Merger(object):
    '''Object representing a single merger (or creation) event.'''
    #inids = []
    #outids = []
    #ids = []

    def __init__(self, instr):
        args = self.parseMergerLine(instr)                                                          # Extract target quanitites from string

        self.time     = args[0]                                                                     # Simulation time at merger
        self.bhin     = Blackhole(args[1], args[2], args[0])                                                 # Accretor BH
        self.bhout    = Blackhole(args[3], args[4], args[0])                                                 # Accreted BH

        #Merger.inids.append(args[1])
        #Merger.outids.append(args[3])
        #Merger.ids.append(args[1])
        #Merger.ids.append(args[3])

    def __str__(self):
        return "Time: %g  In , Out  =  %s , %s" % (self.time, str(self.bhin), str(self.bhout))



    @staticmethod
    def parseMergerLine(instr):
        '''
        Parse a line from an Illustris blachole_mergers_#.txt file

        The line is formatted (in C) as:
            '%d %g %llu %g %llu %g\n', 
            ThisTask, All.Time, (long long) id,  mass, (long long) P[no].ID, BPP(no).BH_Mass

        return time, accretor_id, accretor_mass, accreted_id, accreted_mass
        '''
        
        args = instr.split()
        return FLT(args[1]), INT(args[2]), FLT(args[3]), INT(args[4]), FLT(args[5])

    '''
    @staticmethod
    def trimIDs():
        Merger.inids  = np.unique( Merger.inids  )
        Merger.outids = np.unique( Merger.outids )
        #Merger.ids = np.unique( Merger.ids )
    '''
