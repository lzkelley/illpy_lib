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

#import LyzeMergers as lm
from ObjBlackhole import Blackhole


INT        = np.int64
FLT        = np.float64


class Merger(object):
    '''Object representing a single merger (or creation) event.'''

    '''
    def __init__(self, instr):
        args          = self.parseMergerLine(instr)                                                 # Extract target quanitites from string

        self.time     = args[0]                                                                     # Simulation time at merger
        self.bhin     = Blackhole(args[3], args[4], args[0])                                        # Accreted BH
        self.bhout    = Blackhole(args[1], args[2], args[0])                                        # Accretor BH
     '''

    def __init__(self, time, bhin, bhout):
        self.time  = time
        self.bhin  = bhin
        self.bhout = bhout


    def __str__(self):
        return "Time: %g  In , Out  =  %s , %s" % (self.time, str(self.bhin), str(self.bhout))


    @classmethod
    def fromMergerFileLine(cls, mline):
        '''Initialize from line of Merger File'''
        time, inid, inma, outid, outma = cls.parseMergerLine(mline)
        bhin  = Blackhole(inid, inma, time)
        bhout = Blackhole(outid, outma, time)
        return cls(time, bhin, bhout)


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

