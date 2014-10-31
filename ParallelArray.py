# ==================================================================================================
# ParallelArray.py
# ----------------
# 
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================

import numpy as np


class ParallelArray(object):
    """
    
    """
    
    def __init__(self, num, names, types, keys=None, zero=True):
        self.NUM = len(names)                                                                       # Number of arrays

        # Make sure number of names matches number of types
        if( self.NUM != len(types) ):
            raise RuntimeError("Num names doesn't match num types!")

        # Make sure keys (if provided) match number of names
        if( keys != None ):
            if( len(keys) != self.NUM ):
                raise RuntimeError("Num keys doesn't match num names or types!")

        # Choose how to initialize arrays
        if( zero ): initFunc = np.zeros                                                             # Initialize arrays to zero  (clean)
        else:       initFunc = np.array                                                             # Initialize arrays to empty (unclean)

        # Create dictionary to store keys for each array (for later access)
        self.__keys = {}
        #self.__names = {}

        ### Initialize arrays ###
        for ii in xrange(self.NUM):
            # Add array as attribute
            setattr(self, names[ii], initFunc(num, dtype=types[ii]) )
            # Establish an ordering for different arrays
            self.__keys[names[ii]] = ii

        self.__len = num                                                                            # Set length to initial value



    def keys(self): 
        ''' List of array 'keys' (names) for each array, and their index for each row '''
        return sorted(self.__keys.items(), key=lambda x:x[1] )

    def __len__(self): return self.__len

    def __getitem__(self, key):
        """
        Return target arrays.

        For a particular merger, key should be that index.
        For a particular array, key is appropriate string argument.
            e.g. MERGER_TIME = 'time' for the time array
        """
        
        if( type(key) == int ): key = np.int(key)                                                   # Make sure ints are numpy ints

        # If int, return slice across all arrays
        if( np.issubdtype( type(key), int ) ):
            return [ getattr(self, name)[key] for name,ind in self.keys() ]
        # If str, try to return that attribute
        elif( type(key) == str ): 
            if( hasattr(self, key) ): return getattr(self, key)
            else: raise KeyError("Unrecognized key '%s' !" % (key) )
        # Otherwise, error
        else: 
            raise KeyError("Key must be a string or integer, not a %s!" % (str(type(key))) )


    def __setitem__(self, key, vals):
        """
        Set target array.

        For a particular merger, key should be that index.
        For a particular array, key is appropriate string argument.
            e.g. MERGER_TIME = 'time' for the time array
        """

        # If int, set same element of each array
        if( isInt(key) ):
            # name is the array name, ind is the corresponding index of 'vals' 
            for (name,ind),aval in zip(self.keys(),vals):
                getattr(self,name)[key] = aval                                                      # set element 'key' of array 'name' to aval (at ind)

        # If str, set full array
        elif( type(key) == str ): 
            getattr(self, key)[:] = vals

        # Otherwise, error
        else: 
            raise KeyError("Key must be a string or integer, not a %s!" % (str(type(key))) )

        return


    def __delitem__(self, key):
        """ Delete target index of each array """
        return self.delete(key)


    def delete(self, keys):
        """ Delete target indices of each array """

        # In each array, delete target element
        for name,ind in self.keys():
            setattr(self, name, np.delete(getattr(self, name), key))
            if( ind == 0 ): self.__len = len(getattr(self, name))

        return self.__len

        
    def append(self, vals):
        """ Append the given merger information as a new last element """

        # In each array, delete target element
        for (name,ind),aval in zip(self.keys(),vals):
            setattr(self, name, np.append(getattr(self, name), aval))
            if( ind == 0 ): self.__len = len(getattr(self, name))

        return self.__len



def isInt(val):
    # python int   ==> True
    if( type(val) == int ): return True
    # numpy int    ==> True
    elif( np.issubdtype( type(val), int ) ): return True
    # anyting else ==> False
    else: return False
