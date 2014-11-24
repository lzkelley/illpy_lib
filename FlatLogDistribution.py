import numpy as np
from numpy import random


class FlatLogDistribution(object):

    """
    Class to sample from a distribution function f(x) which is flat in the log
    of x.  I.e. f(x) ~ 1/x
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

        # Find the normalization (i.e. the integral between extrema)
        self.norm = np.log(self.max/self.min)
        

    def sample(self):
        """
        Draw randomly from the Flat-Log distribution function.
        """
        # Draw from a random uniform distribution
        rand = np.random.random()
        # Inver the CDF to draw a sample
        xx = self.__invert_cdf(rand)
        return xx


    def __invert_cdf(self, qq):
        """
        Invert the Cumulative Distribution Function.

        i.e. For F(x) = CDF[ f(x) = 1/x ], find 'x' s.t. F(x) = `qq`
        """
        xx = self.min*np.exp( self.norm*qq )
        return xx

        
    @staticmethod    
    def flatlog(min, max, num=1):
        """
        Draw samples from a flat log distribution (~1/x) between `min`, `max`.

        Parameters
        ----------
        min : scalar
            minimum value of distribution

        max : scalar
            maximum value of distribution

        num : scalar, optional, default = 1
            Number of samples to draw

        Returns
        -------
        samps : scalar, [array-like : `num` elements]
            Samples from a 1/x distribution function.

        """
        norm = np.log(max/min)
        inv = lambda x: min*np.exp(x*norm)
        
        samps = np.random.random(size=num)
        flats = np.array( map(inv, samps) )
        return flats

