from abc import ABCMeta
from abc import abstractmethod, abstractproperty


class Solution(object):
    __metaclass__ = ABCMeta

    def __init__(self, chromosome=None):
        """
        Optional decoding from chromosome during initialization
        """
        if chromosome:
            self.decode(chromosome)

    @abstractmethod
    def encode(self):
        """
        Encodes this solution as chromosome
        """
        pass

    @abstractmethod
    def decode(self, chromosome):
        """
        Decodes solution from given chromosome
        """
        pass

    @abstractproperty
    def fitness(self):
        """
        Solution fitness - some non-negative number
        """
        pass

    @abstractmethod
    def initialize_chromosome(self):
        """
        Creates new random chromosome instance of required length
        """
        pass
