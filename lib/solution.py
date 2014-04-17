from abc import ABCMeta
from abc import abstractmethod, abstractproperty


class Solution(object):
    __metaclass__ = ABCMeta

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
