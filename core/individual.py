from abc import ABCMeta, abstractmethod


"""
Example of subclass:
----------------------------------------------------
class SpecificIndividual(Individual):
    genotype = BinaryChromosome(length=10)

    def __init__(self, *args, **kwargs):
        super(SpecificIndividual, self).__init__(*args, **kwargs)

    def _decode(self, chromosome):
        # Mapping from genotype to phenotype goes here
        # Nothing to return here
        raise NotImplementedError

    def _calculate_fitness(self):
        # Fitness calculation goes here
        # Value has to be returned
        raise NotImplementedError

"""


class Individual(object):
    __metaclass__ = ABCMeta

    def __init__(self, chromosome=None):
        self._fitness = None
        # Can be initialized with existing genetic data
        if chromosome is not None:
            self.chromosome = chromosome
        else:
            # Subclassed individual type must have genotype set
            self.chromosome = type(self).genotype()

    @property
    def chromosome(self):
        return self._chromosome

    @chromosome.setter
    def chromosome(self, value):
        self._chromosome = value
        # Reset fitness
        self._fitness = None

        # Refresh individual traits on chromosome update
        if self.chromosome is not None:
            self._decode(self.chromosome)

    def mutate(self, *args, **kwargs):
        """
        Chromosomes are considered immutable,
        so replace existing one with mutated.
        """
        self.chromosome = self.chromosome.mutate(*args, **kwargs)

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        self._fitness = value

    @abstractmethod
    def _decode(self, chromosome):
        """
        Translate genotype to phenotype.
        """
        pass

    @abstractmethod
    def _calculate_fitness(self):
        pass
