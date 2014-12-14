from abc import ABCMeta, abstractmethod
from core.parallelizer import parallel_task


"""
Example of subclass:
----------------------------------------------------
class SpecificIndividual(Individual):
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


@parallel_task
# These must be keyword arguments!
# Phenotype is distributed to workers once, when GA is started,
# while different chromosome is passed each time when this function
# is called
def calculate_fitness_parallel(**kwargs):
    phenotype = kwargs['phenotype']
    chromosome = kwargs['chromosome']
    individual = phenotype(chromosome)
    return individual._calculate_fitness()


class Individual(object):
    __metaclass__ = ABCMeta

    def __init__(self, genotype=None, chromosome=None):
        if chromosome is not None:
            # Can be initialized with existing genetic data
            self.chromosome = chromosome
        elif genotype is not None:
            # Or a new chromosome can be created now
            self.chromosome = genotype()
        else:
            self.chromosome = None

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
