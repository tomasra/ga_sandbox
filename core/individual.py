from abc import ABCMeta, abstractmethod


class Individual(object):
    __metaclass__ = ABCMeta

    def __init__(self, chromosome=None):
        # Can be initialized with existing genetic data
        if chromosome is not None:
            self.chromosome = chromosome
        else:
            self.chromosome = self._initialize_chromosome()

    @property
    def chromosome(self):
        return self._chromosome

    @chromosome.setter
    def chromosome(self, value):
        self._chromosome = value
        # Refresh individual traits on chromosome update
        if self.chromosome is not None:
            self._decode(self.chromosome)
            self._fitness = self._calculate_fitness()
        else:
            self._fitness = None

    def mutate(self, *args, **kwargs):
        """
        Chromosomes are considered immutable,
        so replace existing one with mutated.
        """
        self.chromosome = self.chromosome.mutate(*args, **kwargs)

    @property
    def fitness(self):
        """
        Returns individual fitness value.
        Only needs to be recalculated when individual traits change,
        and they should only change when the chromosome changes.
        See chromosome setter.
        """
        return self._fitness

    @abstractmethod
    def _decode(self, chromosome):
        """
        Translate genotype to phenotype.
        """
        pass

    @abstractmethod
    def _calculate_fitness(self):
        pass

    @abstractmethod
    def _initialize_chromosome(self):
        pass
