# -*- coding: utf-8 -*-
import numpy as np
import copy
from abc import ABCMeta, abstractmethod


class Crossover(object):
    """
        Abstract base class for various genetic crossovers
    """
    __metaclass__ = ABCMeta

    def __init__(self, rate):
        self._rate = rate
        # Can be replaced with fake one in unit tests
        self._randomizer = np.random.RandomState()

    @property
    def rate(self):
        return self._rate

    def run(self, parent1, parent2, rate=None):
        """
        Does crossover of two individuals
        """
        # Chance can be overriden:
        if not rate:
            rate = self.rate

        # Class of individual
        phenotype = type(parent1)

        do_crossover = self._randomizer.random_sample() < rate
        if do_crossover:
            # Update chromosomes of the offspring
            chromo1, chromo2 = self._run_specific(
                parent1.chromosome,
                parent2.chromosome)
        else:
            chromo1 = copy.deepcopy(parent1.chromosome)
            chromo2 = copy.deepcopy(parent2.chromosome)

        offspring1 = phenotype(chromosome=chromo1)
        offspring2 = phenotype(chromosome=chromo2)
        return offspring1, offspring2

    @abstractmethod
    def _run_specific(self, parent1, parent2):
        pass


class OnePointCrossover(Crossover):
    def __init__(self, *args, **kwargs):
        super(OnePointCrossover, self).__init__(*args, **kwargs)

    def _run_specific(self, parent1, parent2):
        point = parent1.pick_split_point()
        # Split each parent into two parts at the same point
        parent1_part1, parent1_part2 = parent1.split(point)
        parent2_part1, parent2_part2 = parent2.split(point)
        offspring1 = parent1_part1.concat(parent2_part2)
        offspring2 = parent2_part1.concat(parent1_part2)
        return offspring1, offspring2


class TwoPointCrossover(Crossover):
    def __init__(self, *args, **kwargs):
        super(TwoPointCrossover, self).__init__(*args, **kwargs)

    def _run_specific(self, parent1, parent2):
        point1 = parent1.pick_split_point()
        point2 = parent1.pick_split_point()
        points = sorted([point1, point2])

        # Split each parent into three parts:
        # parent1: | 1111 | 1111 | 1111 |
        # parent2: | 2222 | 2222 | 2222 |
        parent1_1, parent1_2, parent1_3 = parent1.split(points)
        parent2_1, parent2_2, parent2_3 = parent2.split(points)

        # Now recombine them:
        # offspring1: | 1111 | 2222 | 1111 |
        # offspring2: | 2222 | 1111 | 2222 |
        offspring1 = parent1_1.concat(parent2_2).concat(parent1_3)
        offspring2 = parent2_1.concat(parent1_2).concat(parent2_3)
        return offspring1, offspring2


class UniformCrossover(object):
    # TODO
    pass


class TreeCrossover(object):
    # TODO
    pass
