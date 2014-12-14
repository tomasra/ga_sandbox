# -*- coding: utf-8 -*-
import numpy as np
import copy
from abc import ABCMeta, abstractmethod


class Crossover(object):
    """
        Abstract base class for various genetic crossovers
    """
    __metaclass__ = ABCMeta
    # Can be replaced with fake one in unit tests
    _randomizer = np.random.RandomState()

    def __init__(self, rate):
        self._rate = rate

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

        do_crossover = self._randomizer.random_sample() < rate

        # WARNING: shallow copies
        offspring1 = copy.copy(parent1)
        offspring2 = copy.copy(parent2)

        if do_crossover:
            # Update chromosomes of the offspring
            chromo1, chromo2 = self._run_specific(
                parent1.chromosome,
                parent2.chromosome)
        else:
            chromo1 = copy.deepcopy(parent1.chromosome)
            chromo2 = copy.deepcopy(parent2.chromosome)

        offspring1.chromosome = chromo1
        offspring2.chromosome = chromo2
        return offspring1, offspring2

    @abstractmethod
    def _run_specific(self, chromo1, chromo2):
        pass


class OnePointCrossover(Crossover):
    def __init__(self, *args, **kwargs):
        super(OnePointCrossover, self).__init__(*args, **kwargs)

    def _run_specific(self, chromo1, chromo2):
        point = chromo1.pick_split_point()
        # Split each parent into two parts at the same point
        chromo1_part1, chromo1_part2 = chromo1.split(point)
        chromo2_part1, chromo2_part2 = chromo2.split(point)
        offspring1 = chromo1_part1.concat(chromo2_part2)
        offspring2 = chromo2_part1.concat(chromo1_part2)
        return offspring1, offspring2


class TwoPointCrossover(Crossover):
    def __init__(self, *args, **kwargs):
        super(TwoPointCrossover, self).__init__(*args, **kwargs)

    def _run_specific(self, chromo1, chromo2):
        point1 = chromo1.pick_split_point()
        point2 = chromo1.pick_split_point()
        points = sorted([point1, point2])

        # Split each parent into three parts:
        # chromo1: | 1111 | 1111 | 1111 |
        # chromo2: | 2222 | 2222 | 2222 |
        chromo1_1, chromo1_2, chromo1_3 = chromo1.split(points)
        chromo2_1, chromo2_2, chromo2_3 = chromo2.split(points)

        # Now recombine them:
        # offspring1: | 1111 | 2222 | 1111 |
        # offspring2: | 2222 | 1111 | 2222 |
        offspring1 = chromo1_1.concat(chromo2_2).concat(chromo1_3)
        offspring2 = chromo2_1.concat(chromo1_2).concat(chromo2_3)
        return offspring1, offspring2


class UniformCrossover(object):
    # TODO
    pass


class TreeCrossover(object):
    # TODO
    pass
