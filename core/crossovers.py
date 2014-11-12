# -*- coding: utf-8 -*-
#!/usr/bin.python

import random
from abc import ABCMeta, abstractmethod


class Crossover(object):
    """
        Abstract base class for various genetic crossovers
    """

    __metaclass__ = ABCMeta

    def __init__(self, rate):
        self._rate = rate

    @property
    def rate(self):
        return self._rate

    def run(self, parent1, parent2, rate=None):
        """
        Does crossover on two chromosome.
        """
        # Chance can be overriden:
        if not rate:
            rate = self.rate

        do_crossover = random.random() < rate
        if do_crossover:
            return self._run_specific(parent1, parent2)
        else:
            # No changes
            return parent1, parent2

    @abstractmethod
    def _run_specific(self, parent1, parent2):
        pass


class OnePointCrossover(Crossover):
    def _run_specific(self, parent1, parent2):
        cross_index = random.randint(0, len(parent1) - 1)
        parent1_part1, parent1_part2 = parent1.split(cross_index)
        parent2_part1, parent2_part2 = parent2.split(cross_index)
        offspring1 = parent1_part1.concat(parent2_part2)
        offspring2 = parent2_part1.concat(parent1_part2)
        return offspring1, offspring2


class TwoPointCrossover(object):
    pass


class UniformCrossover(object):
    pass
