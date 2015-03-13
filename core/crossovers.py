# -*- coding: utf-8 -*-
import numpy as np
import copy
from abc import ABCMeta, abstractmethod
from core.chromosomes import RealChromosome


"""
Crossover 'factory'
"""
def get_crossover(params):
    # Crossover type
    if params['crossover'] == 'one_point':
        crossover = OnePointCrossover(params['crossover_rate'])
    elif params['crossover'] == 'two_point':
        crossover = TwoPointCrossover(params['crossover_rate'])
    elif params['crossover'] == 'uniform':
        crossover = UniformCrossover(params['crossover_rate'])
    elif params['crossover'] == 'whole_arithmetic':
        crossover = WholeArithmeticCrossover(
            params['crossover_alpha'],
            params['crossover_rate'])
    else:
        raise ValueError("Unknown crossover type: %s" % params['crossover'])
    return crossover


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


class UniformCrossover(Crossover):
    def __init__(self, *args, **kwargs):
        super(UniformCrossover, self).__init__(*args, **kwargs)

    def _run_specific(self, chromo1, chromo2):
        # Offspring
        o1 = copy.deepcopy(chromo1)
        o2 = copy.deepcopy(chromo2)
        for index in xrange(len(chromo1)):
            # Hardcoded 0.5 probability - might be nice to pass this in
            rnd = self._randomizer.randint(2)
            if rnd == 1:
                # Swap genes
                o1[index], o2[index] = o2[index], o1[index]
        return o1, o2


class CutSpliceCrossover(Crossover):
    def __init__(self, *args, **kwargs):
        super(CutSpliceCrossover, self).__init__(*args, **kwargs)

    def _run_specific(self, chromo1, chromo2):
        # Split first chromosome at random point
        point1 = chromo1.pick_split_point()
        chromo1_1, chromo1_2 = chromo1.split(point1)

        # Split second at another random point
        point2 = chromo2.pick_split_point()
        chromo2_1, chromo2_2 = chromo2.split(point2)

        offspring1 = chromo1_1.concat(chromo2_2)
        offspring2 = chromo1_2.concat(chromo2_1)
        return offspring1, offspring2


class WholeArithmeticCrossover(Crossover):
    """
    For real-coded chromosomes only
    """
    def __init__(self, alpha=0.3, *args, **kwargs):
        self.alpha = alpha
        super(WholeArithmeticCrossover, self).__init__(*args, **kwargs)

    def _run_specific(self, chromo1, chromo2):
        offspring = zip(*[   
            (
                self.alpha * pair[0] + (1.0 - self.alpha) * pair[1],
                self.alpha * pair[1] + (1.0 - self.alpha) * pair[0]
            )
            for pair in zip(chromo1, chromo2)
        ])
        offspring1 = RealChromosome(
            chromo1.length, chromo1.min_val, chromo1.max_val,
            content=np.array(offspring[0]))
        offspring2 = RealChromosome(
            chromo2.length, chromo2.min_val, chromo2.max_val,
            content=np.array(offspring[1]))
        return offspring1, offspring2
