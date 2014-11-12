import random
from abc import ABCMeta, abstractmethod


class Selection(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def run(self, population):
        """
        Returns a single selected individual from population
        """
        pass


class RouletteWheelSelection(Selection):
    def run(self, population):
        """
        Picks one chromosome from the current population
        by "roulette wheel" method
        """
        # Random number between 0 and 1
        picked_value = random.random()
        total_fitness = population.total_fitness
        cumulative_ratio = 0.0
        for i, solution in enumerate(population.solutions):
            # Probability of current chromosome being picked
            chromo_ratio = solution.fitness / total_fitness
            cumulative_ratio_next = cumulative_ratio + chromo_ratio
            if cumulative_ratio <= picked_value <= cumulative_ratio_next:
                return population.chromosomes[i]
            else:
                cumulative_ratio = cumulative_ratio_next
        # Should never reach this
        raise RuntimeError('Roulette wheel did not pick any chromosome.')
