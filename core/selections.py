import numpy as np
from abc import ABCMeta, abstractmethod
from core.population import Population


class Selection(object):
    __metaclass__ = ABCMeta
    _randomizer = np.random.RandomState()

    @abstractmethod
    def run(self, population):
        """
        Returns a single selected individual from population
        """
        pass


class RouletteWheelSelection(Selection):
    """
    Picks one chromosome from the current population
    by "roulette wheel" method.
    """
    def run(self, population):
        # Random number between 0 and 1
        picked_value = self._randomizer.random_sample()
        total_fitness = population.total_fitness
        cumulative_ratio = 0.0
        for index, individual in enumerate(population.best_individuals()):
            # Probability of current chromosome being picked
            individual_ratio = individual.fitness / total_fitness
            cumulative_ratio_next = cumulative_ratio + individual_ratio
            if cumulative_ratio <= picked_value <= cumulative_ratio_next:
                return individual
            else:
                cumulative_ratio = cumulative_ratio_next
        # Should never reach this
        raise RuntimeError('Roulette wheel did not pick any chromosome.')


class TournamentSelection(Selection):
    def __init__(self, size):
        self.tournament_size = size

    def run(self, population):
        # Create new special population
        tournament_population = Population(
            population.phenotype, 0)

        # Pick several individuals randomly
        for _ in xrange(self.tournament_size):
            random_index = self._randomizer.random_integers(
                0, len(population) - 1)
            tournament_population += population[random_index]

        # And the winner is...
        return tournament_population.best_individual
