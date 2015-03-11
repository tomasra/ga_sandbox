import numpy as np
from abc import ABCMeta, abstractmethod


"""
Selection 'factory'
"""
def get_selection(params):
    if params['selection'] == 'roulette':
        selection = RouletteWheelSelection()
    elif params['selection'] == 'tournament':
        selection = TournamentSelection(params['tournament_size'])
        # output['parameters']['tournament_size'] = params.tournament_size
    elif params['selection'] == 'rank':
        selection = RankSelection()
    else:
        raise ValueError("Unknown selection type: %s" % params['selection'])
    return selection


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


class RankSelection(Selection):
    """
    Linear Rank selection.
    """
    def run(self, population):
        n = len(population)
        rank_sum = (n * (n - 1)) / 2
        alpha = self._randomizer.random_integers(1, rank_sum)
        cumulative_rank = 0
        for index, individual in enumerate(population.best_individuals()):
            # Individual rank, higher means better fitness
            rank = n - index
            cumulative_rank += rank
            if alpha <= cumulative_rank:
                return individual

        # Should never reach this
        raise RuntimeError('Rank selection did not pick any chromosome.')


class TournamentSelection(Selection):
    """
    Randomly picks several (specified by size parameter) population individuals
    and selects the best one of them
    """
    def __init__(self, size):
        self.tournament_size = size

    def run(self, population):
        tournament_population = []

        # Pick several individuals randomly
        for _ in xrange(self.tournament_size):
            random_index = self._randomizer.random_integers(
                0, len(population) - 1)
            tournament_population.append(population[random_index])

        # And the winner is...
        best_individual = max(
            tournament_population, key=lambda a: a.fitness)
        return best_individual
