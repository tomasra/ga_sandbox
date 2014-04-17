import unittest
from mock import MagicMock, patch
from lib.selections.roulette_wheel import RouletteWheelSelection


class RouletteWheelSelectionTests(unittest.TestCase):
    def test_selection(self):
        """
        Roulette wheel selection - pick one chromosome
        """
        # Fake population
        chromo1, chromo2, chromo3 = "10101010", "11110000", "00001111"
        (sol1, sol2, sol3) = [MagicMock() for _ in xrange(3)]
        sol1.fitness = 0.1  # probability: 1/15
        sol2.fitness = 1.0  # probability: 10/15
        sol3.fitness = 0.4  # probability: 4/15

        population = MagicMock()
        population.chromosomes = [chromo1, chromo2, chromo3]
        population.solutions = [sol1, sol2, sol3]
        population.total_fitness = sol1.fitness + sol2.fitness + sol3.fitness

        selection = RouletteWheelSelection()

        # pick 1st
        with patch('random.random', return_value=0.01):
            # picked = alg.pick_chromosome()
            picked = selection.run(population)
            self.assertEquals(picked, chromo1)
        # pick 2nd
        with patch('random.random', return_value=0.5):
            picked = selection.run(population)
            self.assertEquals(picked, chromo2)
        # pick 3rd
        with patch('random.random', return_value=0.9):
            picked = selection.run(population)
            self.assertEquals(picked, chromo3)
