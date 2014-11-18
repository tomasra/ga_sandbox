import unittest
from mock import Mock, MagicMock, patch
from core.selections import RouletteWheelSelection, TournamentSelection


class _FakePopulation(list):
    phenotype = Mock

    def best_individuals(self):
        return self

    @property
    def total_fitness(self):
        return sum([chromo.fitness for chromo in self])


class RouletteWheelSelectionTests(unittest.TestCase):
    def test_selection(self):
        """
        Roulette wheel selection - pick one chromosome
        """
        population = _FakePopulation()
        population += [
            Mock(fitness=0.1),  # probability: 1/15
            Mock(fitness=1.0),  # probability: 10/15
            Mock(fitness=0.4),  # probability: 4/15
        ]
        selection = RouletteWheelSelection()

        # Pick 1st
        selection._randomizer = Mock(
            random_sample=Mock(return_value=0.01))
        self.assertEquals(selection.run(population), population[0])

        # Pick 2nd
        selection._randomizer = Mock(
            random_sample=Mock(return_value=0.5))
        self.assertEquals(selection.run(population), population[1])

        # Pick 3rd
        selection._randomizer = Mock(
            random_sample=Mock(return_value=0.9))
        self.assertEquals(selection.run(population), population[2])


class TournamentSelectionTests(unittest.TestCase):
    def test_selection(self):
        """
        Tournament selection - run
        """
        # Fake population with 4 chromosomes
        population = _FakePopulation()
        population += [
            Mock(fitness=0.1),
            Mock(fitness=0.2),
            Mock(fitness=0.3),
            Mock(fitness=0.4),
        ]

        selection = TournamentSelection(size=2)
        # Let's say randomizer would pick 1st and 3rd chromos
        fake_rand = Mock()
        fake_rand.random_integers.side_effect = [0, 2]
        selection._randomizer = fake_rand

        # Selection should return the third chromosome
        self.assertEqual(
            selection.run(population).fitness, 0.3)
