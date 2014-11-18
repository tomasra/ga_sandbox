import unittest
from mock import Mock, patch
from core.crossovers import Crossover, OnePointCrossover, TwoPointCrossover
from core.chromosomes import BinaryChromosome


class CrossoverTest(unittest.TestCase):
    class _SpecificCrossover(Crossover):
        def _run_specific(self, parent1, parent2):
            # Crossover happened!
            raise Warning

    def test_crossover_does_happen(self):
        """
        Crossover - happens
        """
        crossover = self._SpecificCrossover(0.5)
        crossover._randomizer = Mock()
        crossover._randomizer.random_sample.return_value = 0.1
        individual1 = Mock(chromosome=Mock())
        individual2 = Mock(chromosome=Mock())
        self.assertRaises(
            Warning,
            lambda: crossover.run(individual1, individual2))

    def test_crossover_does_not_happen(self):
        """
        Crossover - does not happen
        """
        crossover = self._SpecificCrossover(0.5)
        crossover._randomizer = Mock()
        crossover._randomizer.random_sample.return_value = 0.9
        parent1 = Mock(fitness=0.14)
        parent2 = Mock(fitness=0.42)
        offspring1, offspring2 = crossover.run(parent1, parent2)
        self.assertEquals(parent1.fitness, offspring1.fitness)
        self.assertEquals(parent2.fitness, offspring2.fitness)


class OnePointCrossoverTest(unittest.TestCase):
    def test_one_point_crossover(self):
        """
        Crossover - one point
        """
        class _FakeChromosome(str):
            def pick_split_point(self):
                return 5

            def concat(self, other):
                return _FakeChromosome(self + other)

            def split(self, split_point):
                return (
                    _FakeChromosome(self[:split_point]),
                    _FakeChromosome(self[split_point:]),
                )

        chromo1 = _FakeChromosome("00000000")
        chromo2 = _FakeChromosome("11111111")

        crossover = OnePointCrossover(0.5)
        new1, new2 = crossover._run_specific(chromo1, chromo2)
        self.assertItemsEqual(
            [new1, new2],
            ["00000111", "11111000"])


class TwoPointCrossoverTest(unittest.TestCase):
    def test_two_point_crossover(self):
        """
        Crossover - two point
        """
        class _FakeChromosome(str):
            def pick_split_point(self):
                pass

            def concat(self, other):
                return _FakeChromosome(self + other)

            def split(self, split_points):
                return (
                    _FakeChromosome(self[0:3]),
                    _FakeChromosome(self[3:6]),
                    _FakeChromosome(self[6:]),
                )

        # Note that split points are put backwards (list.pop())
        chromo1 = _FakeChromosome("000000000")
        chromo2 = _FakeChromosome("111111111")

        crossover = TwoPointCrossover(0.5)
        new1, new2 = crossover._run_specific(chromo1, chromo2)
        self.assertItemsEqual(
            [new1, new2],
            ["000111000", "111000111"])
