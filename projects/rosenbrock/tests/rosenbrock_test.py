import unittest
from lib.solutions.rosenbrock import RosenbrockSolution
from lib.chromosomes.binary import BinaryChromosome


class RosenbrockSolutionTests(unittest.TestCase):
    def test_encode_decode(self):
        """
        Rosenbrock solution - encoding and decoding
        """
        x, y = 0.25, 1.72
        rs = RosenbrockSolution(x=x, y=y)
        chromo = rs.encode()
        rs.decode(chromo)
        self.assertAlmostEqual(rs.x, x, 5)
        self.assertAlmostEqual(rs.y, y, 5)

    def test_fitness(self):
        """
        Rosenbrock solution - fitness
        """
        rs00 = RosenbrockSolution(x=0, y=0)
        rs01 = RosenbrockSolution(x=0, y=1)
        rs10 = RosenbrockSolution(x=1, y=0)
        rs11 = RosenbrockSolution(x=1, y=1)

        # Fitness values should be in the following order:
        # rs01, rs10, rs00, rs11
        self.assertLess(rs01.fitness, rs10.fitness)
        self.assertLess(rs10.fitness, rs00.fitness)
        self.assertLess(rs00.fitness, rs11.fitness)
        # self.assertEquals(rs00.fitness, -1)
        # self.assertEquals(rs01.fitness, -101)
        # self.assertEquals(rs10.fitness, -100)
        # self.assertEquals(rs11.fitness, 0)

    def test_initialize_chromosome(self):
        """
        Rosenbrock solution - initialize chromosome
        """
        chromo = RosenbrockSolution().initialize_chromosome()
        self.assertIsInstance(chromo, BinaryChromosome)
        self.assertEquals(RosenbrockSolution.VAR_LENGTH * 2, len(chromo))
