import unittest
from lib.chromosomes.binary import BinaryChromosome
from lib.solutions.arithmetic_expression import ArithExpSolution


class ArithExpSolutionTests(unittest.TestCase):
    def test_encoding(self):
        solution = ArithExpSolution(target=None, length=None)
        solution.expression = "6+5*4/2"
        chromo = solution.encode()
        self.assertIsInstance(chromo, BinaryChromosome)
        self.assertEqual(chromo.content, "0110101001011100010011010010")

    def test_decoding(self):
        chromo = BinaryChromosome(content="0110101001011100010011010010")
        solution = ArithExpSolution(target=None, length=None).decode(chromo)
        self.assertEqual(solution.expression, "6+5*4/2")

    def test_fitness(self):
        (sol1, sol2, sol3) = [
            ArithExpSolution(target=10, length=3)
            for _ in xrange(3)
        ]
        sol1.expression = "1+2"
        sol2.expression = "3+4"
        sol3.expression = "5+5"
        self.assertLess(sol1.fitness, sol2.fitness)
        self.assertLess(sol2.fitness, sol3.fitness)

    def test_initialize_chromosome(self):
        chromo = ArithExpSolution(target=None, length=5).initialize_chromosome()
        self.assertIsInstance(chromo, BinaryChromosome)
        self.assertEquals(len(chromo), 5 * 4)

    def test_evaluation(self):
        solution1 = ArithExpSolution(target=None, length=None)
        solution2 = ArithExpSolution(target=None, length=None)
        solution1.expression = "6+5*4/2"
        solution2.expression = "22+?-72"
        self.assertEqual(solution1._evaluate(), 22)
        self.assertEqual(solution2._evaluate(), 9)
