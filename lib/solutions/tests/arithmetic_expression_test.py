import unittest
from lib.chromosomes.binary import BinaryChromosome
from lib.solutions.arithmetic_expression import ArithExpSolution


class ArithExpSolutionTests(unittest.TestCase):
    def test_encoding(self):
        solution = ArithExpSolution()
        solution.expression = "6+5*4/2"
        chromo = solution.encode()
        self.assertIsInstance(chromo, BinaryChromosome)
        self.assertEqual(chromo.content, "0110101001011100010011010010")

    def test_decoding(self):
        chromo = BinaryChromosome(content="0110101001011100010011010010")
        solution = ArithExpSolution().decode(chromo)
        self.assertEqual(solution.expression, "6+5*4/2")

    def test_fitness(self):
        target_backup = ArithExpSolution.target
        length_backup = ArithExpSolution.length
        ArithExpSolution.target = 10
        ArithExpSolution.length = 3

        (sol1, sol2, sol3) = [ArithExpSolution() for _ in xrange(3)]
        sol1.expression = "1+2"
        sol2.expression = "3+4"
        sol3.expression = "5+5"
        self.assertLess(sol1.fitness, sol2.fitness)
        self.assertLess(sol2.fitness, sol3.fitness)

        ArithExpSolution.target = target_backup
        ArithExpSolution.length = length_backup

    def test_initialize_chromosome(self):
        length_backup = ArithExpSolution.length
        ArithExpSolution.length = 5
        chromo = ArithExpSolution().initialize_chromosome()
        self.assertIsInstance(chromo, BinaryChromosome)
        self.assertEquals(len(chromo), 5 * 4)
        ArithExpSolution.length = length_backup

    def test_evaluation(self):
        solution1 = ArithExpSolution()
        solution2 = ArithExpSolution()
        solution1.expression = "6+5*4/2"
        solution2.expression = "22+?-72"
        self.assertEqual(solution1._evaluate(), 22)
        self.assertEqual(solution2._evaluate(), 9)
