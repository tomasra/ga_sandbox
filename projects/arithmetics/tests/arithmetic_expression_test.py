import unittest
from core.chromosomes import BinaryChromosome
from projects.arithmetics.solution import ArithExpSolution


class ArithExpSolutionTests(unittest.TestCase):
    def test_encoding(self):
        """
        Arithmetic expression solution - encoding
        """
        solution = ArithExpSolution(target=None, length=None)
        solution.expression = "6+5*4/2"
        chromo = solution.encode()
        chromo_str = "".join([str(c) for c in chromo.content])
        self.assertIsInstance(chromo, BinaryChromosome)
        self.assertEqual(chromo_str, "0110101001011100010011010010")

    def test_decoding(self):
        """
        Arithmetic expression solution - decoding
        """
        chromo = BinaryChromosome(content="0110101001011100010011010010")
        solution = ArithExpSolution(target=None, length=None).decode(chromo)
        self.assertEqual(solution.expression, "6+5*4/2")

    def test_fitness(self):
        """
        Arithmetic expression solution - fitness
        """
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
        """
        Arithmetic expression solution - chromosome initialization
        """
        chromo = ArithExpSolution(target=None, length=5).initialize_chromosome()
        self.assertIsInstance(chromo, BinaryChromosome)
        self.assertEquals(len(chromo), 5 * 4)

    def test_evaluation(self):
        """
        Arithmetic expression solution - evaluation
        """
        solution1 = ArithExpSolution(target=None, length=None)
        solution2 = ArithExpSolution(target=None, length=None)
        solution1.expression = "6+5*4/2"
        solution2.expression = "22+?-72"
        self.assertEqual(solution1._evaluate(), 22)
        self.assertEqual(solution2._evaluate(), 9)

    def test_layout_errors(self):
        """
        Arithmetic expression solution - layout errors
        """
        solution1 = ArithExpSolution(target=None, length=None)
        solution2 = ArithExpSolution(target=None, length=None)
        solution3 = ArithExpSolution(target=None, length=None)
        solution1.expression = "1+2*3/4"
        solution2.expression = "?8+9-?"
        solution3.expression = "?8+9-1"
        self.assertEquals(solution1._layout_errors(), 0)
        # if expression does not end with a digit
        # then the previous operator should be treated as an error
        self.assertEquals(solution2._layout_errors(), 3)
        self.assertEquals(solution3._layout_errors(), 1)

    def test_same_evalution_different_layouts(self):
        """
        Arithmetic expression solutions - same evaluation with different layouts
        """
        solution1 = ArithExpSolution(target=10, length=5)
        solution2 = ArithExpSolution(target=10, length=5)
        solution1.expression = "1+2*3"  # evaluates to 9
        solution2.expression = "3*3?9"  # evaluates to 9 too
        self.assertGreater(solution1.fitness, solution2.fitness)
