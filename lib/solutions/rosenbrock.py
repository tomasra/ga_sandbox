from lib.solution import Solution, SolutionFactory
from lib.chromosomes.binary import BinaryChromosome


class RosenbrockSolution(Solution):
    MIN_X, MAX_X = -3.0, 3.0
    MIN_Y, MAX_Y = -3.0, 3.0
    VAR_LENGTH = 20

    def __init__(self, chromosome=None, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        res = "X=" + "{:1.3f}".format(self.x)
        res += ", Y=" + "{:1.3f}".format(self.y)
        return res

    def encode(self):
        """
        Converts x and y to binary representations
        """
        binary_x = self._real_to_binary(
            self.x,
            RosenbrockSolution.MIN_X,
            RosenbrockSolution.MAX_X)
        binary_y = self._real_to_binary(
            self.y,
            RosenbrockSolution.MIN_Y,
            RosenbrockSolution.MAX_Y)
        return BinaryChromosome(content=binary_x + binary_y)

    def decode(self, chromosome):
        """
        Decodes chromosome to set x and y
        """
        length = RosenbrockSolution.VAR_LENGTH
        binary_x = chromosome[0:length]
        binary_y = chromosome[length:length * 2]
        self.x = self._binary_to_real(
            binary_x,
            RosenbrockSolution.MIN_X,
            RosenbrockSolution.MAX_X)
        self.y = self._binary_to_real(
            binary_y,
            RosenbrockSolution.MIN_Y,
            RosenbrockSolution.MAX_Y)
        return self

    @property
    def fitness(self):
        # Evaluate Rosenbrock's function,
        val = (1 - self.x)**2 + 100 * (self.y - self.x**2)**2
        # return 0.0 - val
        # return 10000 - val
        # return 1.0 / val
        return 1.0 / (1.0 + val)

    def initialize_chromosome(self):
        """
        Return randomized binary chromosome
        """
        return BinaryChromosome(
            length=RosenbrockSolution.VAR_LENGTH * 2)

    def _real_to_binary(self, value, min_val, max_val):
        scale_top = 2 ** RosenbrockSolution.VAR_LENGTH
        res = scale_top * (value - min_val) / (max_val - min_val)
        return "{0:b}".format(int(res))

    def _binary_to_real(self, value, min_val, max_val):
        value = float(int(value, 2))
        scale_top = 2 ** RosenbrockSolution.VAR_LENGTH
        res = (value * (max_val - min_val)) / scale_top + min_val
        return res


class RosenbrockSolutionFactory(SolutionFactory):
    # TODO: fix this to pass MAX_* and other parameters
    # to each solution object
    def create(self):
        return RosenbrockSolution()
