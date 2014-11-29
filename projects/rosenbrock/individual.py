import numpy as np
from core.individual import Individual
from core.chromosomes import BinaryChromosome


class RosenbrockSolution(Individual):
    MIN_X, MAX_X = -3.0, 3.0
    MIN_Y, MAX_Y = -3.0, 3.0
    # Number of bits for one variable
    VAR_LENGTH = 10
    STEP_X = (MAX_X - MIN_X) / ((2 ** VAR_LENGTH) - 1)
    STEP_Y = (MAX_Y - MIN_Y) / ((2 ** VAR_LENGTH) - 1)

    print STEP_X

    def _decode(self, chromosome):
        length = self.VAR_LENGTH
        binary_x = chromosome[0:length]
        binary_y = chromosome[length:length * 2]
        int_x = self.binary_to_int(binary_x)
        int_y = self.binary_to_int(binary_y)
        self.x = self.MIN_X + (int_x * self.STEP_X)
        self.y = self.MIN_Y + (int_y * self.STEP_Y)

    def _calculate_fitness(self):
        # Evaluate Rosenbrock's function,
        val = (1 - self.x)**2 + 100 * (self.y - self.x**2)**2
        return 1.0 / (1.0 + val)

    def _initialize_chromosome(self):
        return BinaryChromosome(RosenbrockSolution.VAR_LENGTH * 2)

    def binary_to_int(self, binary):
        # Convert gray coding to int
        i = 0
        for bit in binary:
            i = i * 2 + bit
        # print i
        return i

    def __repr__(self):
        res = "X=" + "{:1.5f}".format(self.x)
        res += ", Y=" + "{:1.5f}".format(self.y)
        return res


class _RosenbrockSolution(Individual):
    MIN_X, MAX_X = -3.0, 3.0
    MIN_Y, MAX_Y = -3.0, 3.0
    VAR_LENGTH = 20

    def _decode(self, chromosome):
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

    def _calculate_fitness(self):
        # Evaluate Rosenbrock's function,
        val = (1 - self.x)**2 + 100 * (self.y - self.x**2)**2
        return 1.0 / (1.0 + val)

    def _initialize_chromosome(self):
        return BinaryChromosome(RosenbrockSolution.VAR_LENGTH * 2)

    def __repr__(self):
        res = "X=" + "{:1.3f}".format(self.x)
        res += ", Y=" + "{:1.3f}".format(self.y)
        return res

    def _real_to_binary(self, value, min_val, max_val):
        scale_top = 2 ** RosenbrockSolution.VAR_LENGTH
        res = scale_top * (value - min_val) / (max_val - min_val)
        return "{0:b}".format(int(res))

    def _binary_to_real(self, value, min_val, max_val):
        if isinstance(value, np.ndarray):
            value = ''.join([str(c) for c in value])
        value = float(int(value, 2))
        scale_top = 2 ** RosenbrockSolution.VAR_LENGTH
        res = (value * (max_val - min_val)) / scale_top + min_val
        return res
