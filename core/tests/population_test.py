import unittest
from core.solution import Solution, SolutionFactory
from core.population import Population


class PopulationTest(unittest.TestCase):
    def test_initialize_with_size(self):
        """
        Population - create new with random chromosomes
        """
        population = Population(
            _FakeSolutionFactory(),
            population_size=2)
        self.assertEquals(len(population.chromosomes), 2)
        self.assertEquals(population.chromosomes[0], "01010101")
        self.assertEquals(population.chromosomes[1], "01010101")

    def test_initialize_with_chromosomes(self):
        """
        Population - create with initial chromosomes
        """
        chromos = ["01010101", "10101010"]
        population = Population(
            _FakeSolutionFactory(),
            chromosomes=chromos)
        self.assertEquals(len(population.chromosomes), 2)
        self.assertEquals(population.chromosomes[0], "01010101")
        self.assertEquals(population.chromosomes[1], "10101010")

    def test_population_solutions(self):
        """
        Population - test decoded chromosomes (solutions)
        """
        chromos = ["01010101", "1000000"]
        population = Population(
            _FakeSolutionFactory(),
            chromosomes=chromos)
        self.assertEquals(len(population.solutions), 2)
        self.assertIsInstance(population.solutions[0], _FakeSolution)
        self.assertIsInstance(population.solutions[1], _FakeSolution)
        self.assertEquals(population.solutions[0].fitness, 4)
        self.assertEquals(population.solutions[1].fitness, 1)

    def test_best_solution(self):
        """
        Population - return best solution
        """
        chromos = ["0000", "0001", "0011"]
        population = Population(
            _FakeSolutionFactory(),
            chromosomes=chromos)
        self.assertEquals(population.best_solution.fitness, 2)

    def test_average_fitness(self):
        """
        Population - average fitness
        """
        chromos = ["0001", "0011", "0111"]
        population = Population(
            _FakeSolutionFactory(),
            chromosomes=chromos)
        self.assertEquals(population.average_fitness, 2)

    def test_total_fitness(self):
        """
        Population - total fitness
        """
        chromos = ["1111", "0111", "1110"]
        population = Population(
            _FakeSolutionFactory(),
            chromosomes=chromos)
        self.assertEquals(population.total_fitness, 10)

    def test_best_chromosomes(self):
        """
        Population - elite chromosomes
        """
        chromos = ["0000", "0111", "0001", "0011"]
        population = Population(
            _FakeSolutionFactory(),
            chromosomes=chromos)
        self.assertSequenceEqual(
            population.best_chromosomes(2),
            ["0111", "0011"])
        self.assertRaises(
            ValueError,
            lambda: population.best_chromosomes(5))


class _FakeSolution(Solution):
    """
    Fake solution class for testing.
    """
    def encode(self):
        return "00001111"

    def decode(self, chromosome):
        # Calculate fitness - number of ones
        self._fitness = len([i for i in chromosome if i == "1"])
        return self

    @property
    def fitness(self):
        return self._fitness

    def initialize_chromosome(self):
        return "01010101"


class _FakeSolutionFactory(SolutionFactory):
    def create(self):
        return _FakeSolution()
