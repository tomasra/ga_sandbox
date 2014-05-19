import unittest
from lib.chromosome import Chromosome
from lib.algorithm import Algorithm
from lib.crossover import Crossover
from lib.selection import Selection
from lib.solution import Solution, SolutionFactory


class AlgorithmTests(unittest.TestCase):
    def test_single_generation(self):
        """
        Algorithm - run single generation.
        """
        crossover = _FakeCrossover()
        selection = _FakeSelection()
        _FakeChromosome.mutation_count = 0

        alg = Algorithm(
            _FakeSolutionFactory(),
            crossover,
            selection,
            population_size=6
        )

        # Check initial chromosomes
        self.assertEquals(len(alg.population), 6)
        for chromo in alg.population.chromosomes:
            self.assertEquals(chromo.content, "00000000")

        alg.run()

        # Check the result and operator call counts
        self.assertEquals(len(alg.population), 6)
        for chromo in alg.population.chromosomes:
            self.assertEquals(chromo.content, "10101010")

        self.assertEquals(selection.call_count, 6)
        self.assertEquals(crossover.call_count, 3)
        self.assertEquals(_FakeChromosome.mutation_count, 5)

    def test_multiple_generations(self):
        """
        Algorithm - run multiple generations
        """
        crossover = _FakeCrossover()
        selection = _FakeSelection()
        _FakeChromosome.mutation_count = 0

        alg = Algorithm(
            _FakeSolutionFactory(),
            crossover,
            selection,
            population_size=6
        )

        # Check initial chromosomes
        self.assertEquals(len(alg.population), 6)
        for chromo in alg.population.chromosomes:
            self.assertEquals(chromo.content, "00000000")

        alg.run(generations=3)

        # Check the result
        self.assertEquals(len(alg.population), 6)
        for chromo in alg.population.chromosomes:
            self.assertEquals(chromo.content, "10101010")

        self.assertEquals(selection.call_count, 18)
        self.assertEquals(crossover.call_count, 6)
        self.assertEquals(_FakeChromosome.mutation_count, 18)


class _FakeChromosome(Chromosome):
    """
    Fake chromosome with mutation count tracking
    """
    mutation_count = 0

    def mutate(self, rate):
        _FakeChromosome.mutation_count += 1

    def split(self, point):
        return self.content[:point], self.content[point:]

    def concat(self, other):
        return self.content + other.content

    def _get_random(self, length):
        return "00000000"


class _FakeSolution(Solution):
    """
    Fake solution object
    """
    def encode(self):
        return _FakeChromosome(content="00000000")

    def decode(self, chromosome):
        return self

    @property
    def fitness(self):
        return 1.0

    def initialize_chromosome(self):
        return _FakeChromosome(content="00000000")


class _FakeSolutionFactory(SolutionFactory):
    """
    Fake solution factory
    """
    def create(self):
        return _FakeSolution()


class _FakeSelection(Selection):
    """
    Fake selection strategy
    """
    def __init__(self):
        self.call_count = 0

    def run(self, population):
        self.call_count += 1
        # Always return the same chromosome
        return _FakeChromosome(content="10101010")


class _FakeCrossover(Crossover):
    """
    Fake crossover strategy
    """
    def __init__(self):
        # Guaranteed crossover
        Crossover.__init__(self, rate=1.0)
        self.call_count = 0

    def _run_specific(self, parent1, parent2):
        self.call_count += 1
        # Just return the parents
        return parent1, parent2
