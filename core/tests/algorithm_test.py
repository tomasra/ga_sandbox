from mock import Mock
import unittest
from core.chromosomes import Chromosome
from core.algorithm import Algorithm
from core.crossovers import Crossover
from core.selections import Selection
from core.solution import Solution, SolutionFactory


class AlgorithmTests(unittest.TestCase):
    def test_one_generation(self):
        """
        here be dragons
        """
        class _FakeIndividual(Mock):
            mutation_count = 0

            def mutate(self, rate):
                _FakeIndividual.mutation_count += 1

            def _calculate_fitness(self):
                # Value shouldn't matter in this test
                return 42

        class _FakeSelection(Mock):
            count = 0

            def run(self, population):
                self.count += 1
                return _FakeIndividual()

        class _FakeCrossover(Mock):
            count = 0

            def run(self, parent1, parent2):
                self.count += 1
                return parent1, parent2

        # Fake genetic operators
        crossover = _FakeCrossover()
        selection = _FakeSelection()

        # Run for one generation
        alg = Algorithm(
            _FakeIndividual,
            crossover,
            selection,
            population_size=4,
            elitism_count=0)
        for _p, _g in alg.run(generations=1):
            pass

        self.assertEquals(selection.count, 4)
        self.assertEquals(crossover.count, 2)
        self.assertEquals(_FakeIndividual.mutation_count, 4)

    @unittest.skip('')
    def test_single_generation(self):
        """
        Algorithm - run single generation.
        """
        crossover = _FakeCrossover()
        selection = _FakeSelection()
        _FakeChromosome.mutation_count = 0

        alg = Algorithm(
            Mock,
            crossover,
            selection,
            population_size=6
        )

        # Check initial chromosomes
        # self.assertEquals(len(alg.population), 6)
        # for chromo in alg.population.chromosomes:
        #     self.assertEquals(chromo.content, "00000000")

        alg.run(generations=1)

        # Check the result and operator call counts
        self.assertEquals(len(alg.population), 6)
        for chromo in alg.population.chromosomes:
            self.assertEquals(chromo.content, "10101010")

        self.assertEquals(selection.call_count, 6)
        self.assertEquals(crossover.call_count, 3)
        self.assertEquals(_FakeChromosome.mutation_count, 6)

    @unittest.skip('')
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
        self.assertEquals(crossover.call_count, 9)
        self.assertEquals(_FakeChromosome.mutation_count, 18)

    @unittest.skip('')
    def test_elitism(self):
        pass


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
