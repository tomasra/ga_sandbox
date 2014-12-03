import unittest
from mock import Mock
from core.solution import Solution, SolutionFactory
from core.population import Population


class _FakeIndividual(object):
    def __init__(self, fitness):
        self.fitness = fitness

    def _calculate_fitness(self):
        return self.fitness


class PopulationTest(unittest.TestCase):
    def test_initialize_with_size(self):
        """
        Population - create new with random chromosomes
        """
        population = Population(Mock, size=2)
        self.assertEquals(len(population), 2)
        self.assertEquals(population.phenotype, Mock)

    def test_initialize_with_chromosomes(self):
        """
        Population - create and add individuals
        """
        individual1, individual2 = Mock(), Mock()

        population = Population(Mock)
        population += individual1
        population += individual2
        self.assertEquals(len(population), 2)
        self.assertEquals(population[0], individual1)
        self.assertEquals(population[1], individual2)

    def test_best_solution(self):
        """
        Population - return best solution
        """
        population = Population(Mock)
        population += [
            _FakeIndividual(0.1),
            _FakeIndividual(0.5)
        ]
        population.calculate_fitness()
        self.assertEquals(population.best_individual.fitness, 0.5)

    def test_average_fitness(self):
        """
        Population - average fitness
        """
        population = Population(Mock)
        population += [
            _FakeIndividual(0.1),
            _FakeIndividual(0.5)
        ]
        population.calculate_fitness()
        self.assertEquals(population.average_fitness, 0.3)

    def test_total_fitness(self):
        """
        Population - total fitness
        """
        population = Population(Mock)
        population += [
            _FakeIndividual(0.1),
            _FakeIndividual(0.5)
        ]
        population.calculate_fitness()
        self.assertEquals(population.total_fitness, 0.6)

    def test_best_chromosomes(self):
        """
        Population - elite chromosomes
        """
        population = Population(Mock)
        population += [
            _FakeIndividual(0.5),
            _FakeIndividual(0.1),
            _FakeIndividual(0.9),
        ]
        population.calculate_fitness()
        self.assertSequenceEqual(
            population.best_individuals(2),
            [population[2], population[0]])
