#! /home/tomas/.virtualenvs/ga/bin/python
import numpy as np
from core.individual import Individual
from core.algorithm import Algorithm
# from core.selections import RouletteWheelSelection
from core.selections import TournamentSelection
from core.crossovers import OnePointCrossover
from core.chromosomes import BinaryChromosome
from core.parallelizer import Parallelizer
import time


class BitStringSolution(Individual):
    """
        Simplest possible application of GA:
        Evolve a bit string to match predefined target.
    """
    TARGET = np.array([
        int(c)
        for c in "111100000000000000000000000000000000000000000000000000001111"
    ])
    LENGTH = len(TARGET)

    def _decode(self, chromosome):
        self.solution = chromosome

    def _calculate_fitness(self):
        # Count symbol matches between current chromosome and the target
        fitness = 0
        for pair in zip(self.solution, self.TARGET):
            if pair[0] == pair[1]:
                fitness += 1
        # for i in xrange(100000):
        #     i * i
        return float(fitness)

    def _initialize_chromosome(self):
        return BinaryChromosome(self.LENGTH)


with Parallelizer() as parallelizer:
    if parallelizer.master_process:
        start = time.time()

        alg = Algorithm(
            BitStringSolution,
            OnePointCrossover(0.8),
            # RouletteWheelSelection(),
            TournamentSelection(5),
            population_size=50,
            mutation_rate=0.015,
            elitism_count=1,
            parallelizer=parallelizer)

        for population, generation in alg.run():
            best_fitness = int(population.best_individual.fitness)
            print "Generation ", generation, best_fitness
            # Solution found
            if best_fitness == BitStringSolution.LENGTH:
                print population.best_individual.solution
                break

        end = time.time()
        print end - start
