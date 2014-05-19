#!venv/bin/python

from lib.algorithm import Algorithm
from lib.crossovers.one_point import OnePointCrossover
from lib.selections.roulette_wheel import RouletteWheelSelection
from lib.solutions.arithmetic_expression import ArithExpSolutionFactory

alg = Algorithm(
    ArithExpSolutionFactory(target=100, length=13),
    OnePointCrossover(rate=0.7),
    RouletteWheelSelection(),
    population_size=50,
    mutation_rate=0.005
)

for i in xrange(15):
    alg.run()
    s = "Average fitness: " + "{:1.3f}".format(
        alg.population.average_fitness)
    s += ", Top fitness: " + "{:1.3f}".format(
        alg.population.best_solution.fitness)
    s += ", best solution: " + str(
        alg.population.best_solution)
    print s
