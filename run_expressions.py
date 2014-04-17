#!/venv/bin/python

from lib.algorithm import Algorithm
from lib.crossovers.one_point import OnePointCrossover
from lib.selections.roulette_wheel import RouletteWheelSelection
from lib.solutions.arithmetic_expression import ArithExpSolutionFactory

alg = Algorithm(
    ArithExpSolutionFactory(target=30, length=3),
    OnePointCrossover(rate=0.4),
    RouletteWheelSelection(),
    population_size=50,
    mutation_rate=0.01
)

for i in xrange(30):
    alg.run()
    s = "Average fitness: " + str(alg.population.average_fitness)
    s += ", best solution: " + str(alg.population.best_solution)
    print s
