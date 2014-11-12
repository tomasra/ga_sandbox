#!/venv/bin/python

from core.algorithm import Algorithm
from core.crossovers import OnePointCrossover
from core.selections import RouletteWheelSelection
from solution import RosenbrockSolutionFactory

crossover = OnePointCrossover(rate=0.7)
selection = RouletteWheelSelection()
alg = Algorithm(
    RosenbrockSolutionFactory(),
    crossover,
    selection,
    population_size=50,
    mutation_rate=0.005
)

for i in xrange(50):
    alg.run()
    s = "Average fitness: " + str(alg.population.average_fitness)
    s += ", best solution: " + str(alg.population.best_solution)
    print s
