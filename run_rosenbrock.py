#!/venv/bin/python

from lib.algorithm import Algorithm
from lib.crossovers.one_point import OnePointCrossover
from lib.selections.roulette_wheel import RouletteWheelSelection
from lib.solutions.rosenbrock import RosenbrockSolution

crossover = OnePointCrossover(rate=0.2)
selection = RouletteWheelSelection()
alg = Algorithm(
    RosenbrockSolution,
    crossover,
    selection,
    population_size=50,
    mutation_rate=0.01
)

for i in xrange(50):
    alg.run()
    print alg.population.best_solution
    print "Average fitness: " + str(alg.population.average_fitness)
