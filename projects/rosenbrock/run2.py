from core.algorithm import Algorithm
from core.crossovers import OnePointCrossover
from core.selections import RouletteWheelSelection, TournamentSelection
from individual import RosenbrockSolution

alg = Algorithm(
    RosenbrockSolution,
    OnePointCrossover(rate=0.5),
    RouletteWheelSelection(),
    # TournamentSelection(size=5),
    population_size=50,
    mutation_rate=0.015,
    elitism_count=5)

for population, generation in alg.run():
    s = "Average fitness: " + str(population.average_fitness)
    s += ", best solution: " + str(population.best_individual)
    print s
    if population.best_individual.fitness >= 0.99:
        break
