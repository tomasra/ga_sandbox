# -*- coding: utf-8 -*-
from core.population import Population


class Algorithm(object):
    def __init__(self,
                 phenotype,
                 crossover,
                 selection,
                 population_size=10,
                 mutation_rate=0.01,
                 elitism_count=0,
                 parallelizer=None):
        # Classes
        self.phenotype = phenotype

        # Parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count

        # GA operator instances
        self._crossover = crossover
        self._selection = selection

        # Etc
        self._parallelizer = parallelizer
        self._population = None

    @property
    def population(self):
        return self._population

    def _next_population(self):
        # Start with an empty population
        new_population = Population(
            self.phenotype,
            size=0,
            parallelizer=self._parallelizer)

        # Pick best individuals from previous population if necessary
        new_population += self.population.best_individuals(
            self.elitism_count)

        # Form the new population
        while len(new_population) < self.population_size:
            # Selection
            individual1 = self._selection.run(self.population)
            individual2 = self._selection.run(self.population)

            # Crossover
            # Individuals are copied, regardless of whether
            # crossover actually occurs.
            offspring1, offspring2 = self._crossover.run(
                individual1, individual2)

            # Mutation
            offspring1.mutate(self.mutation_rate)
            offspring2.mutate(self.mutation_rate)

            new_population += [offspring1, offspring2]

        new_population.calculate_fitness()
        return new_population

    def run(self, generations=None):
        """
        Runs genetic algorithm for a number of iterations,
        starting with a randomly created initial population.
        If iteration count is not specified, algorithm would run
        until terminated explicitly.
        Yields current population AND generation number.
        """
        if generations is not None and generations < 1:
            raise ValueError(
                "Generation count must be positive, non-zero integer")

        # Initial random population for generation-0:
        self._population = Population(
            self.phenotype,
            self.population_size,
            parallelizer=self._parallelizer)
        self._population.calculate_fitness()

        # Run specified amount of iterations
        if generations:
            for generation in xrange(1, generations + 1):
                self._population = self._next_population()
                yield self.population, generation

        # Run indefintely
        else:
            generation = 0
            while True:
                self._population = self._next_population()
                generation += 1
                yield self.population, generation
