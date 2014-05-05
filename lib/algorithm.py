from population import Population


class Algorithm(object):
    def __init__(self,
                 solution_factory,
                 crossover_strategy,
                 selection_strategy,
                 population_size=10,
                 mutation_rate=0.01,
                 elitism_count=4):
        self._solution_factory = solution_factory
        self._crossover_strategy = crossover_strategy
        self._selection_strategy = selection_strategy
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._elitism_count = elitism_count
        # Start with new random population
        self.initialize_population()

    @property
    def population(self):
        return self._population

    def initialize_population(self):
        """
        Creates new population with initial (random) chromosomes
        """
        self._population = Population(
            self._solution_factory,
            self._population_size)

    def run(self, generations=1):
        # Multiple generations
        if generations > 1:
            for i in xrange(generations):
                self.run(generations=1)
        # Single generation
        else:
            new_chromos = []

            if self._elitism_count:
                # Take specified number of best chromosomes
                new_chromos += self._population.best_chromosomes(
                    self._elitism_count)

            while len(new_chromos) < len(self._population):
                chromo1 = self._selection_strategy.run(self.population)
                chromo2 = self._selection_strategy.run(self.population)
                chromo1, chromo2 = self._crossover_strategy.run(
                    chromo1, chromo2)
                chromo1.mutate(self._mutation_rate)
                chromo2.mutate(self._mutation_rate)
                new_chromos += [chromo1, chromo2]

            # Replace current population
            self._population = Population(
                self._solution_factory,
                chromosomes=new_chromos
            )
