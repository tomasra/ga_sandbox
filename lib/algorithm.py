from population import Population


class Algorithm(object):
    def __init__(self,
                 solution_class,
                 crossover_strategy,
                 selection_strategy,
                 population_size=10,
                 mutation_rate=0.01):
        self._solution_class = solution_class
        self._crossover_strategy = crossover_strategy
        self._selection_strategy = selection_strategy
        self._population_size = population_size
        self._mutation_rate = mutation_rate
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
            self._solution_class,
            self._population_size)

    def run(self, generations=1):
        # Multiple generations
        if generations > 1:
            for i in xrange(generations):
                self.run(generations=1)
        # Single generation
        else:
            new_chromos = []
            parent1, parent2 = None, None

            # Selection and crossover
            for i in xrange(len(self._population)):
                # Odd number - pick a chromosome
                if i % 2 == 0:
                    parent1 = self._selection_strategy.run(self._population)
                # Even number - pick another, do crossover, save offspring
                else:
                    parent2 = self._selection_strategy.run(self._population)
                    offspring1, offspring2 = self._crossover_strategy.run(
                        parent1, parent2)
                    new_chromos += [offspring1, offspring2]
                    parent1, parent2 = None, None

            # Need one more chromosome?
            if parent1 and not parent2:
                new_chromos += [parent1]

            # Mutation
            for chromo in new_chromos:
                chromo.mutate(self._mutation_rate)

            # Replace current population
            self._population = Population(
                self._solution_class,
                chromosomes=new_chromos
            )
