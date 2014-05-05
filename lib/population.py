class Population(object):
    def __init__(self,
                 solution_factory,
                 population_size=None,
                 chromosomes=None):
        self._solution_factory = solution_factory
        if chromosomes:
            self._chromosomes = chromosomes
        elif population_size:
            # Create random chromosomes
            self._population_size = population_size
            self._chromosomes = [
                self._solution_factory.create().initialize_chromosome()
                for i in xrange(0, self._population_size)
            ]
        else:
            raise ValueError(
                "Must supply either population size or initial chromosomes")

        # Calculate and store various properties now
        self._solutions = [
            self._solution_factory.create().decode(chromo)
            for chromo in self._chromosomes
        ]
        self._best_solution = max(self.solutions, key=lambda s: s.fitness)
        self._total_fitness = sum([
            solution.fitness
            for solution in self.solutions
        ])
        self._average_fitness = self.total_fitness / len(self)

    def __len__(self):
        return len(self._chromosomes)

    @property
    def chromosomes(self):
        """
        Encoded chromosome list
        """
        return self._chromosomes

    @property
    def solutions(self):
        """
        Decoded chromosome list - solutions.
        """
        return self._solutions

    @property
    def best_solution(self):
        return self._best_solution

    @property
    def total_fitness(self):
        return self._total_fitness

    @property
    def average_fitness(self):
        return self._average_fitness

    def best_chromosomes(self, count):
        if count > len(self):
            raise ValueError("Count is higher than total number of chromosomes")
        else:
            # Specified number of solutions with highest fitness
            best_solutions = sorted(
                self._solutions,
                key=lambda s: s.fitness,
                reverse=True)[:count]
            best_solutions_indexes = [
                self._solutions.index(s)
                for s in best_solutions
            ]
            # Corresponding chromosomes
            best_chromos = [
                self._chromosomes[idx]
                for idx in best_solutions_indexes
            ]
            return best_chromos
