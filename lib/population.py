class Population(object):
    def __init__(self,
                 solution_class,
                 population_size=None,
                 chromosomes=None):
        self._solution_class = solution_class
        if chromosomes:
            self._chromosomes = chromosomes
        elif population_size:
            # Create random chromosomes
            self._population_size = population_size
            self._chromosomes = [
                self._solution_class().initialize_chromosome()
                for i in xrange(0, self._population_size)
            ]
        else:
            raise ValueError(
                "Must supply either population size or initial chromosomes")

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
        # for chromo in self._chromosomes:
            # yield self._solution_class().decode(chromo)
        return [
            self._solution_class().decode(chromo)
            for chromo in self._chromosomes
        ]

    @property
    def best_solution(self):
        """
        Solution with maximum fitness
        """
        return max(self.solutions, key=lambda s: s.fitness)

    @property
    def total_fitness(self):
        return sum([solution.fitness for solution in self.solutions])

    @property
    def average_fitness(self):
        return self.total_fitness / len(self)
