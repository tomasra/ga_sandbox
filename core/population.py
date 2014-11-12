class Population(object):
    def __init__(self,
                 solution_factory,
                 population_size=None,
                 chromosomes=[],
                 solutions=[]):
        self._solution_factory = solution_factory
        if population_size:
            # Initialize with random chromosomes
            self._population_size = population_size
            self._chromosomes = [
                self._solution_factory.create().initialize_chromosome()
                for i in xrange(0, self._population_size)
            ]
            self._solutions = [
                self._solution_factory.create().decode(chromo)
                for chromo in self._chromosomes
            ]
        elif not chromosomes and not solutions:
            raise ValueError((
                "Must supply either population size ",
                "or initial chromosomes/solutions"
            ))

        if chromosomes:
            # Initialize with raw chromosomes and decode to make solutions
            self._chromosomes = chromosomes
            self._solutions = [
                self._solution_factory.create().decode(chromo)
                for chromo in self._chromosomes
            ]

        if solutions:
            # Add pre-decoded solutions
            # Works good with elitism, because the elite solutions already have
            # their fitness computed, so no need to recreate them from
            # chromosomes
            self._solutions += solutions
            self._chromosomes += [
                solution.encode()
                for solution in solutions
            ]

        # Calculate and store various properties now
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

    def best_solutions(self, count):
        # Specified number of solutions with highest fitness
        return sorted(
            self._solutions,
            key=lambda s: s.fitness,
            reverse=True)[:count]

    def best_chromosomes(self, count):
        if count > len(self):
            raise ValueError("Count is higher than total number of chromosomes")
        else:
            return [
                # Chromosome by index of each best solution
                self._chromosomes[self._solutions.index(s)]
                for s in self._best_solutions(count)
            ]
