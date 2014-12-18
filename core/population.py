
class Population(object):
    def __init__(self, phenotype, size=0, parallelizer=None):
        # Class of concrete individual
        self.phenotype = phenotype

        # Initialize with new random individuals
        # Distribute chromosome creation/fitness calculation to workers
        self.parallelizer = parallelizer

        self._individuals = [
            phenotype()
            for _ in xrange(size)
        ]
        self._total_fitness = None
        self._average_fitness = None
        self._best_individuals = []

    def __len__(self):
        return self._individuals.__len__()

    def __iter__(self):
        return self._individuals.__iter__()

    def __getitem__(self, key):
        return self._individuals.__getitem__(key)

    def __setitem__(self, index, value):
        return self._individuals.__setitem__(index, value)

    def __iadd__(self, individual):
        # Add either single individual or a list
        if isinstance(individual, list):
            self._individuals += individual
        else:
            self._individuals.append(individual)
        return self

    def calculate_fitness(self, truncate_if_above=None):
        """
        Calculate individual fitness values in parallel
        """
        # Distribute: individual index as task ID
        if self.parallelizer is not None:
            for task_id, individual in enumerate(self):
                if individual.fitness is None:
                    # This is quite inefficient in this case:
                    # self.parallelizer.start_task(
                    #     task_id, lambda: individual._calculate_fitness())

                    # So use this:
                    self.parallelizer.start_prepared_task(
                        task_id, 'calculate_fitness_parallel',
                        # Arbitrary number of kwargs
                        chromosome=individual.chromosome)

            # Collect and assign calculated fitness values for each individual
            for task_id, task_result in self.parallelizer.finished_tasks():
                self[task_id].fitness = task_result
        else:
            # Calculate fitness in an ordinary way
            for individual in self:
                if individual.fitness is None:
                    individual.fitness = individual._calculate_fitness()

        self._best_individuals = sorted(
            self,
            key=lambda individual: individual.fitness,
            reverse=True)

        # HACK: if actual population size is larger than expected,
        # remove least fit individuals
        if truncate_if_above is not None:
            to_remove = len(self) - truncate_if_above
            if to_remove > 0:
                for individual in self.worst_individuals(to_remove):
                    self._individuals.remove(individual)
                    self._best_individuals.remove(individual)

        # Calculate these properties once
        self._total_fitness = sum([individual.fitness for individual in self])
        self._average_fitness = self._total_fitness / len(self)

    @property
    def total_fitness(self):
        return self._total_fitness

    @property
    def average_fitness(self):
        return self._average_fitness

    @property
    def best_individual(self):
        """
        Individual with the highest fitness
        """
        try:
            return self.best_individuals(1)[0]
        except IndexError:
            return None

    def best_individuals(self, count=None):
        """
        Specified number of solutions with highest fitness
        """
        if count is None:
            return self._best_individuals
        else:
            return self._best_individuals[:count]

    @property
    def worst_individual(self):
        """
        Individual with lowest fitness
        """
        try:
            # return self.best_individuals()[-1]
            return self.worst_individuals(1)[0]
        except IndexError:
            return None

    def worst_individuals(self, count=None):
        """
        Specified number of individuals with lowest fitness
        """
        if count is None:
            return self._best_individuals[::-1]
        else:
            return self._best_individuals[::-1][:count]
