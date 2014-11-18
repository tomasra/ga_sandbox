class Population(object):
    def __init__(self, phenotype, size=0):
        # Class of concrete individual
        self.phenotype = phenotype

        # Initialize with new random individuals
        self._individuals = [
            phenotype()
            for _ in xrange(size)
        ]

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

    @property
    def total_fitness(self):
        return sum([
            individual.fitness
            for individual in self
        ])

    @property
    def average_fitness(self):
        return self.total_fitness / len(self)

    @property
    def best_individual(self):
        try:
            return self.best_individuals(1)[0]
        except:
            return None

    def best_individuals(self, count=None):
        """
        Specified number of solutions with highest fitness
        """
        if not isinstance(count, int):
            # Return all individuals, sorted by fitness in descending order
            count = len(self)
        return sorted(
            self,
            key=lambda individual: individual.fitness,
            reverse=True)[:count]
