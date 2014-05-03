import random
import numpy as np
from lib.chromosome import Chromosome


class IntegerChromosome(Chromosome):
    def __init__(
            self,
            min_val,
            max_val,
            length=None,
            content=None):
        self.min_val = min_val
        self.max_val = max_val
        super(IntegerChromosome, self).__init__(length, content)

    def mutate(self, rate):
        """
        Replaces integer with another random integer
        from current interval.
        """
        for i in xrange(len(self)):
            do_mutation = random.random() < rate
            if do_mutation:
                new_value = np.random.random_integers(
                    self.min_val,
                    self.max_val,
                    1
                )
                self.content[i] = new_value

    def split(self, point):
        """
        Splits chromsome in two
        """
        part1 = self.content[:point]
        part2 = self.content[point:]
        chromo1 = IntegerChromosome(self.min_val, self.max_val, content=part1)
        chromo2 = IntegerChromosome(self.min_val, self.max_val, content=part2)
        return chromo1, chromo2

    def concat(self, other):
        """
        Concatenates two chromosomes
        """
        new_content = np.concatenate((self.content, other.content))
        return IntegerChromosome(
            self.min_val,
            self.max_val,
            content=new_content
        )

    def _get_random(self, length):
        """
        Sequence of random integers in [min_val, max_val] interval
        """
        return np.random.random_integers(
            self.min_val,
            self.max_val,
            length
        )
