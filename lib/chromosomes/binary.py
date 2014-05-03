import random
import numpy as np
from lib.chromosome import Chromosome


class BinaryChromosome(Chromosome):
    def mutate(self, rate):
        """
        Tries to flip each bit with a given probability
        """
        for i in xrange(len(self)):
            do_mutation = random.random() < rate
            if do_mutation:
                self._flip_bit(i)

    def split(self, point):
        """
        Splits content (numpy array) in two
        """
        part1 = self.content[:point]
        part2 = self.content[point:]
        chromo1 = BinaryChromosome(content=part1)
        chromo2 = BinaryChromosome(content=part2)
        return chromo1, chromo2

    def concat(self, other):
        """
        Concatenates two chromosomes
        """
        new_content = np.concatenate((
            self.content,
            other.content
        ))
        return BinaryChromosome(content=new_content)

    def _get_random(self, length):
        return np.random.randint(2, size=length)

    def _flip_bit(self, index):
        if self[index] == 0:
            self[index] = 1
        elif self[index] == 1:
            self[index] = 0
        return self
