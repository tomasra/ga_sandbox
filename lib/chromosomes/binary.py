import random
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

    def _get_random(self, length):
        return "{0:b}".format(random.getrandbits(length)).zfill(length)

    def _flip_bit(self, index):
        if self[index] == "0":
            self[index] = "1"
        elif self[index] == "1":
            self[index] = "0"
        return self
