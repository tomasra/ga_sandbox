from lib.chromosomes.binary import BinaryChromosome
from lib.solution import Solution, SolutionFactory


class CustomFiltersSolution(Solution):
    SEQUENCE_LENGTH = 5
    KERNEL_SIZE = 15

    def __init__(self, factory):
        # sequence of filters with respective parameters
        # target images and ground truths for fitness evaluation
        pass

    def encode(self):
        # filter sequence to binary chromosome
        pass

    def decode(self, chromosome):
        # binary chromosome to filter sequence
        pass

    @property
    def fitness(self):
        return self._fitness

    def initialize_chromosome(self):
        pass


class FiltersSolutionFactory(SolutionFactory):
    def __init__(self):
        # Target image(s)
        # Target ground truths (volcanoes with radiuses)
        pass

    def create(self):
        pass
