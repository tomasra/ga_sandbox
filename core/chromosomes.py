# -*- coding: utf-8 -*-
import numpy as np
import copy
from abc import ABCMeta, abstractmethod


class Chromosome(object):
    """
        Abstract base class for all types of chromosomes.
        Instances are considered immutable and all operations
        (mutate, split, concat) return new chromosomes.
    """
    __metaclass__ = ABCMeta

    # Set randomizer to specific instance if necessary
    # This helps to unit test the code dealing with random data.
    _randomizer = np.random.RandomState()

    def __init__(self, content):
        self._content = content

    def mutate(self, rate):
        """
        Mutate each chromosome gene with specified probability
        and return a new chromosome
        """
        new_chromosome = self[:]
        for index, gene in enumerate(new_chromosome):
            # Draw a random number from [0, 1)
            do_mutation = self._randomizer.random_sample() < rate
            if do_mutation:
                new_chromosome._content[index] = self._mutate_gene(gene, index)

        return new_chromosome

    def split(self, split_point):
        """
        Splits chromosome in two at the specified index.
        Used in crossover.
        """
        # Multiple points?
        if isinstance(split_point, list):
            if len(split_point) == 2:
                chromo1 = self[:split_point[0]]
                chromo2 = self[split_point[0]:split_point[1]]
                chromo3 = self[split_point[1]:]
                return chromo1, chromo2, chromo3
            else:
                # More complex case
                raise NotImplementedError
        else:
            chromo1 = self[:split_point]
            chromo2 = self[split_point:]
            return chromo1, chromo2

    def concat(self, other):
        """
        Concatenates this chromosome with another.
        Used in crossover.
        """
        new_chromosome = copy.deepcopy(self)
        new_chromosome._content = self._concat_genes(other)
        return new_chromosome

    @abstractmethod
    def _mutate_gene(self, gene, index):
        """
        Mutate single gene - leave this to implementations
        """
        pass

    @abstractmethod
    def _concat_genes(self, gene, index):
        """
        Concatenate genetic content
        """
        pass

    def pick_split_point(self):
        """
        Random point over the whole chromosome length
        """
        index = self._randomizer.random_integers(
            0, len(self) - 1, 1)
        return index

    # Redirect to content
    def __eq__(self, other):
        return self._content == other._content

    def __len__(self):
        return self._content.__len__()

    def __iter__(self):
        return self._content.__iter__()

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Return a chromosome
            new_chromosome = copy.deepcopy(self)
            new_chromosome._content = self._content.__getitem__(key)
            return new_chromosome
        else:
            # Return one gene
            return self._content.__getitem__(key)


class IntegerChromosome(Chromosome):
    def __init__(self, min_val, max_val, initial_length):
        # Numpy array of random numbers from specified interval
        super(IntegerChromosome, self).__init__(
            self._randomizer.random_integers(
                min_val, max_val, initial_length))
        self.min_val = min_val
        self.max_val = max_val

    def _mutate_gene(self, gene, index):
        """
        Random integer from specified interval
        """
        return self._randomizer.random_integers(
            self.min_val, self.max_val, 1)

    def _concat_genes(self, other):
        """
        Concatenate both numpy arrays
        """
        return np.concatenate((self[:], other[:]))


class BinaryChromosome(Chromosome):
    def __init__(self, initial_length):
        # Numpy array of bits
        super(BinaryChromosome, self).__init__(
            self._randomizer.randint(2, size=initial_length))

    def _mutate_gene(self, gene, index):
        """
        Flip numpy array bit
        """
        return 1 - gene

    def _concat_genes(self, other):
        """
        Concatenate both numpy arrays
        """
        return np.concatenate((self[:], other[:]))

    def __repr__(self):
        return ''.join(str(int(c)) for c in self)
