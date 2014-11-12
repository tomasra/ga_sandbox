# -*- coding: utf-8 -*-
#!/usr/bin.python

import random
import numpy as np
from abc import ABCMeta, abstractmethod


class Chromosome(object):
    """
        Abstract base class for all types of chromosomes
    """

    __metaclass__ = ABCMeta

    def __init__(self, length=None, content=None):
        """
        Initializes chromosome as random bit string of specified length.
        """
        if content is not None:
            self.content = content
        elif length:
            self.content = self._get_random(length)
        else:
            raise ValueError(
                "Must supply either chromosome length or initial content")

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    @abstractmethod
    def _get_random(self, length):
        """
        Randomizes chromosome content.
        """
        pass

    @abstractmethod
    def mutate(self, rate):
        """
        Mutates single item with given probability
        """
        pass

    @abstractmethod
    def split(self, point):
        """
        Splits chromosome in two at the specified point
        """
        pass

    @abstractmethod
    def concat(self, other):
        """
        Concatenates this chromosome with another
        """
        pass

    def __len__(self):
        return len(self.content)

    def __iter__(self):
        for i in range(0, len(self.content)):
            yield self.content[i]

    def __getitem__(self, key):
        """
        Returns chromosome value by specified index or slice.
        """
        return self.content[key]

    def __setitem__(self, index, value):
        """
        Sets a single item by index
        """
        self.content[index] = value


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


class BinaryChromosome(Chromosome):
    def __init__(self, length=None, content=None):
        """
        For convenience: if content is passed as string,
        convert it to numpy array
        """
        if isinstance(content, str):
            converted_content = np.array([
                int(char)
                for char in content
            ])
        else:
            converted_content = content
        super(BinaryChromosome, self).__init__(length, converted_content)

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


class RealChromosome(object):
    pass
