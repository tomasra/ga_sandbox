from abc import ABCMeta, abstractmethod


class Chromosome(object):
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
