from abc import ABCMeta, abstractmethod


class Chromosome(object):
    __metaclass__ = ABCMeta

    def __init__(self, length=None, content=None):
        """
        Initializes chromosome as random bit string of specified length.
        """
        if content:
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

    def __len__(self):
        return len(self.content)

    def __iter__(self):
        for i in range(0, len(self.content)):
            yield self.content[i]

    def __getitem__(self, key):
        """
        Returns chromosome value by specified index or slice.
        """
        if isinstance(key, slice):
            # Collect items according to slice
            items = [self[i] for i in xrange(*key.indices(len(self)))]
            if isinstance(self.content, str):
                # Concatenate items into string
                return ''.join(items)
            else:
                return items
        elif isinstance(key, int):
            # Check for negative or out-of-range indexes
            if key >= len(self.content) or key < 0:
                raise IndexError("Sequence index out of range.")
            return self.content[key]
        else:
            raise TypeError("Invalid argument type.")

    def __setitem__(self, index, value):
        """
        Sets a single item by index
        """
        if isinstance(self.content, str):
            # Can't set value directly
            l = list(self.content)
            l[index] = value
            self.content = ''.join(l)
        else:
            self.content[index] = value
