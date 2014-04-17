from abc import ABCMeta, abstractmethod


class Selection(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def run(self, population):
        """
        Returns a single selected individual from population
        """
        pass
