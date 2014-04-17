from abc import ABCMeta, abstractmethod


class SolutionFactory(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def create(self):
        """
        Instantiates new object.
        """
        pass
