import random
from abc import ABCMeta, abstractmethod


class Crossover(object):
    __metaclass__ = ABCMeta

    def __init__(self, rate):
        self._rate = rate

    @property
    def rate(self):
        return self._rate

    def run(self, parent1, parent2, rate=None):
        """
        Does crossover on two chromosome.
        """
        # Chance can be overriden:
        if not rate:
            rate = self.rate

        do_crossover = random.random() < rate
        if do_crossover:
            return self._run_specific(parent1, parent2)
        else:
            # No changes
            return parent1, parent2

    @abstractmethod
    def _run_specific(self, parent1, parent2):
        pass
