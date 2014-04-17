import random
import copy
from lib.crossover import Crossover


class OnePointCrossover(Crossover):
    def _run_specific(self, parent1, parent2):
        cross_index = random.randint(0, len(parent1) - 1)
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)
        offspring1.content = parent1[:cross_index] + parent2[cross_index:]
        offspring2.content = parent2[:cross_index] + parent1[cross_index:]
        return offspring1, offspring2
