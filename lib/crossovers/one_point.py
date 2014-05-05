import random
# import copy
# import numpy as np
from lib.crossover import Crossover


class OnePointCrossover(Crossover):
    def _run_specific(self, parent1, parent2):
        cross_index = random.randint(0, len(parent1) - 1)
        parent1_part1, parent1_part2 = parent1.split(cross_index)
        parent2_part1, parent2_part2 = parent2.split(cross_index)
        offspring1 = parent1_part1.concat(parent2_part2)
        offspring2 = parent2_part1.concat(parent1_part2)
        return offspring1, offspring2
