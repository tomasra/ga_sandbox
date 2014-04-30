#!/venv/bin/python

import os
from imaging.utils import read_image, render_image
import imaging.filters as flt

# from lib.algorithm import Algorithm
# from lib.crossovers.one_point import OnePointCrossover
# from lib.selections.roulette_wheel import RouletteWheelSelection
from lib.solutions.filter_sequence import FilterSequenceEvaluator

# Image setup

filters = [
    flt.mean,
    flt.minimum,
    flt.maximum,
    flt.hsobel,
    flt.vsobel,
    flt.sobel,
    flt.lightedge,
    flt.darkedge,
    flt.erosion,
    flt.dilation,
    flt.inversion,
    flt.logical_sum,
    flt.logical_product,
    flt.algebraic_sum,
    flt.algebraic_product,
    flt.bounded_sum,
    flt.bounded_product
]

evaluator = FilterSequenceEvaluator(filters)

# Algorithm setup
# crossover = OnePointCrossover(rate=0.7)
# selection = RouletteWheelSelection()
# alg = Algorithm(
#     evaluator,
#     crossover,
#     selection,
#     population_size=300,
#     mutation_rate=0.001)

# for i in xrange(800):
#     alg.run()

cwd = os.getcwd()
filepath = os.path.join(cwd, 'data/target_chars/test_a1.png')
image = read_image(filepath)

test_sequence = [
    39, 35, 45, 11, 41,
    41, 33, 59, 43, 13,
    13, 66, 22, 22, 22,
    10, 22, 22, 41, 61,
    39, 46, 66, 38
]

filtered_image = evaluator._filter_image(image, test_sequence)
render_image(filtered_image)
