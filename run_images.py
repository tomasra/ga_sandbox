#!/venv/bin/python

import os
cwd = os.getcwd()
from imaging.utils import read_image, render_image
import imaging.filters as flt

from lib.algorithm import Algorithm
from lib.crossovers.one_point import OnePointCrossover
from lib.selections.roulette_wheel import RouletteWheelSelection
from lib.solutions.filter_sequence import FilterSequenceEvaluator
from imaging.filter_call import FilterCall

# Filter setup
filters_one_arg = [
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
    flt.inversion
]

filters_two_args = [
    flt.logical_sum,
    flt.logical_product,
    flt.algebraic_sum,
    flt.algebraic_product,
    flt.bounded_sum,
    flt.bounded_product
]

# filter_calls = FilterCall.make_calls(filters)
filters1 = FilterCall.make_calls(filters_one_arg)
filters2 = FilterCall.make_calls(filters_two_args)
filter_calls = filters1 + filters2

# Image setup
training_input = [
    read_image(os.path.join(cwd, 'samples/training/input/d1.png')),
    read_image(os.path.join(cwd, 'samples/training/input/d2.png'))
]
training_target = [
    read_image(os.path.join(cwd, 'samples/training/target/d1.png')),
    read_image(os.path.join(cwd, 'samples/training/target/d2.png'))
]

testing_input = [
    read_image(os.path.join(cwd, 'samples/testing/input/d1.png')),
    read_image(os.path.join(cwd, 'samples/testing/input/d2.png'))
]
testing_target = [
    read_image(os.path.join(cwd, 'samples/testing/target/d1.png')),
    read_image(os.path.join(cwd, 'samples/testing/target/d2.png'))
]

evaluator = FilterSequenceEvaluator(
    filter_calls,
    training_input,
    training_target)

# Algorithm setup
crossover = OnePointCrossover(rate=0.7)
selection = RouletteWheelSelection()
alg = Algorithm(
    evaluator,
    crossover,
    selection,
    population_size=2,
    mutation_rate=0.001)

for i in xrange(20):
    alg.run()
    s = "Average fitness: " + str(alg.population.average_fitness)
    s += ", best solution: " + str(alg.population.best_solution)
    print s


# See results
best_solution = alg.population.best_solution
best_filter_calls = evaluator.call_list(best_solution.sequence)

render_image(FilterCall.run_sequence(
    testing_input[0],
    best_filter_calls
))
render_image(FilterCall.run_sequence(
    testing_input[1],
    best_filter_calls
))
