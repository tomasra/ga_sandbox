#!venv/bin/python
# -*- coding: utf-8 -*-
from imaging.char_drawer import CharDrawer
import imaging.noises as noises
from imaging.utils import render_image

import imaging.filters as flt
from lib.algorithm import Algorithm
from lib.crossovers.one_point import OnePointCrossover
from lib.selections.roulette_wheel import RouletteWheelSelection
from lib.solutions.filter_sequence import FilterSequenceEvaluator
from imaging.filter_call import FilterCall

# Filter setup
print "Setting up filters..."
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
filters1 = FilterCall.make_calls(filters_one_arg)
filters2 = FilterCall.make_calls(filters_two_args)
filter_calls = filters1 + filters2


# Character setup
print "Setting up images..."
char_drawer = CharDrawer(
    text_color=(0, 180, 0),
    bg_color=(0, 255, 0)
)
chars = u"AĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽ0123456789"
pairs = [
    char_drawer.create_pair(char)
    for char in chars
]
bws = [pair[0] for pair in pairs]
colors = [pair[1] for pair in pairs]
noisified = noises.salt_and_pepper_all(colors, 0.2)
# noisified = noises.gaussian_all(colors, sigma=60.0)


# GA setup
print "Setting up GA..."
training_input = noisified[chars.index(u'A')]
training_target = bws[chars.index(u'A')]

input = CharDrawer.create_mosaic(noisified, 8, 6)
render_image(input)
# render_image(training_input)
# render_image(CharDrawer.create_mosaic(bws, 8, 6))

evaluator = FilterSequenceEvaluator(
    filter_calls,
    [training_input],
    # [training_target],
    target_images=None,
    sequence_length=20)

crossover = OnePointCrossover(rate=0.8)
selection = RouletteWheelSelection()
alg = Algorithm(
    evaluator,
    crossover,
    selection,
    population_size=300,
    mutation_rate=0.001,
    elitism_count=30)


print "Running GA..."
for i in xrange(20):
    alg.run()
    s = "Avg fitness: " + str(alg.population.average_fitness)
    s += ", best fitness: " + str(alg.population.best_solution.fitness)
    s += ", best solution: " + str(alg.population.best_solution)
    # for solution in alg.population.solutions:
        # print solution.fitness, solution
        # continue
    # pdb.set_trace()
    print s


# Results
best_solution = alg.population.best_solution
best_filter_calls = evaluator.call_list(best_solution.sequence)
result_images = [
    FilterCall.run_sequence(image, best_filter_calls)
    for image in colors
]

# input = CharDrawer.create_mosaic(noisified, 8, 6)
expected = CharDrawer.create_mosaic(bws, 8, 6)
actual = CharDrawer.create_mosaic(result_images, 8, 6)

# render_image(input)
render_image(expected)
render_image(actual)
