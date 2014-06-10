#!venv/bin/python
# -*- coding: utf-8 -*-
from imaging.char_drawer import CharDrawer
import imaging.noises as noises
from imaging.utils import render_image

from lib.algorithm import Algorithm
from lib.crossovers.one_point import OnePointCrossover
from lib.selections.roulette_wheel import RouletteWheelSelection
from lib.solutions.filter_sequence import FilterSequenceEvaluator
from imaging.filter_call import FilterCall

# GA parameters
FITNESS_THRESHOLD = 0.97
MAX_ITERATIONS = 500
POPULATION_SIZE = 300
ELITISM_COUNT = 30
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.001
CHROMOSOME_LENGTH = 30

# Character parameters
ALL_CHARS = chars = u"AĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽ0123456789"
TEXT_COLOR = (0, 160, 0)
BACKGROUND_COLOR = (0, 255, 0)

# Noise parameters
SNP_NOISE_PARAM = 0.15
GAUSSIAN_NOISE_SIGMA = 40.0


def get_clear_char(character):
    """
    Clear color image of specified character
    """
    char_drawer = CharDrawer(
        text_color=TEXT_COLOR,
        bg_color=BACKGROUND_COLOR
    )
    pair = char_drawer.create_pair(character)
    return pair[1]


def get_binary_char(character):
    """
    Black and white image of specified character
    """
    char_drawer = CharDrawer(
        text_color=TEXT_COLOR,
        bg_color=BACKGROUND_COLOR
    )
    pair = char_drawer.create_pair(character)
    return pair[0]


def get_snp_noise_char(character, intensity=SNP_NOISE_PARAM):
    """
    Returns color character image with salt and pepper noise
    """
    clear_char = get_clear_char(character)
    noisified = noises.salt_and_pepper(clear_char, intensity)
    return noisified


def get_gaussian_noise_char(character, sigma=GAUSSIAN_NOISE_SIGMA):
    """
    Returns color character image with gaussian noise
    """
    clear_char = get_clear_char(character)
    noisified = noises.gaussian(clear_char, sigma=sigma)
    return noisified


def apply_filters(image, filter_sequence):
    """
    Apply filter sequence on image
    """
    # filter_calls = [
    #     FilterCall.all_calls()[idx]
    #     for idx in filter_sequence
    # ]
    # return FilterCall.run_sequence(image, filter_calls)
    evaluator = FilterSequenceEvaluator(FilterCall.all_calls(), [image], None)
    return evaluator._filter_image(image, filter_sequence)


def run(
        source_image,
        target_image=None,
        fitness_threshold=FITNESS_THRESHOLD,
        elitism=True):

    # wow such hack
    if target_image:
        target_image = [target_image]

    # Fitness calculator
    evaluator = FilterSequenceEvaluator(
        filter_calls=FilterCall.all_calls(),
        input_images=[source_image],
        target_images=target_image,
        sequence_length=CHROMOSOME_LENGTH)

    if elitism:
        elitism_count = ELITISM_COUNT
    else:
        elitism_count = 0

    # Algorithm setup
    crossover = OnePointCrossover(rate=CROSSOVER_RATE)
    selection = RouletteWheelSelection()
    alg = Algorithm(
        evaluator,
        crossover,
        selection,
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        elitism_count=elitism_count)

    print "Running GA..."
    best_fitness = 0
    iteration_count = 0
    average_fitnesses, best_fitnesses = [], []
    while best_fitness < fitness_threshold and iteration_count <= MAX_ITERATIONS:
        alg.run()
        s = "Avg fitness: " + str(alg.population.average_fitness)
        s += ", best fitness: " + str(alg.population.best_solution.fitness)
        # s += ", best solution: " + str(alg.population.best_solution)
        print s
        best_fitness = alg.population.best_solution.fitness
        average_fitnesses += [alg.population.average_fitness]
        best_fitnesses += [alg.population.best_solution.fitness]
        iteration_count += 1

    best_solution = alg.population.best_solution.sequence

    # best_filter_calls = evaluator.call_list(best_solution)
    # render_image(source_image)
    # best_solution_image = FilterCall.run_sequence(
    #     source_image, best_filter_calls)

    # best_solution_image = evaluator._filter_image(source_image, best_solution)
    best_solution_image = apply_filters(source_image, best_solution)

    return (
        average_fitnesses,
        best_fitnesses,
        best_solution,
        best_solution_image
    )
