#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance test.
"""

from core.algorithm import Algorithm
from core.chromosomes import IntegerChromosome
from core.crossovers import OnePointCrossover
from core.selections import RouletteWheelSelection
from core.parallelizer import Parallelizer, parallel_task
from projects.denoising.solution import FilterSequence
from projects.denoising.imaging.utils import render_image
from projects.denoising.imaging.char_drawer import CharDrawer
import projects.denoising.imaging.noises as noises
import time

# Initialize all RNGs with constant seeds
import numpy as np
from core.chromosomes import Chromosome
from core.crossovers import Crossover
from core.selections import Selection

Chromosome._randomizer = np.random.RandomState(0)
Crossover._randomizer = np.random.RandomState(1)
Selection._randomizer = np.random.RandomState(2)
noises._randomizer = np.random.RandomState(3)

# Suppress numpy's FutureWarnings
import warnings
warnings.filterwarnings('ignore')


# GA parameters
POPULATION_SIZE = 100
ELITISM_COUNT = 10
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.001
CHROMOSOME_LENGTH = 30

# Character parameters
ALL_CHARS = u"AĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽ0123456789"
TEXT_COLOR = (0, 160, 0)
BACKGROUND_COLOR = (0, 255, 0)
TTB_RATIO = 0.1

# Noise parameters
SNP_NOISE_PARAM = 0.15
GAUSSIAN_NOISE_SIGMA = 40.0

chars = CharDrawer()
#---------------------------------------------------
# Salt-and-pepper
# Known target image
#---------------------------------------------------
source_image = chars.create_colored_char(
    'A', TEXT_COLOR, BACKGROUND_COLOR)

# Add noise
source_image = noises.salt_and_pepper(source_image, SNP_NOISE_PARAM)

# Clean binary target image
target_image = chars.create_binary_char('A')

phenotype = FilterSequence(
    sequence_length=CHROMOSOME_LENGTH,
    source_image=source_image,
    target_image=target_image)


def calculate_fitness(chromosome):
    individual = phenotype()
    individual.chromosome = chromosome
    return individual._calculate_fitness()
prepared_tasks = [calculate_fitness]

with Parallelizer(prepared_tasks) as parallelizer:
    if parallelizer.master_process:
        solution = None

        # Start GA
        algorithm = Algorithm(
            phenotype=phenotype,
            crossover=OnePointCrossover(CROSSOVER_RATE),
            selection=RouletteWheelSelection(),
            population_size=POPULATION_SIZE,
            mutation_rate=MUTATION_RATE,
            elitism_count=ELITISM_COUNT,
            parallelizer=parallelizer)

        start = time.time()
        for population, generation in algorithm.run(100):
            pass
            # best = population.best_individual.fitness
            # worst = population.worst_individual.fitness
            # average = population.average_fitness
            # solution = population.best_individual
            # print "#%i | best: %f, worst: %f, avg: %f" % (
            #     generation, best, worst, average)

        end = time.time()
        duration = end - start
        print parallelizer.proc_count, duration
