#! /home/tomas/.virtualenvs/ga/bin/python
# -*- coding: utf-8 -*-
from core.algorithm import Algorithm
from core.chromosomes import IntegerChromosome
from core.crossovers import OnePointCrossover
from core.selections import RouletteWheelSelection
from core.parallelizer import Parallelizer
from projects.denoising.solution import FilterSequence
from projects.denoising.imaging.utils import render_image
from projects.denoising.imaging.char_drawer import CharDrawer
import projects.denoising.imaging.noises as noises


# GA parameters
FITNESS_THRESHOLD = 0.97
MAX_ITERATIONS = 500
POPULATION_SIZE = 300
ELITISM_COUNT = 30
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

with Parallelizer() as parallelizer:
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

        for population, generation in algorithm.run():
            best = population.best_individual.fitness
            average = population.average_fitness
            solution = population.best_individual
            print best, average
            if best > FITNESS_THRESHOLD:
                break

        # Sufficiently good solution found
        filtered_image = source_image.run_filters(solution)
        render_image(filtered_image)
