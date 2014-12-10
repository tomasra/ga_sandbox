#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Variable-length integer chromosome test
"""

from core.algorithm import Algorithm
from core.chromosomes import IntegerChromosome
from core.crossovers import OnePointCrossover
from core.selections import RouletteWheelSelection
from core.parallelizer import Parallelizer, parallel_task
from projects.denoising.solution import FilterSequence
from projects.denoising.imaging.utils import render_image
from projects.denoising.imaging.char_drawer import CharDrawer
from projects.denoising.imaging.filter_call import FilterCall
import projects.denoising.imaging.noises as noises
import time
import numpy as np

# Suppress numpy's FutureWarnings
import warnings
warnings.filterwarnings('ignore')

FITNESS_THRESHOLD = 0.97

# GA parameters
POPULATION_SIZE = 100
ELITISM_COUNT = 10
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01
CHROMOSOME_LENGTH = 30
NULL_GENE_RATE = 0.2

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
    genotype=IntegerChromosome(
        length=CHROMOSOME_LENGTH,
        min_val=0,
        max_val=len(FilterCall.all()) - 1,
        null_gene_rate=NULL_GENE_RATE),
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
        for population, generation in algorithm.run():
            # pass
            best = population.best_individual.fitness
            worst = population.worst_individual.fitness
            average = population.average_fitness
            solution = population.best_individual

            active_genes = [
                individual.chromosome.active_gene_count()
                for individual in population
            ]
            ag_mean = np.mean(active_genes)
            ag_std = np.std(active_genes)

            print "#%i | best: %f, worst: %f, avg: %f, AG mean: %f, AG var: %f" % (
                generation, best, worst, average, ag_mean, ag_std)

            # print "#%i | best: %f, worst: %f, avg: %f" % (
            #     generation, best, worst, average)

            if best >= FITNESS_THRESHOLD:
                render_image(source_image.run_filters(solution))
                break

        end = time.time()
        duration = end - start
        print parallelizer.proc_count, duration
