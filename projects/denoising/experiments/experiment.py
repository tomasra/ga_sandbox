#!/usr/bin/env python
import os
import time
import json
import pickle
import numpy as np
from bunch import bunchify

from core.algorithm import Algorithm
from core.crossovers import get_crossover
from core.selections import get_selection
from core.parallelizer import Parallelizer
from projects.denoising.solution import get_phenotype
import projects.denoising.imaging.noises as noises
from projects.denoising.experiments.parameters import parse_cli_args

# Suppress numpy's FutureWarnings
import warnings
warnings.filterwarnings('ignore')


def run(args):
    if args['rng_freeze'] is True:
        from core.chromosomes import Chromosome
        from core.crossovers import Crossover
        from core.selections import Selection

        Chromosome._randomizer = np.random.RandomState(0)
        Crossover._randomizer = np.random.RandomState(1)
        Selection._randomizer = np.random.RandomState(2)
        noises._rng_seed = 3

    #---------------------------------------------------------------------------
    # GA setup
    #---------------------------------------------------------------------------
    with Parallelizer() as parallelizer:
        if parallelizer.master_process:
            # Create phenotype representing specific problem to solve
            # and distribute copies to workers
            phenotype = get_phenotype(args)
            parallelizer.broadcast(phenotype=phenotype)
            selection = get_selection(args)
            crossover = get_crossover(args)

            # Put all parameters into json
            output = {}
            output['parameters'] = args

            # Save images into results file?
            if args['dump_images'] is True:
                output['parameters']['source_image_dump'] = pickle.dumps(
                    phenotype.source_image)
                output['parameters']['target_image_dump'] = pickle.dumps(
                    phenotype.target_image)

            # Populate during run
            output['iterations'] = []
            output['results'] = {}
            output['parameters']['proc_count'] = parallelizer.proc_count

            solution = None

            # Start GA
            algorithm = Algorithm(
                phenotype=phenotype,
                crossover=crossover,
                selection=selection,
                population_size=args['population_size'],
                mutation_rate=args['mutation_rate'],
                elitism_count=args['elite_size'],
                parallelizer=parallelizer)

            # Start counting NOW!
            start = time.time()
            for population, generation in algorithm.run(args['max_iterations']):
                # For debugging, mostly
                if args['print_iterations'] is True:
                    best = population.best_individual.fitness
                    worst = population.worst_individual.fitness
                    average = population.average_fitness
                    solution = population.best_individual
                    print "#%i | best: %f, worst: %f, avg: %f" % (
                        generation, best, worst, average)

                # Write each iteration statistics into output
                iteration_output = {
                    'number': generation,
                    'best_fitness': population.best_individual.fitness,
                    # 'worst_fitness': population.worst_individual.fitness,
                    'average_fitness': population.average_fitness,
                }
                output['iterations'].append(iteration_output)

                solution = population.best_individual
                if solution.fitness >= args['fitness_threshold']:
                    break

            # Time's up
            end = time.time()
            duration = end - start

            # Write results to file
            # But first - unset source and target images to prevent them
            # from being serialized
            solution.source_image = None
            solution.target_image = None

            output['results'] = {
                'solution_dump': pickle.dumps(solution),
                'run_time': duration,
                'iterations': generation,
            }

            with open(args['output_file'], 'w') as f:
                json.dump(output, f)

if __name__ == "__main__":
    args = parse_cli_args()
    run(args)
