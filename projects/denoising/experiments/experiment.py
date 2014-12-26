#!/usr/bin/env python
import os
import argparse
import time
import json
import pickle
import numpy as np
from bunch import bunchify

# Setup GA
from core.algorithm import Algorithm
from core.chromosomes import IntegerChromosome
from core.crossovers import OnePointCrossover, TwoPointCrossover
from core.crossovers import UniformCrossover
from core.selections import RouletteWheelSelection, TournamentSelection
from core.selections import RankSelection
from core.parallelizer import Parallelizer
from projects.denoising.solution import FilterSequence
from projects.denoising.imaging.char_drawer import CharDrawer
from projects.denoising.imaging.filter_call import FilterCall
import projects.denoising.imaging.noises as noises

# Suppress numpy's FutureWarnings
import warnings
warnings.filterwarnings('ignore')

RESULT_FORMAT = '.json'
TEXT_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (255, 255, 255)


def parse_cli_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='GA experiment'
    )

    # GA params
    parser.add_argument('--population-size',
                        action='store', type=int, default=100)
    parser.add_argument('--elite-size',
                        action='store', type=int, default=10)
    parser.add_argument('--crossover',
                        action='store', type=str,
                        choices=(
                            'one_point',
                            'two_point',
                            'uniform'),
                        default='one_point')
    parser.add_argument('--crossover-rate',
                        action='store', type=float, default=0.8)
    parser.add_argument('--selection',
                        action='store', type=str,
                        choices=(
                            'roulette',
                            'tournament',
                            'rank'),
                        default='roulette',)
    # Only relevant to tournament selection
    parser.add_argument('--tournament-size',
                        action='store', type=float, default=2)
    parser.add_argument('--mutation-rate',
                        action='store', type=float, default=0.005)
    parser.add_argument('--chromosome-length',
                        action='store', type=int, default=30)
    parser.add_argument('--fitness-threshold',
                        action='store', type=float, default=0.98)
    parser.add_argument('--max-iterations',
                        action='store', type=int, default=1000)
    parser.add_argument('--rng-freeze',
                        action='store', type=bool, default=False)

    # Filtering params
    parser.add_argument('--noise-type',
                        action='store', type=str,
                        choices=('snp', 'gaussian'),
                        default='snp')
    parser.add_argument('--noise-param',
                        action='store', type=float, default=0.2)

    # Output
    parser.add_argument('--dump-images',
                        action='store', type=bool, default=False)
    parser.add_argument('--output-file',
                        action='store', type=str, default='output.json')
    parser.add_argument('--print-iterations',
                        action='store', type=bool, default=False)

    args = parser.parse_args()
    return args


def read_results_file(filepath):
    """
    Read single JSON-formatted result file
    """
    with open(filepath, 'r') as f:
        json_results = json.load(f)
    result_set = bunchify(json_results)
    return result_set


def read_results(directory):
    """
    Read JSON-formatted results from directory
    """
    # Absolute dir path
    abs_directory = os.path.abspath(directory)

    # Enumerate json files
    filepaths = [
        os.path.join(abs_directory, filename)
        for filename in os.listdir(abs_directory)
        if os.path.isfile(os.path.join(abs_directory, filename))
        and filename.endswith(RESULT_FORMAT)
    ]

    # Read the actual results
    result_sets = [
        read_results_file(filepath)
        for filepath in filepaths
    ]
    return result_sets


def generate_images(noise_type, noise_param):
    """
    Create source image (with noise) and clean target image
    """
    chars = CharDrawer(
        image_size=40,
        char_size=36,
        text_color=TEXT_COLOR,
        bg_color=BACKGROUND_COLOR)

    target_image = chars.create_colored_char(
        'A', TEXT_COLOR, BACKGROUND_COLOR)
    # Add noise
    if noise_type == 'snp':
        source_image = noises.salt_and_pepper(
            target_image, noise_param)
    elif noise_type == 'gaussian':
        source_image = noises.gaussian(
            target_image, var=noise_param)
    else:
        raise ValueError("Unknown noise type: %s" % noise_type)

    return source_image, target_image


def run(args):
    if args.rng_freeze is True:
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
            source_image, target_image = generate_images(
                args.noise_type, args.noise_param)

            phenotype = FilterSequence(
                genotype=IntegerChromosome(
                    length=args.chromosome_length,
                    min_val=0,
                    max_val=len(FilterCall.all()) - 1),
                source_image=source_image,
                target_image=target_image)

            parallelizer.broadcast(phenotype=phenotype)

            # Put all parameters into json
            output = {}
            output['parameters'] = {
                'population_size': args.population_size,
                'elite_size': args.elite_size,
                'selection': args.selection,
                'crossover_rate': args.crossover_rate,
                'mutation_rate': args.mutation_rate,
                'chromosome_length': args.chromosome_length,
                'fitness_threshold': args.fitness_threshold,
                'max_iterations': args.max_iterations,
                'noise_type': args.noise_type,
                'noise_param': args.noise_param,
            }

            # Selection type
            if args.selection == 'roulette':
                selection = RouletteWheelSelection()
            elif args.selection == 'tournament':
                selection = TournamentSelection(args.tournament_size)
                output['parameters']['tournament_size'] = args.tournament_size
            elif args.selection == 'rank':
                selection = RankSelection()
            else:
                raise ValueError("Unknown selection type: %s" % args.selection)

            # Crossover type
            if args.crossover == 'one_point':
                crossover = OnePointCrossover(args.crossover_rate)
            elif args.crossover == 'two_point':
                crossover = TwoPointCrossover(args.crossover_rate)
            elif args.crossover == 'uniform':
                crossover = UniformCrossover(args.crossover_rate)
            else:
                raise ValueError("Unknown crossover type: %s" % args.crossover)

            # Save images into results file?
            if args.dump_images is True:
                output['parameters']['source_image_dump'] = pickle.dumps(
                    source_image)
                output['parameters']['target_image_dump'] = pickle.dumps(
                    target_image)

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
                population_size=args.population_size,
                mutation_rate=args.mutation_rate,
                elitism_count=args.elite_size,
                parallelizer=parallelizer)

            # Start counting NOW!
            start = time.time()
            for population, generation in algorithm.run(args.max_iterations):
                # For debugging, mostly
                if args.print_iterations is True:
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
                if solution.fitness >= args.fitness_threshold:
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

            with open(args.output_file, 'w') as f:
                json.dump(output, f)

if __name__ == "__main__":
    args = parse_cli_args()
    run(args)
