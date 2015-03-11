#!/usr/bin/env python
import os
import sys
from bunch import bunchify
from projects.denoising.experiments import experiment


def all_runs(result_dir):
    args = {
        # Set in the loop
        'population_size': 0,
        'elite_size': 0,
        'selection': 'roulette',
        'tournament_size': 0,
        'crossover': 'one_point',
        'crossover_rate': 0.8,
        'mutation_rate': 0.005,
        # Set in the loop
        'chromosome_length': 0,
        # Run indefinitely
        'fitness_threshold': 1.0,
        'max_iterations': 50,
        'rng_freeze': False,

        'noise_type': 'snp',
        'noise_param': 0.2,

        'dump_images': False,
        'output_file': 'output.json',
        'print_iterations': True,
    }

    # One run for each elitism value
    pid = os.getpid()
    for chromosome_length in xrange(10, 50 + 1, 5):
        for population_size in xrange(10, 50 + 1, 5):
            output_filename = "pop-%i-len-%i-%i.json" % (
                population_size, chromosome_length, pid)
            filepath = os.path.join(result_dir, output_filename)
            args['chromosome_length'] = chromosome_length
            args['population_size'] = population_size
            args['output_file'] = filepath
            experiment.run(args)

if __name__ == "__main__":
    rel_dir = sys.argv[1]
    abs_dir = os.path.abspath(rel_dir)
    all_runs(abs_dir)
