#!/usr/bin/env python
import os
import sys
from bunch import bunchify
from projects.denoising.experiments import experiment


def all_runs(result_dir):
    args = {
        'population_size': 100,
        'elite_size': 0,
        'selection': 'roulette',
        'tournament_size': 0,
        'crossover_rate': 0.8,
        'mutation_rate': 0.005,
        'chromosome_length': 30,
        'fitness_threshold': 0.98,
        'max_iterations': 1000,
        'rng_freeze': False,

        'noise_type': 'snp',
        'noise_param': 0.2,

        'dump_images': False,
        'output_file': 'output.json',
        'print_iterations': False,
    }

    # Increase population size by 10
    pid = os.getpid()
    for population_size in xrange(10, 1000 + 10, 10):
        # Elitism - 1/10 population
        elite_size = int(population_size * 0.1)
        output_filename = "population-%i-%i.json" % (population_size, pid)
        filepath = os.path.join(result_dir, output_filename)
        args['output_file'] = filepath
        args['population_size'] = population_size
        args['elite_size'] = elite_size
        experiment.run(bunchify(args))

if __name__ == "__main__":
    rel_dir = sys.argv[1]
    abs_dir = os.path.abspath(rel_dir)
    if not os.path.exists(abs_dir):
        os.makedirs(abs_dir)
    all_runs(abs_dir)
