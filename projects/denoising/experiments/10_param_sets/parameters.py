#!/usr/bin/env python
import os
import argparse
import json
import numpy as np
from collections import OrderedDict
import projects.denoising.experiments as exp
import projects.denoising.experiments.performance as perf
import projects.denoising.experiments.parameters as params


PROC_COUNT = 8
MAX_ITERATIONS = 500
MAX_BATCH_TIME = 3600

CHOICES = OrderedDict()
CHOICES['population_size'] = np.concatenate((
    # Small population: 9 options
    list(xrange(20, 180 + 1, 20)),
    # Large population: 7 options
    list(xrange(200, 500 + 1, 50))
))
CHOICES['elite_size'] = list(xrange(6))
CHOICES['selection'] = ['roulette', 'tournament', 'rank']
CHOICES['tournament_size'] = [2, 3, 4, 5]
CHOICES['crossover'] = ['one_point', 'two_point', 'uniform']
CHOICES['crossover_rate'] = np.arange(0.5, 1.0 + 0.001, 0.05)
CHOICES['mutation_rate'] = np.concatenate((
    # Low mutation rate: 5 options
    np.arange(0.001, 0.01, 0.002),
    # High mutation rate: 5 options
    np.arange(0.01, 0.1 + 0.0001, 0.02)
))
CHOICES['chromosome_length'] = list(xrange(10, 100 + 1, 10))
# CHOICES['noise_type'] = ['snp', 'gaussian']
# CHOICES['noise_param'] = np.arange(0.1, 0.5 + 0.001, 0.2)
CHOICES['noise'] = [
    'snp-0.1',
    'snp-0.3',
    'snp-0.5',
    'gaussian-0.1',
    'gaussian-0.3',
    'gaussian-0.5',
]


def generate_random(instances=100):
    """
    Generates 100 (or other specified number) parameter sets
    for each choice of each param.
    """
    all_args = []
    for current_name, choices in CHOICES.items():
        for choice in choices:
            for i in xrange(instances):
                # Start populating single parameter set
                args = {}
                for name in CHOICES.keys():
                    if current_name == name:
                        # Take next choice value for current parameter
                        args[name] = choice
                    else:
                        # Generate randomly
                        args[name] = np.random.choice(CHOICES[name])

                # Unpack noise info
                noise_type, noise_param = tuple(args['noise'].split('-'))
                args['noise_type'] = noise_type
                args['noise_param'] = float(noise_param)

                # args['noise_type'], args['noise_param'] = tuple(
                #     args['noise'].split('-')
                # )
                del args['noise']

                # Remaining params
                args['fitness_threshold'] = 1.01    # Run forever!
                args['max_iterations'] = MAX_ITERATIONS
                args['rng_freeze'] = False

                args['dump_images'] = False
                # Assign numbers to output files
                args['output_file'] = str(len(all_args)) + '.json'
                args['print_iterations'] = False

                # Finished
                all_args.append(args)

    return all_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir', help='Directory for storing generated parameter sets')
    parser.add_argument(
        'model', help='Path to trained and pickled PerformanceModel instance')
    args = parser.parse_args()

    # Create directory if necessary
    result_dir = os.path.abspath(args.dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        if len(os.listdir(result_dir)) > 0:
            raise ValueError("Directory exists and is not empty")

    time_estimator = perf.PerformanceModel.load(
        os.path.abspath(args.model))

    # 100 randomized combinations for each option of each parameter
    # Should be 8300 param sets total
    param_sets = generate_random(100)
    batches, batch_times = params.make_batches(param_sets, time_estimator)

    params.write_batches(result_dir, batches)

    # write(result_dir, param_sets)
    print "Generated parameter sets: %i" % len(param_sets)
    print "Total estimated time (with optimal process counts): %f" % (
        sum(batch_times))
