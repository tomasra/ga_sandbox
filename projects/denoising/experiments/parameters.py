#!/usr/bin/env python
import os
import argparse
import json
import numpy as np
from collections import OrderedDict
import projects.denoising.experiments as exp


CHOICES = OrderedDict()
CHOICES['population_size'] = np.concatenate((
    list(xrange(10, 100, 10)),
    list(xrange(100, 300, 20)),
    list(xrange(300, 500 + 1, 50))
))
CHOICES['elite_size'] = list(xrange(6))
CHOICES['selection'] = ['roulette', 'tournament', 'rank']
CHOICES['crossover'] = ['one_point', 'two_point', 'uniform']
CHOICES['crossover_rate'] = np.arange(0.5, 1.0 + 0.001, 0.05)
CHOICES['mutation_rate'] = np.concatenate((
    np.arange(0.001, 0.01, 0.001),
    np.arange(0.01, 0.1 + 0.0001, 0.01)
))
CHOICES['chromosome_length'] = list(xrange(10, 100 + 1, 10))
CHOICES['noise_type'] = ['snp', 'gaussian']
CHOICES['noise_param'] = np.arange(0.1, 0.5 + 0.001, 0.1)


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

                # Remaining params
                args['fitness_threshold'] = 1.01    # Run forever!
                args['max_iterations'] = 1000
                args['rng_freeze'] = False

                args['dump_images'] = False
                # Assign numbers to output files
                args['output_file'] = str(len(all_args)) + '.json'
                args['print_iterations'] = False

                # Finished
                all_args.append(args)

    return all_args


def write(directory, param_sets):
    """
    Write .json files with parameter sets into directory
    """
    for idx, param_set in enumerate(param_sets):
        filepath = os.path.join(directory, str(idx) + '.json')
        with open(filepath, 'w') as fp:
            json.dump(param_set, fp)


def read(directory):
    """
    Read parameter sets from directory
    """
    # Read .json files
    filepaths = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, filename))
    ]

    param_sets = []
    for filepath in filepaths:
        with open(filepath, 'r') as fp:
            param_set = json.load(fp)
            param_sets.append(param_set)
    return param_sets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir', help='Directory for storing generated parameter sets')
    args = parser.parse_args()

    # Create directory if necessary
    result_dir = os.path.abspath(args.dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        if len(os.listdir(result_dir)) > 0:
            raise ValueError("Directory exists and is not empty")

    # 100 randomized combinations for each option of each parameter
    # Should be 8300 param sets total
    param_sets = generate_random(100)
    write(result_dir, param_sets)

    print "Generated parameter sets: %i" % len(params)
