#!/usr/bin/env python
import os
import sys
import argparse
from bunch import bunchify
from projects.denoising.experiments import experiment


def filter_image(ann_path, noisy_image_path, output_path):
    from fann2 import libfann
    from skimage import io, util
    from projects.denoising.neural.filtering import filter_fann
    
    noisy_image = util.img_as_float(io.imread(noisy_image_path))
    ann = libfann.neural_net()
    ann.create_from_file(ann_path)
    filtered_image = filter_fann(noisy_image, ann)
    io.imsave(output_path, filtered_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('param_file', help='Path to parameter file')
    parser.add_argument('image')
    parser.add_argument('run')
    args = vars(parser.parse_args())
    
    INPUT_IMAGE = '/home/tomas/Masters/4_semester/synthetic_tests/noisy/1/' + args['image'] + '.png'
    OUTPUT_FILE = '/home/tomas/Masters/4_semester/synthetic_tests/results/ga/' + args['run'] + '/' + args['image'] + '.json'

    args = {
        # Set in the loop
        'population_size': 200,
        'elite_size': 10,
        'selection': 'tournament',
        'tournament_size': 2,
        # 'crossover': 'whole_arithmetic',
        'crossover': 'one_point',
        # 'crossover': 'uniform',
        'crossover_rate': 0.8,
        # 'crossover_alpha': 0.2,
        'mutation_rate': 0.01,
        'init_method': 'normal',
        'fitness_func': 'stat',

        # Run indefinitely
        'fitness_threshold': 1.0,
        'max_iterations': 10,
        'rng_freeze': False,

        'dump_images': False,
        # 'input_image': '/home/tomas/Masters/4_semester/synthetic_tests/noisy/1/noisy-20-003-04.png',
        'input_image': INPUT_IMAGE,
        # 'output_file': '/home/tomas/Masters/4_semester/synthetic_tests/results/ga_simple/noisy-10-003-06.json',
        # 'output_file': 'noisy-20-003-04.json',
        'output_file': OUTPUT_FILE,
        'print_iterations': True,
        'filter_type': 'mlp'
    }
    # experiment.run(args)
    # filter_image(
    #     'noisy-20-003-04.net',
    #     '/home/tomas/Masters/4_semester/synthetic_tests/noisy/1/noisy-20-003-04.png',
    #     'filtered.png'
    # )
