#!/usr/bin/env python
import os
import sys
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('param_file', help='Path to parameter file')
    args = {
        # Set in the loop
        'population_size': 40,
        'elite_size': 4,
        'selection': 'tournament',
        'tournament_size': 2,
        # 'crossover': 'whole_arithmetic',
        'crossover': 'one_point',
        'crossover_rate': 0.8,
        # 'crossover_alpha': 0.2,
        'mutation_rate': 0.01,

        # Run indefinitely
        'fitness_threshold': 1.0,
        'max_iterations': 5,
        'rng_freeze': False,

        'dump_images': False,
        'input_image': '/home/tomas/Masters/4_semester/synthetic_tests/noisy/1/noisy-10-003-06.png',
        # 'output_file': '/home/tomas/Masters/4_semester/synthetic_tests/results/ga_simple/noisy-10-003-06.json',
        'output_file': 'noisy-10-003-06.json',
        'print_iterations': True,
        'filter_type': 'mlp'
    }
    experiment.run(args)
