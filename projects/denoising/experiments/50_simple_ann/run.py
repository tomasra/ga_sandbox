#!/usr/bin/env python
import os
import sys
from bunch import bunchify
from projects.denoising.experiments import experiment

if __name__ == "__main__":
    args = {
        # Set in the loop
        'population_size': 100,
        'elite_size': 10,
        'selection': 'tournament',
        'tournament_size': 2,
        # 'crossover': 'whole_arithmetic',
        'crossover': 'two_point',
        'crossover_rate': 0.8,
        # 'crossover_alpha': 0.2,
        'mutation_rate': 0.01,

        # Run indefinitely
        'fitness_threshold': 1.0,
        'max_iterations': 50,
        'rng_freeze': False,

        'noise_type': 'gaussian',
        'noise_param': 0.005,

        'dump_images': True,
        'output_file': '1.json',
        'print_iterations': True,
        'filter_type': 'mlp'
    }
    experiment.run(args)
