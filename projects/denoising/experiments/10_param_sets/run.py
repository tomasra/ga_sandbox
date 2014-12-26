#!/usr/bin/env python
import os
import argparse
from bunch import bunchify
from parameters import read
from projects.denoising.experiments import experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_dir', help='Input directory with parameter sets')
    parser.add_argument(
        'output_dir', help='Output directory for results')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    param_sets = read(input_dir)
    for param_set in param_sets:
        # Point output file to result directory
        param_set['output_file'] = os.path.join(
            output_dir,
            param_set['output_file'])
        # Run GA
        experiment.run(bunchify(param_set))
