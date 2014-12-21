#!/usr/bin/env python
import os
import argparse
import projects.denoising.experiments.performance as perf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir',
        help='Directory with .json result files')
    parser.add_argument(
        'output',
        help='Output .pickle file with fitted model')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.dir)
    output_filepath = os.path.abspath(args.output)

    # Do the fitting
    data = perf.PerformanceData.load(args.dir)
    model = perf.PerformanceModel()
    model.fit(data)

    # Pickle into the specified file
    model.save(output_filepath)
