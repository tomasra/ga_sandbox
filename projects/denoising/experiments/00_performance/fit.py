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
    model = perf.PerformanceModel(curve='hyperbolic')
    model.fit(data)


    # perf.plot_real_predicted(data, model, 30, 30)

    # TEST TEST TEST
    # data2_dir = '/home/tomas/Masters/00_performance/results2'
    # data2 = perf.PerformanceData.load(data2_dir)
    # print model.r2_score(data2)

    # Pickle into the specified file
    model.save(output_filepath)

    # perf.plot_real_predicted(
    #     data2, model,
    #     population_size=30,
    #     chromosome_length=35)

    # import pdb; pdb.set_trace()
