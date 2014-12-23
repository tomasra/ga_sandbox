#!/usr/bin/env python
import os
import argparse
import projects.denoising.experiments.parameters as params
import projects.denoising.experiments.performance as perf

MAX_ITERATIONS = 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir',
        help='Directory with generated parameter sets')
    parser.add_argument(
        'model',
        help='Pickled performance model')

    args = parser.parse_args()
    param_dir = os.path.abspath(args.dir)
    model_filepath = os.path.abspath(args.model)

    # Read params
    param_sets = params.read(param_dir)

    # Unpickle performance estimator
    model = perf.PerformanceModel.load(model_filepath)

    # Estimate optimal process counts
    proc_counts = [
        model.optimal_proc_count(
            param_set['population_size'],
            param_set['chromosome_length'],
            speedup_threshold=10.0)
        for param_set in param_sets
    ]

    # Estimate iteration times
    estimation_params = [
        (
            pair[0]['population_size'],
            pair[0]['chromosome_length'],
            pair[1])
        for pair in zip(param_sets, proc_counts)
    ]
    iteration_times = model.predict(estimation_params)
    total_estimated_time = sum([
        iteration_time * MAX_ITERATIONS
        for iteration_time in iteration_times
    ])
    print "Total estimated time: %f" % total_estimated_time

    # Group all param sets by optimal process count
    params_by_proc_count = {
        proc_count: [
            param_set
            for idx, param_set in enumerate(param_sets)
            if proc_counts[idx] == proc_count
        ]
        for proc_count in set(proc_counts)
    }


    # TEST TEST TEST
    # test_indexes = [
    #     idx
    #     for idx, ps in enumerate(param_sets)
    #     if ps['population_size'] == 50
    #     and ps['chromosome_length'] == 50
    # ]
    # test_pcs = [proc_counts[idx] for idx in test_indexes]

    # import pdb; pdb.set_trace()
