#!/usr/bin/env python
import os
import argparse
import json
import projects.denoising.experiments.parameters as params
import projects.denoising.experiments.performance as perf

MAX_ITERATIONS = 500
PROC_COUNT = 8


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


def read_batches(directory, sort=False):
    """
    Same as above, but reads param sets from different folders
    representing separate batches
    """
    dirpaths = [
        os.path.join(directory, dirname)
        for dirname in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, dirname))
    ]
    all_param_sets = []
    for dirpath in dirpaths:
        batch_param_sets = read(dirpath)
        all_param_sets += batch_param_sets

    if sort is True:
        all_param_sets = sorted(
            all_param_sets,
            # Sort by output file ID
            key=lambda ps: int(ps['output_file'].split('.')[0])
        )

    return all_param_sets


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
    param_sets = read_batches(param_dir, sort=True)

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
    # estimation_params = [
    #     (
    #         pair[0]['population_size'],
    #         pair[0]['chromosome_length'],
    #         pair[1])
    #     for pair in zip(param_sets, proc_counts)
    # ]

    estimation_params = [
        (
            param_set['population_size'] - param_set['elite_size'],
            param_set['chromosome_length'],
            PROC_COUNT)
        for param_set in param_sets
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
    # print params_by_proc_count


    # TEST TEST TEST
    # test_indexes = [
    #     idx
    #     for idx, ps in enumerate(param_sets)
    #     if ps['population_size'] == 50
    #     and ps['chromosome_length'] == 50
    # ]
    # test_pcs = [proc_counts[idx] for idx in test_indexes]

    import pdb; pdb.set_trace()
