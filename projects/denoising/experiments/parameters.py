import os
import json
import argparse

PROC_COUNT = 8
MAX_ITERATIONS = 500
MAX_BATCH_TIME = 3600


def parse_cli_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='GA experiment'
    )

    # GA params
    parser.add_argument('--population-size',
                        action='store', type=int, default=100)
    parser.add_argument('--elite-size',
                        action='store', type=int, default=10)
    parser.add_argument('--crossover',
                        action='store', type=str,
                        choices=(
                            'one_point',
                            'two_point',
                            'uniform'),
                        default='one_point')
    parser.add_argument('--crossover-rate',
                        action='store', type=float, default=0.8)
    parser.add_argument('--selection',
                        action='store', type=str,
                        choices=(
                            'roulette',
                            'tournament',
                            'rank'),
                        default='roulette',)
    # Only relevant to tournament selection
    parser.add_argument('--tournament-size',
                        action='store', type=float, default=2)
    parser.add_argument('--mutation-rate',
                        action='store', type=float, default=0.005)
    parser.add_argument('--chromosome-length',
                        action='store', type=int, default=30)
    parser.add_argument('--fitness-threshold',
                        action='store', type=float, default=0.98)
    parser.add_argument('--max-iterations',
                        action='store', type=int, default=1000)
    parser.add_argument('--rng-freeze',
                        action='store', type=bool, default=False)

    # Filtering params
    parser.add_argument('--noise-type',
                        action='store', type=str,
                        choices=('snp', 'gaussian'),
                        default='snp')
    parser.add_argument('--noise-param',
                        action='store', type=float, default=0.2)

    # Output
    parser.add_argument('--dump-images',
                        action='store', type=bool, default=False)
    parser.add_argument('--output-file',
                        action='store', type=str, default='output.json')
    parser.add_argument('--print-iterations',
                        action='store', type=bool, default=False)

    args = parser.parse_args()
    return args


def make_batches(param_sets, time_estimator):
    """
    Estimate algorithm run time for each param set
    and group them together in such way that group's time
    would not exceed specific time.
    Trained PerformanceModel instance needs to be passed for
    time prediction.
    """
    # # Estimate optimal process counts
    # proc_counts = [
    #     time_estimator.optimal_proc_count(
    #         param_set['population_size'],
    #         param_set['chromosome_length'],
    #         speedup_threshold=10.0)
    #     for param_set in param_sets
    # ]

    # Estimate time for each param set
    estimation_params = [
        (
            param_set['population_size'],
            param_set['chromosome_length'],
            PROC_COUNT)
        for param_set in param_sets
    ]
    param_set_times = [
        iteration_time * MAX_ITERATIONS
        for iteration_time in time_estimator.predict(
            estimation_params)
    ]

    batches, batch_times = [], []
    current_batch, current_time = [], 0.0
    for idx, param_set in enumerate(param_sets):
        if current_time + param_set_times[idx] > MAX_BATCH_TIME:
            # Close this batch
            batches.append(current_batch)
            batch_times.append(current_time)
            current_batch, current_time = [], 0.0

        # Continue adding param sets
        current_batch.append(param_set)
        current_time += param_set_times[idx]

    # Last batch
    batches.append(current_batch)
    batch_times.append(current_time)

    return batches, batch_times


def write(directory, param_sets):
    """
    Write .json files with parameter sets into directory
    """
    for idx, param_set in enumerate(param_sets):
        filepath = os.path.join(directory, str(idx) + '.json')
        with open(filepath, 'w') as fp:
            json.dump(param_set, fp)


def write_batches(directory, batches):
    """
    Write param sets as .json files in batches.
    Create separate folder for each.
    """
    for idx, batch in enumerate(batches):
        # Create dir
        batch_dir = os.path.join(directory, str(idx))
        os.makedirs(batch_dir)

        # Write param sets
        write(batch_dir, batch)


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
