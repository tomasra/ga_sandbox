#!/usr/bin/env python
import os
import argparse
import json
import itertools

_SET_NAME = 'set'
_START_ID = 1

MAX_ITERATIONS = 1000
ITERATIONS_BETWEEN_REPORTS = 10
DESIRED_ERROR = 0.0
LEARNING_RATE = 1.0
ACTIVATION_FUNCTION = 'sigmoid'
TRAINING_ALGORITHM = 'quickprop'

PATCH_SIZES = [3, 5, 7, 9, 11]
HIDDEN_NEURONS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_dir', help='Directory for storing parameter sets')
    args = parser.parse_args()

    id = _START_ID
    for patch_size, hidden_neurons in itertools.product(PATCH_SIZES, HIDDEN_NEURONS):
        params = {}
        params['max_iterations'] = MAX_ITERATIONS
        params['iterations_between_reports'] = ITERATIONS_BETWEEN_REPORTS
        params['desired_error'] = DESIRED_ERROR
        params['learning_rate'] = LEARNING_RATE
        params['activation_function'] = ACTIVATION_FUNCTION
        params['training_algorithm'] = TRAINING_ALGORITHM

        params['patch_size'] = patch_size
        params['hidden_neurons'] = hidden_neurons

        param_set_name = _SET_NAME + '-' + str(id) + '.json'
        params['id'] = _SET_NAME + '-' + str(id)
        param_set_path = os.path.join(args.output_dir, param_set_name)
        with open(param_set_path, 'w') as fp:
            json.dump(params, fp)
        id += 1
