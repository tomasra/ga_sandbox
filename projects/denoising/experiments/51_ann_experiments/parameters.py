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
ACTIVATION_FUNCTION = 'gaussian'
TRAINING_ALGORITHM = 'rprop'

#PATCH_SIZES = [3, 5, 7, 9, 11]
PATCH_SIZES = [3, 5, 7, 9]
#HIDDEN_NEURONS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
HIDDEN_NEURONS = [5, 10, 15, 20]
HIDDEN_NEURONS2 = [0, 5, 10, 15, 20]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_dir', help='Directory for storing parameter sets')
    args = parser.parse_args()

    id = _START_ID
    for patch_size, hidden_neurons, hidden_neurons2 in itertools.product(PATCH_SIZES, HIDDEN_NEURONS, HIDDEN_NEURONS2):
	if hidden_neurons2 <= hidden_neurons:
	    params = {}
            params['max_iterations'] = MAX_ITERATIONS
            params['iterations_between_reports'] = ITERATIONS_BETWEEN_REPORTS
            params['desired_error'] = DESIRED_ERROR
            params['learning_rate'] = LEARNING_RATE
            params['activation_function'] = ACTIVATION_FUNCTION
            params['training_algorithm'] = TRAINING_ALGORITHM

            params['patch_size'] = patch_size
            params['hidden_neurons'] = hidden_neurons
            if hidden_neurons2 > 0:
                params['hidden_neurons2'] = hidden_neurons2

            param_set_name = _SET_NAME + '-' + str(id) + '.json'
            params['id'] = _SET_NAME + '-' + str(id)
            param_set_path = os.path.join(args.output_dir, param_set_name)
            with open(param_set_path, 'w') as fp:
                json.dump(params, fp)
            id += 1
