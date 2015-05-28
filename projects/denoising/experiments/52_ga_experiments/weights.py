#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import json
import numpy as np
from fann2 import libfann

import pylab as plt
import matplotlib
matplotlib.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D

RESULT_DIR = '/home/tomas/Masters/4_semester/synthetic_tests/results/rprop_1k_gaussian/'
PS_NAME = 'set-19'

if __name__ == "__main__":
    weights = []
    for run in xrange(1, 10 + 1):
        run_dir = os.path.join(RESULT_DIR, str(run))
        for image in os.listdir(run_dir):
            image_dir = os.path.join(run_dir, image)
            ann_path = os.path.join(image_dir, PS_NAME + '.net')
            # Open ANN
            ann = libfann.neural_net()
            ann.create_from_file(ann_path)
            ann_weights = [
                connection[2]
                for connection in ann.get_connection_array()
            ]
            weights.append(ann_weights)
    weights = np.array(weights)

    means = list(np.mean(weights, axis=0))
    variances = list(np.var(weights, axis=0))

    fig = plt.figure(figsize=(8, 3))
    plt.plot(means, linewidth=1)
    plt.xlabel(u'Jungties eil. nr.')
    plt.ylabel(u'Jungties svorių vidurkis')
    # plt.ylim([-10, 10])
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(8, 3))
    plt.plot(variances, linewidth=1)
    plt.xlabel(u'Jungties eil. nr.')
    plt.ylabel(u'Jungties svorių dispersija')
    # plt.ylim([-10, 10])
    plt.tight_layout()
    plt.show()

    with open('/home/tomas/Masters/4_semester/synthetic_tests/results/gaussian-set-19-stats.json', 'w') as fp:
        json.dump({
            'mean': means,
            'var': variances
        }, fp)

    # import pdb; pdb.set_trace()
