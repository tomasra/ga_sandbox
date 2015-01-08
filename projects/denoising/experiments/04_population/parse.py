#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pylab as plt
import matplotlib
from collections import OrderedDict
matplotlib.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
from projects.denoising.experiments.experiment import read_results


def make_plots(result_dir):
    # Read JSON
    result_sets = []
    for entry in os.listdir(result_dir):
        path = os.path.abspath(
            os.path.join(result_dir, entry))
        print path
        if os.path.isdir(path):
            result_sets += read_results(path)

    # Population size -> iteration count
    iteration_counts = {}
    for rs in result_sets:
        population_size = rs.parameters.population_size
        iteration_count = rs.results.iterations
        if population_size in iteration_counts:
            iteration_counts[population_size].append(iteration_count)
        else:
            iteration_counts[population_size] = []

    # Analysis
    mean = OrderedDict()
    variance = OrderedDict()
    items = sorted(iteration_counts.items(), key=lambda i: i[0])
    for population_size, counts in items:
        mean[population_size] = np.mean(counts)
        variance[population_size] = np.var(counts)

    # Sort by population size
    # pdb.set_trace()

    # Mean plot
    fig = plt.figure(figsize=(14, 7))
    plt.xlabel(u'Populiacijos dydis')
    plt.ylabel(u'Vidutinis iteracijų skaičius')
    plt.plot(
        [m for m in mean.values()],
        color='red', label=u'Vidurkis')
    plt.legend(loc='lower right')
    plt.savefig('population_mean.png')

    # Variance plot
    fig = plt.figure(figsize=(14, 7))
    plt.xlabel(u'Populiacijos dydis')
    plt.ylabel(u'Vidutinis iteracijų skaičius')
    plt.plot(
        [v for v in variance.values()],
        color='blue', label=u'Dispersija')
    plt.legend(loc='lower right')
    plt.savefig('population_variance.png')

if __name__ == "__main__":
    result_dir = sys.argv[1]
    make_plots(result_dir)
