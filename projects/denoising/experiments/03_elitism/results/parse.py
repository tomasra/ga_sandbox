#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pylab as plt
import matplotlib
matplotlib.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
from projects.denoising.experiments.experiment import read_results

# Read JSON
result_sets = []
for entry in os.listdir(os.getcwd()):
    path = os.path.abspath(entry)
    if os.path.isdir(path):
        result_sets += read_results(path)

# Elitism size -> iteration count
iteration_counts = {}
for rs in result_sets:
    elite_size = rs.parameters.elite_size
    iteration_count = rs.results.iterations
    if elite_size in iteration_counts:
        iteration_counts[elite_size].append(iteration_count)
    else:
        iteration_counts[elite_size] = []

# Analysis
mean, variance = {}, {}
for elite_size, counts in iteration_counts.items():
    mean[elite_size] = np.mean(counts)
    variance[elite_size] = np.var(counts)


# Mean plot
fig = plt.figure(figsize=(14, 7))
plt.xlabel(u'Elitizmo reikšmė')
plt.ylabel(u'Vidutinis iteracijų skaičius')
plt.plot(
    [m for m in mean.values()],
    color='red', label=u'Vidurkis')
plt.legend(loc='lower right')
plt.savefig('elitism_mean.png')

# Variance plot
fig = plt.figure(figsize=(14, 7))
plt.xlabel(u'Elitizmo reikšmė')
plt.ylabel(u'Vidutinis iteracijų skaičius')
plt.plot(
    [v for v in variance.values()],
    color='blue', label=u'Dispersija')
plt.legend(loc='lower right')
plt.savefig('elitism_variance.png')
