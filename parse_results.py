#!venv/bin/python
# -*- coding: utf-8 -*-
import argparse
import pickle
import pylab as plt
import matplotlib
matplotlib.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
import os

parser = argparse.ArgumentParser(description='Rezultatų atvaizdavimas')
parser.add_argument(
    'filename', type=str,
    help='Serializuotas .pickle rezultatų failas')
args = parser.parse_args()

with open(args.filename, 'rU') as f:
    results = pickle.load(f)

filename_no_ext = os.path.splitext(args.filename)[0]
# print filename_no_ext
for result in results:
    print result[2]

#-------------------------------------------------------
#-- Iteration graph
#-------------------------------------------------------
iterations = [
    len(result[0])
    for result in results
]

# fig = plt.figure()
plt.bar(xrange(1, len(iterations) + 1), iterations, align='center')
plt.xlabel(u'Vykdymo eil. nr.')
plt.ylabel(u'Iteracijų skaičius')
plt.xticks(xrange(1, len(iterations) + 1))
plt.savefig(filename_no_ext + '_iterations.png')


#-------------------------------------------------------
#-- Filtered characters
#-------------------------------------------------------
from imaging.char_drawer import CharDrawer
import imaging.utils as iu

filtered_images = [
    result[3]
    for result in results
]

mosaic = CharDrawer.create_mosaic(
    filtered_images, 10, 1)
iu.render_image(mosaic, filename_no_ext + '_images.png')


#-------------------------------------------------------
#-- Best case average/best fitness graph
#-------------------------------------------------------
idx_min_iterations = iterations.index(min(iterations))
avg_fitnesses = results[idx_min_iterations][0]
best_fitnesses = results[idx_min_iterations][1]

fig = plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.xlabel(u'Iteracija')
plt.ylabel(u'Fitneso funkcijos reikšmė')
plt.plot(best_fitnesses, color='red', label=u'Geriausias individas')
plt.plot(avg_fitnesses, color='blue', label=u'Populiacijos vidurkis')
plt.legend(loc='lower right')
# plt.show()


#-------------------------------------------------------
#-- Worst case average/best fitness graph
#-------------------------------------------------------
idx_max_iterations = results.index(sorted(
    # Results with maximum number of iterations
    [
        result
        for result in results
        if len(result[0]) == max(iterations)
    ],
    # Sort by best fitness, in ascending order
    key=lambda r: r[1][len(r[1]) - 1]
)[0])

avg_fitnesses = results[idx_max_iterations][0]
best_fitnesses = results[idx_max_iterations][1]
# print best_fitnesses

plt.subplot(1, 2, 2)
plt.xlabel(u'Iteracija')
plt.ylabel(u'Fitneso funkcijos reikšmė')
plt.plot(best_fitnesses, color='red', label=u'Geriausias individas')
plt.plot(avg_fitnesses, color='blue', label=u'Populiacijos vidurkis')
plt.legend(loc='lower right')
# plt.show()
plt.savefig(filename_no_ext + '_best_worst.png')
