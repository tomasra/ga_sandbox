#!venv/bin/python
# -*- coding: utf-8 -*-
import argparse
import pickle
import pylab as plt
import matplotlib
matplotlib.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
import os
import numpy as np


def draw_graph(best_solutions):
    filter_counts = [0] * 70
    for flt_idx in np.concatenate(best_solutions):
        filter_counts[flt_idx] += 1
    # print filter_counts

    plt.bar(xrange(1, len(filter_counts)), filter_counts[1:], align='center')
    plt.xlabel(u'Filtro indeksas')
    plt.ylabel(u'Panaudojimų skaičius')
    plt.xticks(xrange(1, len(filter_counts), 2))
    plt.show()

parser = argparse.ArgumentParser(description='Rezultatų filtrų sekų statistika')
parser.add_argument(
    'pickle_dir', type=str,
    help='Serializuotų .pickle rezultatų failų aplankas')
args = parser.parse_args()

best_solutions = []
for filename in os.listdir(args.pickle_dir):
    if os.path.splitext(filename)[1] == '.pickle':
        print filename
        pickle_filename = os.path.join(args.pickle_dir, filename)
        with open(pickle_filename, 'rU') as f:
            results = pickle.load(f)
            file_best_solutions = []
            for result in results:
                file_best_solutions += [result[2]]
                best_solutions += [result[2]]
                print str(result[2])
            draw_graph(file_best_solutions)

draw_graph(best_solutions)
