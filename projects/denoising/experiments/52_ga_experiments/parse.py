#!/usr/bin/env python
#-*- coding: utf-8 -*-
import re
import os
import json
import csv
import numpy as np
import itertools
from scipy.optimize import curve_fit
from scipy import stats
from fann2 import libfann
from skimage import io, util
from projects.denoising.neural.filtering import filter_fann
from projects.denoising.imaging.metrics import q_py, ocr_accuracy, mse

import pylab as plt
import matplotlib as mpl
mpl.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D


OCR_LEVEL_FILES = {
    'high': '/home/tomas/Masters/4_semester/synthetic_tests/high_ocr.txt',
    'medium': '/home/tomas/Masters/4_semester/synthetic_tests/medium_ocr.txt',
    'low': '/home/tomas/Masters/4_semester/synthetic_tests/low_ocr.txt',
}


def recreate_images(result_dir, noisy_image_dir):
    for run in os.listdir(result_dir):
        run_dir = os.path.join(result_dir, run)
        for filename in os.listdir(run_dir):
            if filename.endswith('.net'):
                ann_path = os.path.join(run_dir, filename)
                image_name = os.path.splitext(filename)[0] + '.png'
                noisy_path = os.path.join(noisy_image_dir, image_name)
                result_path = os.path.join(run_dir, image_name)
                filter_image(ann_path, noisy_path, result_path)


def _parse_image_name(image_name):
    """
    Split image name into its parameter values
    """
    blur_values = {'00': 0.0, '05': 0.5, '10': 1.0, '15': 1.5, '20': 2.0}
    noise_values = {'00': 0.0, '001': 0.01, '002': 0.02, '003': 0.03, '004': 0.04, '005': 0.05}
    contrast_values = {'02': 0.2, '04': 0.4, '06': 0.6, '08': 0.8, '10': 1.0}
    parts = image_name.split('-')
    return blur_values[parts[1]], noise_values[parts[2]], contrast_values[parts[3]]


def run_metrics(result_dir, metrics_initial_path, noisy_image_dir):
    # Add initial metric values
    csv_file = open(metrics_initial_path, 'r')
    csv_reader = csv.DictReader(csv_file)
    metrics_initial = {}
    for row in csv_reader:
        metrics_initial[row['image']] = {
            'q': float(row['q']),
            'ocr': float(row['ocr']),
            'mse': float(row['mse'])
        }
    csv_file.close()

    results = []
    DEFAULT_TEXT = 'Lorem ipsum\ndolor sit amet,\nconsectetur\n\nadipiscing elit.\n\nDonec vel\naliquet velit,\nid congue\nposuere.'
    runs = list(xrange(1, 10 + 1, 1))
    for run in runs:
        # print "--- RUN " + run
        run_dir = os.path.join(result_dir, str(run))
        for filename in os.listdir(run_dir):
            if not filename.endswith('.png'):
                continue
            image_name = os.path.splitext(filename)[0]
            blur, noise, contrast = _parse_image_name(image_name)
            image_path = os.path.join(run_dir, filename)
            image = util.img_as_float(io.imread(image_path))
            ocr = ocr_accuracy(image, DEFAULT_TEXT)
            try:
                result = next(
                    r for r in results
                    if r['image']['name'] == image_name)
            except StopIteration:
                result = {
                    'image': {
                        'name': image_name,
                        'blur': blur,
                        'noise': noise,
                        'contrast': contrast
                    },
                    'metrics_initial': metrics_initial[image_name],
                    'ocr': [],
                    'q': [],
                    'mse': [],
                }
                results.append(result)
            result['ocr'].append(ocr)
            result['q'].append(q_py(image))

            # MSE
            ideal_image_name = 'noisy-00-00-' + str(contrast).replace('.', '') + '.png'
            ideal_image_path = os.path.join(noisy_image_dir, ideal_image_name)
            ideal_image = util.img_as_float(io.imread(ideal_image_path))
            mse_val = mse(ideal_image, image)
            result['mse'].append(mse_val)
    return results


def plot_ga_ocr(results):
    ordering = [
        'noisy-00-003-06',
        'noisy-05-002-10',
        'noisy-10-001-08',
        'noisy-10-004-04',
        'noisy-15-00-08',
        'noisy-15-003-10',
        'noisy-20-002-06',
        'noisy-20-003-04',
        'noisy-20-003-08',
        'noisy-20-003-10',
    ]
    results = sorted(results, key=lambda p: ordering.index(p['image']['name']))
    fig = plt.figure(figsize=(6, 4))
    initial_ocr = [r['metrics_initial']['ocr'] for r in results]
    best_ocr = [np.max(r['ocr']) for r in results]
    average_ocr = [np.mean(r['ocr']) for r in results]
    plt.plot(initial_ocr, linewidth=2, label=u'Pradinis OCR')
    plt.plot(best_ocr, linewidth=2, label=u'Didžiausias OCR po filtravimo')
    plt.plot(average_ocr, linewidth=2, label=u'Vidutinis OCR po filtravimo')
    plt.grid(True)
    plt.ylabel(u'OCR reikšmė')
    plt.xlabel(u'Testinio vaizdo eil. nr.')
    plt.xticks(xrange(0, 10, 1), xrange(1, 11, 1))
    plt.legend(loc='upper right')
    plt.ylim([0, 1.2])
    plt.tight_layout(True)
    plt.show()
    return results


def read_results(result_dir):
    results = []
    runs = list(xrange(1, 10 + 1, 1))
    for run in runs:
        run_dir = os.path.join(result_dir, str(run))
        for filename in sorted(os.listdir(run_dir)):
            if filename.endswith('.json'):
                filepath = os.path.join(run_dir, filename)
                with open(filepath, 'r') as fp:
                    result = json.load(fp)
                results.append(result)
    return results


def plot_runtimes(results):
    runtimes = [r['results']['run_time'] for r in results]
    fig = plt.figure(figsize=(6, 4))
    plt.plot(runtimes)
    plt.show()
    print np.mean(runtimes)
    return None
