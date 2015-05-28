#!/usr/bin/env python
#-*- coding: utf-8 -*-
import re
import os
import json
import csv
import numpy as np
from fann2 import libfann
from skimage import io, util
from projects.denoising.neural.filtering import filter_fann
from projects.denoising.imaging.metrics import q_py, ocr_accuracy, mse

import pylab as plt
import matplotlib
matplotlib.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D

import shutil
import random
import tempfile
from fann2 import libfann

def test_anns(results):
    data = read_metrics(results)
    split = int(len(data) * 0.8)
    num_input = len(data[0]) - 1
    random.shuffle(data)
    test_data = data[split:]

    # Read ANNs
    ann_dir = '/home/tomas/Dropbox/Git/ga_sandbox/projects/denoising/neural/trained_anns'
    trained_anns = []
    for filename in os.listdir(ann_dir):
        if filename.endswith('.net'):
            ann_path = os.path.join(ann_dir, filename)
            ann = libfann.neural_net()
            ann.create_from_file(ann_path)
            trained_anns.append(ann)

    points = []
    for row in test_data:
        actual_output = row[num_input]
        ann_mean_output = np.mean([
            ann.run(row[:num_input])
            for ann in trained_anns
        ])
        points.append([ann_mean_output, actual_output])
        print "actual: " + str(actual_output) + ", predicted: " + str(ann_mean_output)

    points = sorted(points, key=lambda p: p[1])
    fig = plt.figure(figsize=(6, 4))
    plt.plot([p[0] for p in points])
    plt.plot([p[1] for p in points])
    plt.ylim([0, 1.2])
    plt.show()


def read_metrics(results):
    data = []
    for result in results:
        if result['image']['name'].startswith('noisy-00-00-'):
            continue
        if result['param_set']['name'] != 'set-19':
            continue
        for run in xrange(10):
            q_noisy = result['metrics_initial']['q']
            q_filtered = result['metrics_absolute']['q'][run]
            ocr_initial = result['metrics_initial']['ocr']
            ocr_filtered = result['metrics_absolute']['ocr'][run]
            data.append((
                q_noisy,
                q_filtered,
                # ocr_initial,
                ocr_filtered,
            ))

    # 3D plot
    # d = zip(*data)
    # x, y, z = tuple(d)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('Q noisy')
    # ax.set_ylabel('Q filtered')
    # ax.set_zlabel('OCR')
    # ax.scatter(x, y, z, c='r', marker='o')
    # plt.show()



    # data.sort(key=lambda d: d[1])

    # fig = plt.figure(figsize=(6, 4))
    # plt.plot([d[0] for d in data])
    # plt.plot([d[1] for d in data])
    # plt.plot([d[2] for d in data])
    #     # plt.plot(
    #     #     [point['mse_median'] for point in points if point['patch_size'] == patch_size],
    #     #     label='Lango dydis: ' + str(patch_size),
    #     #     linewidth=2.0)
    # plt.legend(loc='lower right')
    # # plt.xticks(
    # #     list(xrange(len(hidden_neuron_values))),
    # #     hidden_neuron_values)
    # plt.ylim([0, 10])
    # # plt.xlabel(u'Paslėptų neuronų skaičius')
    # # plt.ylabel(u'Vidurinė MSE įverčio pokyčio reikšmė')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return data


def _get_ann_data_file(data):
    temp_file = tempfile.NamedTemporaryFile()
    num_input = len(data[0]) - 1
    header = "%i %i %i\n" % (
        len(data), num_input, 1)
    temp_file.write(header)

    content = "\n".join([
        ' '.join(["{:1.6f}".format(num) for num in row[:num_input]]) + '\n' + "{:1.6f}".format(row[num_input])
        for row in data
    ])
    temp_file.write(content)
    temp_file.flush()
    return temp_file


def train_and_test_non_ann(results):
    data = read_metrics(results)
    split = int(len(data) * 0.8)
    num_input = len(data[0]) - 1
    random.shuffle(data)

    x, y = [], []
    for q_initial, q_filtered, ocr_filtered in data:
        x.append([q_initial, q_filtered])
        y.append(ocr_filtered)
    # Train
    # n = len(data) / 2
    n = 5000
    clf = svm.SVR(kernel='rbf')
    clf.fit(x[:n], y[:n])
    for idx in xrange(n, n + 10):
        print clf.predict(x[idx]), y[idx]
    return None


# def train_cross_validation(results):
#     data = read_metrics(results)
#     num_input = len(data[0]) - 1
#     random.shuffle(data)
#     # Divide into several sets
#     n = 4
#     split = len(data) / n
#     epochs = 1000

#     archs = [
#         [2, 5, 1],
#         [2, 10, 1],
#         [2, 15, 1],
#         [2, 20, 1],
#         # [2, 10, 10, 1],
#         # [2, 20, 20, 1]
#     ]
#     results = []
#     for arch in archs:
#         for idx in xrange(n):
#             if idx == 0:
#                 test_set = data[:split]
#                 train_set = data[split:]
#             elif idx == (n - 1):
#                 test_set = data[:()]
#                 # .......





def train_and_test(results, iterations, output_dir, name):
    data = read_metrics(results)
    split = int(len(data) * 0.8)
    num_input = len(data[0]) - 1
    random.shuffle(data)

    # # Plot train data output distribution
    # outputs_train = sorted([row[num_input] for row in data[:split]])
    # outputs_test = sorted([row[num_input] for row in data[split:]])
    # fig = plt.figure(figsize=(6, 4))
    # plt.plot(outputs_train)
    # plt.ylim([0, 2])
    # plt.show()

    # fig = plt.figure(figsize=(6, 4))
    # plt.plot(outputs_test)
    # plt.ylim([0, 2])
    # plt.show()

    # return None

    train_data_file = _get_ann_data_file(data[:split])
    test_data_file = _get_ann_data_file(data[split:])

    ann = libfann.neural_net()
    ann.create_standard_array([num_input, 20, 5, 1])
    # ann.set_activation_function_layer(libfann.SIGMOID, 1)
    # ann.set_activation_function_layer(libfann.SIGMOID, 2)

    ann.set_activation_steepness_hidden(1.0)

    # Train
    # TEST TEST TEST
    # train_data = libfann.training_data()
    # train_data.read_train_from_file(train_data_file.name)
    # ann.init_weights(train_data)
    # ann.set_learning_rate(1.0)
    # ann.set_training_algorithm(libfann.TRAIN_INCREMENTAL)
    ann.train_on_file(train_data_file.name, iterations, 10, 0.0)

    # Save results
    ann.save(os.path.join(output_dir, name + '.net'))
    shutil.copy(train_data_file.name, os.path.join(output_dir, name + '_train.data'))
    shutil.copy(test_data_file.name, os.path.join(output_dir, name + '_test.data'))

    # Test
    print "train MSE: " + str(ann.get_MSE())
    test_data = libfann.training_data()
    test_data.read_train_from_file(test_data_file.name)
    ann.reset_MSE()
    ann.test_data(test_data)
    print "test MSE: " + str(ann.get_MSE())
    test_data.destroy_train()

    outputs = sorted([
        (ann.run(row[:num_input]), row[num_input])
        for row in data[split:]
    ], key=lambda p: p[1])

    fig = plt.figure(figsize=(6, 4))
    plt.plot([p[0] for p in outputs])
    plt.plot([p[1] for p in outputs])
    plt.ylim([0, 1.2])
    plt.show()

    train_data_file.close()
    test_data_file.close()
    return None
