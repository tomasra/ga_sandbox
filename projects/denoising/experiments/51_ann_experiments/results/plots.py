#!/usr/bin/env python
#-*- coding: utf-8 -*-
import re
import os
import json
import csv
import numpy as np
import itertools
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from scipy import stats
from fann2 import libfann
from skimage import io, util
from projects.denoising.neural.filtering import filter_fann
from projects.denoising.imaging.metrics import q_py, ocr_accuracy, mse

import pylab as plt
import matplotlib as mpl
mpl.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
mpl.rc('text', usetex=False)
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D


OCR_LEVEL_FILES = {
    'high': '/home/tomas/Masters/4_semester/synthetic_tests/high_ocr.txt',
    'medium': '/home/tomas/Masters/4_semester/synthetic_tests/medium_ocr.txt',
    'low': '/home/tomas/Masters/4_semester/synthetic_tests/low_ocr.txt',
}

def filter_ocr_level(results, level):
    path = OCR_LEVEL_FILES[level]
    with open(path, 'r') as fp:
        image_names = json.load(fp)
    return [
        result
        for result in results
        if result['image']['name'] in image_names
    ]


def plot_average_ocr_by_image(results, level=None):
    if level is not None:
        results = filter_ocr_level(results, level)
    points = []
    for result in results:
        if result['image']['blur'] == 0.0 and result['image']['noise'] == 0.0:
            continue
        try:
            point = next(
                p for p in points 
                if p['image']['name'] == result['image']['name'])
        except StopIteration:
            point = {
                'image': result['image'],
                'initial_ocr': result['metrics_initial']['ocr'],
                'ocr_gaussian': [],
                'ocr_sigmoid': [],
            }
            points.append(point)
        if result['param_set']['activation_function'] == 'gaussian':
            point['ocr_gaussian'] += result['metrics_absolute']['ocr']
        else:
            point['ocr_sigmoid'] += result['metrics_absolute']['ocr']

    for point in points:
        point['ocr_max'] = np.max(point['ocr_gaussian'] + point['ocr_sigmoid'])
        point['ocr_all'] = np.mean(point['ocr_gaussian'] + point['ocr_sigmoid'])
        # print len(point['ocr_gaussian'] + point['ocr_sigmoid'])
        point['ocr_gaussian'] = np.mean(point['ocr_gaussian'])
        point['ocr_sigmoid'] = np.mean(point['ocr_sigmoid'])
        # print point['ocr_all']

    points = sorted(points, key=lambda p: p['initial_ocr'])
    # poits = sorted(points, key=lambda p: (p['image']['blur'], p['image']['noise'], p['image']['contrast']))
    fig = plt.figure(figsize=(8, 4))
    plt.ylim([-0.1, 1.1])
    plt.plot([p['ocr_gaussian'] for p in points], linewidth=2, label=u'DNT su Gauso a.f.')
    plt.plot([p['ocr_sigmoid'] for p in points], linewidth=2, label=u'DNT su sigmoidine a.f.')
    plt.plot([p['initial_ocr'] for p in points], linewidth=2, label=u'Pradinė OCR reikšmė')
    plt.xlabel(u'Testinio vaizdo eil. nr.')
    plt.ylabel(u'OCR vidurkis')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.show()

    # Improvements
    impr = []
    for point in points:
        # gaussian_impr = point['ocr_gaussian'] - point['initial_ocr']
        # sigmoid_impr = point['ocr_sigmoid'] - point['initial_ocr']
        # impr.append((point['image']['name'], gaussian_impr, sigmoid_impr))
        all_impr = point['ocr_max'] - point['initial_ocr']
        impr.append((point['image']['name'], all_impr))
    
        point['ocr_improvement'] = all_impr
        # print point['image']['name'], gaussian_impr, sigmoid_impr
    # impr = sorted(impr, key=lambda p: (p[1], p[2]), reverse=True)
    points_sorted = sorted(points, key=lambda p: p['ocr_improvement'], reverse=True)
    # points_sorted = sorted(points, key=lambda p: p['ocr_max'], reverse=True)
    # impr = sorted(impr, key=lambda p: p[1], reverse=True)
    
    fig = plt.figure(figsize=(8, 2.5))
    plt.ylim([0.0, 2.5])
    # plt.plot([p['image']['blur'] for p in points_sorted])
    plt.plot([p['ocr_improvement'] for p in points_sorted], linewidth=2)
    plt.plot([p['image']['blur'] for p in points_sorted])
    # plt.scatter(xrange(len(points)), [p['image']['blur'] for p in points_sorted])
    # plt.plot([p['image']['noise'] for p in points_sorted])
    plt.xlabel(u'Maksimalus OCR pagerėjimas (mažėjimo tvarka)')
    plt.ylabel(u'Vaizdo ryškumo parametras')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.show()

    fig = plt.figure(figsize=(8, 2.5))
    plt.ylim([0.0, 1.1])
    # plt.plot([p['image']['blur'] for p in points_sorted])
    plt.plot([p['ocr_improvement'] for p in points_sorted], linewidth=2)
    plt.plot([p['image']['noise'] * 20.0 for p in points_sorted])
    # plt.plot([p['image']['noise'] for p in points_sorted])
    plt.xlabel(u'Maksimalus OCR pagerėjimas (mažėjimo tvarka)')
    plt.ylabel(u'Vaizdo ryškumo parametras')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.show()

    fig = plt.figure(figsize=(8, 2.5))
    plt.ylim([0.0, 1.1])
    # plt.plot([p['image']['blur'] for p in points_sorted])
    plt.plot([p['ocr_improvement'] for p in points_sorted], linewidth=2)
    plt.plot([p['image']['contrast'] for p in points_sorted])
    # plt.plot([p['image']['noise'] for p in points_sorted])
    plt.xlabel(u'Maksimalus OCR pagerėjimas (mažėjimo tvarka)')
    plt.ylabel(u'Vaizdo ryškumo parametras')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.show()

    return points_sorted


def hist_ocr_initial_vs_best_filtered(results, level=None):
    if level is not None:
        results = filter_ocr_level(results, level)
    # Each point:
    # - image name
    # - initial OCR
    # - best filtered OCR
    points = []
    for result in results:
        if result['image']['blur'] == 0.0 and result['image']['noise'] == 0.0:
            continue
        try:
            point = next(
                p for p in points 
                if p['image']['name'] == result['image']['name'])
        except StopIteration:
            point = {
                'image': result['image'],
                'initial_ocr': result['metrics_initial']['ocr'],
                'filtered_ocr': [],
            }
            points.append(point)
        point['filtered_ocr'] += result['metrics_absolute']['ocr']

        # For display in the paper
        if result['image']['name'] == 'noisy-15-004-04':
            if 0.89215686274509809 in result['metrics_absolute']['ocr']:
                print result['param_set']['name'], result['metrics_absolute']['ocr']
    
    initial_ocr, best_filtered_ocr = [], []
    for point in points:
        point['filtered_ocr'] = np.max(point['filtered_ocr'])
        initial_ocr.append(point['initial_ocr'])
        best_filtered_ocr.append(point['filtered_ocr'])

    points = sorted(points, key=lambda p: p['filtered_ocr'] - p['initial_ocr'], reverse=True)
    # Best results
    print
    for p in points[:10]:
        print p
    
    fig = plt.figure(figsize=(8, 4))
    plt.hist(
        (initial_ocr, best_filtered_ocr),
        label=('Pradiniai vaizdai', 'Filtruoti vaizdai (geriausias rezultatas)'))
    plt.grid(True)
    plt.ylabel(u'Testinių vaizdų kiekis')
    plt.xlabel(u'OCR reikšmė')
    plt.xticks(np.arange(0.0, 1.0001, 0.1))
    plt.tight_layout(True)
    plt.legend(loc='upper left')
    plt.show()


def ocr_by_mse(results):
    points = []
    for result in results:
        if result['image']['blur'] == 0.0 and result['image']['noise'] == 0.0:
            continue
        try:
            point = next(
                p for p in points 
                if p['image']['name'] == result['image']['name'])
        except StopIteration:
            point = {
                'image': result['image'],
                'metrics_initial': result['metrics_initial'],
                'filtered_ocr': [],
            }
            points.append(point)
        point['filtered_ocr'] += result['metrics_absolute']['ocr']
    # Best OCR value
    for point in points:
        point['filtered_ocr'] = np.max(point['filtered_ocr'])

    # points = sorted(points, key=lambda p: p['metrics_initial']['mse'])
    x = [p['metrics_initial']['mse'] for p in points]
    y1 = [p['filtered_ocr'] for p in points]
    y2 = [p['metrics_initial']['ocr'] for p in points]
    fig = plt.figure(figsize=(8, 4))
    # plt.plot([p['filtered_ocr'] for p in points])
    plt.scatter(x, y1, color='blue')
    plt.scatter(x, y2, color='red')
    plt.grid(True)
    # plt.ylabel(u'Testinių vaizdų kiekis')
    # plt.xlabel(u'OCR reikšmė')
    # plt.xticks(np.arange(0.0, 1.0001, 0.1))
    plt.tight_layout(True)
    # plt.legend(loc='upper left')
    plt.show()


# def scoring_correlation(results, level=None):
#     if level is not None:
#         results = filter_ocr_level(results, level)



def q_to_ocr_plot(results, param_set=None):
    # fig = plt.figure(figsize=(8, 3))
    fig = plt.figure(figsize=(8, 5))
    contrasts = [0.2, 0.4, 0.6, 0.8, 1.0]
    # contrasts = [0.8]
    colors = itertools.cycle(['r', 'g', 'b', 'y', 'c'])
    plots = []

    # For parabola fitting
    ideal_q = [
        0.047271226048408527,
        0.094269601769361597,
        0.1416107523154358,
        0.1886077067734106,
        0.23588222469506701,
    ]
    parabola_coefs = []


    xs, ys = [], []
    for c_idx, contrast in enumerate(contrasts):
        q, ocr = [], []
        initial = {}
        for result in results:
            if param_set is not None and result['param_set']['name'] != param_set:
                continue
            if result['image']['name'].startswith('noisy-00-00-'):
                continue

            if result['image']['contrast'] != contrast:
                continue
            # ps_name = result['param_set']['activation_function']
            # ps_name += '-' + result['param_set']['name']
            # q.append(result['metrics_relative']['q'])
            for q_val in result['metrics_absolute']['q']:
                # if result['metrics_initial']['q'] > 0.0:
                    # qq = (q_val - result['metrics_initial']['q']) / 0.15
                    # qq = (q_val - result['metrics_initial']['q']) / result['metrics_initial']['q']
                    # q.append(qq)
                q.append(q_val)
                # q.append(q_val - result['metrics_initial']['q'])
                if result['image']['name'] not in initial:
                    initial[result['image']['name']] = (
                        result['metrics_initial']['q'],
                        result['metrics_initial']['ocr'],
                    )
            ocr += result['metrics_absolute']['ocr']

        # Plot filtered
        plot = plt.scatter(q, ocr, color=next(colors), marker='x')
        plots.append(plot)

        # Plot initial
        # qi, ocri = [], []
        # for v1, v2 in initial.values():
        #     qi.append(v1)
        #     ocri.append(v2)

        # xsi = [q for q in qi]
        # ysi = [ideal_q[c_idx]] * len(xsi)

        # xs += xsi
        # ys += ysi
        # plot = plt.scatter(
        #     xsi, ysi,
        #     color=next(colors))
        # plots.append(plot)


        # Fit to parabola with fixed vertex
        # 
        parabola_xa = lambda x, a: a * (x - ideal_q[c_idx])**2 + 1.0
        popt, pcov = curve_fit(
            parabola_xa,
            q,
            ocr)
        print popt
        parabola_coefs.append(popt[0])

        # Plot
        parabola_x = lambda x: parabola_xa(x, popt[0])
        px = np.arange(0.0, 0.30, 0.001)
        py = [parabola_x(x) for x in px]


        # plt.plot(px, py, color='black')

        # poly = np.poly1d(np.polyfit(q, ocr, 3))
        # print poly
        # ocr_fit = poly(np.arange(0.0, 0.3, 0.1))
        # plt.plot(np.arange(0.0, 0.3, 0.1), ocr_fit)

    line = stats.linregress(xs, ys)
    print line

    # plt.xlabel(u'Pradinio vaizdo Q reikšmė')
    # plt.ylabel(u'Idealaus vaizdo Q reikšmė')
    plt.xlabel(u'Q reikšmė')
    plt.ylabel(u'OCR reikšmė')
    plt.legend(
        plots,
        ['c = ' + str(c) for c in contrasts],
        loc='lower right')
    
    # plt.ylim([0, 0.3])
    # plt.xlim([0, 0.30])
    
    plt.ylim([0, 1.0])
    plt.xlim([0, 0.30])
    plt.grid(True)
    plt.tight_layout(True)
    plt.show()

    print ideal_q

    # Try to fit something to initial Q's/parabola coefs
    exp = lambda x, a, b, c, d: -a * np.exp(-b * x + c) + d
    
    popt, pcov = curve_fit(
        exp,
        ideal_q,
        parabola_coefs)
    print popt

    # popt = [
    #     6.58953834,
    #     29.54967305,
    #     6.00895362,
    #     -40.3269125
    # ]

    exp_x = lambda x: exp(x, *popt)
    px = np.arange(0.0, 0.30, 0.001)
    py = [exp_x(x) for x in px]

    # # Draw plot
    fig2 = plt.figure(figsize=(8, 3))
    plt.scatter(ideal_q, parabola_coefs, label=u'Idealūs vaizdai')
    plt.ylim([-800, 0])
    plt.xlim([0, 0.30])
    plt.xlabel(u'Idealaus vaizdo Q reikšmė')
    plt.ylabel(u'Parabolės išlinkio koeficientas')
    plt.plot(px, py, label='Aproksimuotas koeficientas')
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def improvement_plot(results, level=None):
    if level is not None:
        results = filter_ocr_level(results, level)
    points = []
    for result in results:
        if result['image']['blur'] == 0.0 and result['image']['noise'] == 0.0:
            continue
        try:
            point = next(
                p for p in points 
                if p['image']['name'] == result['image']['name'])
        except StopIteration:
            point = {
                'image': result['image'],
                'metrics_initial': result['metrics_initial'],
                'q': [],
                'mse': [],
                'ocr': [],
                'abs_ocr': [],
            }
            points.append(point)
        point['q'] += result['metrics_relative']['q']
        point['mse'] += result['metrics_relative']['mse']
        point['ocr'] += result['metrics_relative']['ocr']
        point['abs_ocr'] += result['metrics_absolute']['ocr']

    for point in points:
        max_ocr = np.max(point['ocr'])
        tmp_q, tmp_mse, tmp_ocr = [], [], []
        for idx, ocr in enumerate(point['ocr']):
            if ocr == max_ocr:
                tmp_q.append(point['q'][idx])
                tmp_mse.append(point['mse'][idx])
                tmp_ocr.append(point['abs_ocr'][idx])
        # print len(tmp_q), len(tmp_mse)
        # print tmp_q

        point['tmp_q'] = tmp_q
        point['tmp_mse'] = tmp_mse
        point['max_abs_ocr'] = np.mean(tmp_ocr)
        point['avg_q'] = np.mean(tmp_q)
        point['avg_mse'] = np.mean(tmp_mse)
        print np.var(tmp_q), np.var(tmp_mse)
        # point['avg_ocr'] = np.mean(point['ocr'])
        # if point['avg_mse'] < -10.0:
        #     import pdb; pdb.set_trace()
        #     pass

    points = sorted(points, key=lambda p: p['metrics_initial']['ocr'])
    # poits = sorted(points, key=lambda p: (p['image']['blur'], p['image']['noise'], p['image']['contrast']))
    fig = plt.figure(figsize=(8, 4))
    # plt.ylim([-0.1, 1.1])
    plt.plot([p['avg_q'] for p in points], linewidth=2, label=u'P_Q')
    plt.plot([p['avg_mse'] for p in points], linewidth=2, label=u'P_MSE')
    # plt.plot([p['max_ocr'] for p in points], linewidth=2, label=u'P_OCR')
    plt.plot([p['metrics_initial']['ocr'] for p in points], linewidth=2, label=u'Pradinis OCR')
    plt.xlabel(u'Testinio vaizdo eil. nr.')
    # plt.ylabel(u'OCR vidurkis')
    plt.ylabel(u'Įverčio reikšmė')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.show()

    print 'Spearman correlation Q-OCR:'
    print [p['max_abs_ocr'] for p in points]
    print spearmanr([p['avg_q'] for p in points], [p['max_abs_ocr'] for p in points])

    # print 'Spearman correlation Q-MSE:'
    # print spearmanr([p['avg_q'] for p in points], [p['avg_mse'] for p in points])

    # print 'Spearman correlation OCR-MSE:'
    # print spearmanr([p['max_abs_ocr'] for p in points], [p['avg_mse'] for p in points])

    return points
