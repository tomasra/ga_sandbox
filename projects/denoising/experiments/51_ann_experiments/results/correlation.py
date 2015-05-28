#!/usr/bin/env python
#-*- coding: utf-8 -*-

import json
import os
import csv
import numpy as np
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
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D


OCR_LEVEL_FILES = {
    'high': '/home/tomas/Masters/4_semester/synthetic_tests/high_ocr.txt',
    'medium': '/home/tomas/Masters/4_semester/synthetic_tests/medium_ocr.txt',
    'low': '/home/tomas/Masters/4_semester/synthetic_tests/low_ocr.txt',
}

def correlation(results):
    for level in ['high', 'medium', 'low', None]:
        level_images = []
        if level is not None:
            with open(OCR_LEVEL_FILES[level], 'r') as fp:
                level_images = json.load(fp)

        p_ocr, p_q, p_mse = [], [], []
        for result in results:
            if result['image']['name'].startswith('noisy-00-00-'):
                continue
            if level is not None:
                if result['image']['name'] not in level_images:
                    continue
            # p_ocr += result['metrics_absolute']['ocr']
            for res_ocr in result['metrics_absolute']['ocr']:
                p_ocr.append(res_ocr - result['metrics_initial']['ocr'])
            # p_ocr += result['metrics_absolute']['ocr'] - result['metrics_initial']['ocr']
            p_q += result['metrics_relative']['q']
            p_mse += result['metrics_relative']['mse']
        corr_q_ocr = spearmanr(p_q, p_ocr)
        corr_q_mse = spearmanr(p_q, p_mse)
        corr_ocr_mse = spearmanr(p_ocr, p_mse)
        if level is not None:
            print "--- LEVEL: " + level
        else:
            print "--- LEVEL: all"
        print "Q-OCR: " + str(corr_q_ocr)
        print "Q-MSE: " + str(corr_q_mse)
        print "OCR-MSE: " + str(corr_ocr_mse)
        # print len(p_q), len(p_ocr), len(p_mse)
