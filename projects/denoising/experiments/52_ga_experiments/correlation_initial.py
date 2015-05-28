#!/usr/bin/env python
#-*- coding: utf-8 -*-

import json
import os
import csv
import numpy as np
from scipy.stats import spearmanr
METRICS_INITIAL_PATH = '/home/tomas/Masters/4_semester/synthetic_tests/metrics_initial.csv'

if __name__ == "__main__":
    csv_file = open(METRICS_INITIAL_PATH, 'r')
    csv_reader = csv.DictReader(csv_file)
    image_names = []
    metric_q, metric_ocr, metric_mse = [], [], []
    for row in csv_reader:
        if row['image'].startswith('noisy-00-00-'):
            continue
        metric_q.append(float(row['q']))
        metric_ocr.append(float(row['ocr']))
        metric_mse.append(float(row['mse']))
        image_names.append(row['image'])
    csv_file.close()

    # Overall correlation
    print '--- OVERALL ---'
    print 'Q-OCR: ', str(spearmanr(metric_q, metric_ocr))
    print 'Q-MSE', str(spearmanr(metric_q, metric_mse))
    print 'OCR-MSE', str(spearmanr(metric_ocr, metric_mse))
    print 'Average OCR: ', np.mean([m for idx, m in enumerate(metric_ocr)])


    # High ocr correlation
    high_ocr_path = os.path.join(
        os.path.dirname(METRICS_INITIAL_PATH),
        'high_ocr.txt')
    with open(high_ocr_path, 'r') as fp:
        high_ocr = json.load(fp)
    high_ocr_indexes = []
    for idx, name in enumerate(image_names):
        if name in high_ocr:
            high_ocr_indexes.append(idx)
    high_ocr_q = [m for idx, m in enumerate(metric_q) if idx in high_ocr_indexes]
    high_ocr_ocr = [m for idx, m in enumerate(metric_ocr) if idx in high_ocr_indexes]
    high_ocr_mse = [m for idx, m in enumerate(metric_mse) if idx in high_ocr_indexes]
    print '--- HIGH OCR ---'
    print 'Q-OCR: ', str(spearmanr(high_ocr_q, high_ocr_ocr))
    print 'Q-MSE', str(spearmanr(high_ocr_q, high_ocr_mse))
    print 'OCR-MSE', str(spearmanr(high_ocr_ocr, high_ocr_mse))
    print 'Average OCR: ', np.mean([m for idx, m in enumerate(metric_ocr) if idx in high_ocr_indexes])
    # import pdb; pdb.set_trace()


    medium_ocr_path = os.path.join(
        os.path.dirname(METRICS_INITIAL_PATH),
        'medium_ocr.txt')
    with open(medium_ocr_path, 'r') as fp:
        medium_ocr = json.load(fp)
    medium_ocr_indexes = []
    for idx, name in enumerate(image_names):
        if name in medium_ocr:
            medium_ocr_indexes.append(idx)
    medium_ocr_q = [m for idx, m in enumerate(metric_q) if idx in medium_ocr_indexes]
    medium_ocr_ocr = [m for idx, m in enumerate(metric_ocr) if idx in medium_ocr_indexes]
    medium_ocr_mse = [m for idx, m in enumerate(metric_mse) if idx in medium_ocr_indexes]
    print '--- MEDIUM OCR ---'
    print 'Q-OCR: ', str(spearmanr(medium_ocr_q, medium_ocr_ocr))
    print 'Q-MSE', str(spearmanr(medium_ocr_q, medium_ocr_mse))
    print 'OCR-MSE', str(spearmanr(medium_ocr_ocr, medium_ocr_mse))
    print 'Average OCR: ', np.mean([m for idx, m in enumerate(metric_ocr) if idx in medium_ocr_indexes])


    low_ocr_path = os.path.join(
        os.path.dirname(METRICS_INITIAL_PATH),
        'low_ocr.txt')
    with open(low_ocr_path, 'r') as fp:
        medium_ocr = json.load(fp)
    low_ocr_indexes = []
    for idx, name in enumerate(image_names):
        if name in medium_ocr:
            low_ocr_indexes.append(idx)
    low_ocr_q = [m for idx, m in enumerate(metric_q) if idx in low_ocr_indexes]
    low_ocr_ocr = [m for idx, m in enumerate(metric_ocr) if idx in low_ocr_indexes]
    low_ocr_mse = [m for idx, m in enumerate(metric_mse) if idx in low_ocr_indexes]
    print '--- LOW OCR ---'
    print 'Q-OCR: ', str(spearmanr(low_ocr_q, low_ocr_ocr))
    print 'Q-MSE', str(spearmanr(low_ocr_q, low_ocr_mse))
    print 'OCR-MSE', str(spearmanr(low_ocr_ocr, low_ocr_mse))
    print 'Average OCR: ', np.mean([m for idx, m in enumerate(metric_ocr) if idx in low_ocr_indexes])


    # Correlation by contrast
    contrasts = ['02', '04', '06', '08', '10']
    for contrast in contrasts:
        q, ocr, mse = [], [], []
        for idx, image_name in enumerate(image_names):
            if image_name.endswith(contrast):
                q.append(metric_q[idx])
                ocr.append(metric_ocr[idx])
                mse.append(metric_mse[idx])
        print '-- Contrast: ', contrast
        print 'Q-OCR: ', str(spearmanr(q, ocr))
        print 'Q-MSE', str(spearmanr(q, mse))
        print 'OCR-MSE', str(spearmanr(ocr, mse))        
