#!/usr/bin/env python
#-*- coding: utf-8 -*-

import json
import os
import csv
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
METRICS_INITIAL_PATH = '/home/tomas/Masters/4_semester/synthetic_tests/metrics_initial.csv'
CLUSTERS = 3



if __name__ == "__main__":
    csv_file = open(METRICS_INITIAL_PATH, 'r')
    csv_reader = csv.DictReader(csv_file)
    metrics_initial = []
    image_names = []
    for row in csv_reader:
        if row['image'].startswith('noisy-00-00-'):
            continue
        # metrics_initial.append([
        #     float(row['q']),
        #     float(row['ocr']),
        #     float(row['mse']),
        # ])
        metrics_initial.append([float(row['ocr'])])
        image_names.append(row['image'])
    csv_file.close()

    # Clustering
    kmeans = KMeans(n_clusters=CLUSTERS)
    kmeans.fit(metrics_initial)

    groups = {0: [], 1: [], 2: []}
    for idx, label in enumerate(kmeans.labels_):
        groups[label].append(idx)

    means = {}
    for group, indexes in groups.items():
        means[group] = np.mean([
            metrics_initial[idx]
            for idx in indexes
        ])

    sorted_means = sorted([
        (key, value) for key, value in means.items()
    ], key=lambda p: p[1])
    print sorted_means
    
    # Low ocr
    low_ocr_path = os.path.join(
        os.path.dirname(METRICS_INITIAL_PATH),
        'low_ocr.txt')
    with open(low_ocr_path, 'w') as fp:
        names = [
            image_names[idx]
            for idx in groups[sorted_means[0][0]]
        ]
        json.dump(names, fp)

    # Medium ocr
    low_ocr_path = os.path.join(
        os.path.dirname(METRICS_INITIAL_PATH),
        'medium_ocr.txt')
    with open(low_ocr_path, 'w') as fp:
        names = [
            image_names[idx]
            for idx in groups[sorted_means[1][0]]
        ]
        json.dump(names, fp)
    
    # High ocr
    low_ocr_path = os.path.join(
        os.path.dirname(METRICS_INITIAL_PATH),
        'high_ocr.txt')
    with open(low_ocr_path, 'w') as fp:
        names = [
            image_names[idx]
            for idx in groups[sorted_means[2][0]]
        ]
        json.dump(names, fp)
