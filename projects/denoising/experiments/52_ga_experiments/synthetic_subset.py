#!/usr/bin/env python
#-*- coding: utf-8 -*-

import csv
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
METRICS_INITIAL_PATH = '/home/tomas/Masters/4_semester/synthetic_tests/metrics_initial.csv'
CLUSTERS = 10



if __name__ == "__main__":
    csv_file = open(METRICS_INITIAL_PATH, 'r')
    csv_reader = csv.DictReader(csv_file)
    metrics_initial = []
    image_names = []
    for row in csv_reader:
        if row['image'].startswith('noisy-00-00-'):
            continue
        metrics_initial.append([
            float(row['q']),
            float(row['ocr']),
            float(row['mse']),
        ])
        image_names.append(row['image'])
    csv_file.close()

    # Clustering
    kmeans = KMeans(n_clusters=CLUSTERS)
    kmeans.fit(metrics_initial)

    # Find images closest to the cluster centers
    selected_images = []
    for cluster in xrange(CLUSTERS):
        center = kmeans.cluster_centers_[cluster]
        distances = np.array([
            euclidean(center, metrics_image)
            for metrics_image in metrics_initial
        ])
        min_index = np.where(distances==distances.min())[0]
        selected_images.append(min_index)

    for idx in sorted(selected_images):
        print image_names[idx], metrics_initial[idx]

    # import pdb; pdb.set_trace()
