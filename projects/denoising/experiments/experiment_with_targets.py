# -*- coding: utf-8 -*-
import projects.denoising.run as chars
from projects.denoising.imaging.utils import render_image
from projects.denoising.imaging.char_drawer import CharDrawer
import os
import pickle

RUNS = 10
FITNESS_THRESHOLD = 0.97

#---------------------------------------------------
# Salt-and-pepper
# Known target image
#---------------------------------------------------
source_image = chars.get_snp_noise_char(u'A')
target_image = chars.get_binary_char(u'A')

results = []
for i in xrange(RUNS):
    results += [chars.run(
        source_image,
        target_image,
        fitness_threshold=FITNESS_THRESHOLD,
        elitism=True
    )]


with open('results/snp_with_targets.pickle', 'w') as f:
    pickle.dump(results, f)

#---------------------------------------------------
# Gaussian
# Known target image
#---------------------------------------------------
source_image = chars.get_gaussian_noise_char(u'A')
target_image = chars.get_binary_char(u'A')

results = []
for i in xrange(RUNS):
    results += [chars.run(
        source_image,
        target_image,
        fitness_threshold=FITNESS_THRESHOLD,
        elitism=True
    )]

with open('results/gaussian_with_targets.pickle', 'w') as f:
    pickle.dump(results, f)
