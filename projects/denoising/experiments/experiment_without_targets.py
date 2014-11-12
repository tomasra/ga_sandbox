# -*- coding: utf-8 -*-
import projects.denoising.run as chars
from projects.denoising.imaging.utils import render_image
from projects.denoising.imaging.char_drawer import CharDrawer
import pickle

# from imaging.utils import histogram, histogram_diff, ideal_histogram
# from imaging.utils import max_histogram_diff

# target_image = chars.get_binary_char(u'A')
# source_image = chars.get_snp_noise_char(u'A')

# target_hist = histogram(target_image)
# source_hist = histogram(source_image)
# template = ideal_histogram(target_image, 0.1)


# actual_diff = histogram_diff(target_hist, template)
# max_diff = max_histogram_diff(target_image)
# print histogram_diff(target_hist, template)
# print max_histogram_diff(target_image)
# print (1 - float(actual_diff) / max_diff) / 1
# print histogram_diff(target_hist, source_hist)
# print histogram_diff(source_hist, template)

# import pdb; pdb.set_trace()


import projects.denoising.imaging.utils as iu


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
        target_image=None,
        fitness_threshold=FITNESS_THRESHOLD,
        elitism=True
    )]
    # best_solution_image = results[len(results) - 1][3]
    # ideal_histogram = iu.ideal_histogram(best_solution_image, 0.1)
    # actual_hist = iu.histogram(best_solution_image)
    # print iu.histogram_diff(ideal_histogram, actual_hist)
    # import pdb; pdb.set_trace()
    # render_image(results[len(results) - 1][3])


with open('results/snp_without_targets.pickle', 'w') as f:
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
        target_image=None,
        fitness_threshold=FITNESS_THRESHOLD,
        elitism=True
    )]

with open('results/gaussian_without_targets.pickle', 'w') as f:
    pickle.dump(results, f)
