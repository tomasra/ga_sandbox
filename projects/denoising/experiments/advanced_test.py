#!venv/bin/python
# -*- coding: utf-8 -*-

# # Add the parent directory to PYTHONPATH
# import sys
# import os.path as path
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from projects.denoising.imaging.char_drawer import CharDrawer
from projects.denoising.imaging.utils import render_image, rgb2gray
from projects.denoising.imaging.utils import quantize, extract_boundaries, region_sizes
from projects.denoising.imaging.utils import connected_regions, char_parameters
from projects.denoising.imaging.utils import histogram
import projects.denoising.imaging.filters as flt
import projects.denoising.imaging.noises as noises
import projects.denoising.imaging.noises as noises
from projects.denoising.imaging.filter_call import FilterCall
from projects.denoising.solution import FilterSequenceEvaluator

char_drawer = CharDrawer(
    text_color=(0, 180, 0),
    bg_color=(0, 255, 0)
)

sequence = [23, 8, 38, 63, 19, 11, 53, 0, 4, 0, 58, 42, 6, 0, 52, 47, 65, 43, 56, 54]

# chars = u"AĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽ0123456789"
chars = u"A"
for char in chars:
    char_clear = char_drawer.create_pair(char)[1]
    # char_noisy = noises.salt_and_pepper(char_clear, 0.2)
    char_noisy = noises.gaussian(char_clear, sigma=40.0)

    print histogram(char_clear)
    print histogram(char_noisy)

    render_image(char_noisy)


    # import numpy as np
    # hist = np.histogram(char_clear[0])
    # import pdb; pdb.set_trace()

    evaluator = FilterSequenceEvaluator(
        FilterCall.all_calls(),
        [char_noisy],
        [char_clear]
    )
    filtered_image = evaluator._filter_image(char_noisy, sequence)

    params_clear = char_parameters(char_clear)
    params_noisy = char_parameters(char_noisy)
    params_filtered = char_parameters(filtered_image)
    # print params
    # print params_noisy
    print params_filtered
    print evaluator._fitness_unknown_target(char_noisy, sequence)

    render_image(filtered_image)
