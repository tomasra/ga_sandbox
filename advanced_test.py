#!venv/bin/python
# -*- coding: utf-8 -*-
from imaging.char_drawer import CharDrawer
from imaging.utils import render_image, rgb2gray
from imaging.utils import quantize, extract_boundaries, region_sizes
from imaging.utils import connected_regions, char_parameters
import imaging.filters as flt
import imaging.noises as noises

char_drawer = CharDrawer(
    text_color=(0, 180, 0),
    bg_color=(0, 255, 0)
)

# chars = u"AĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽ0123456789"
chars = u"A"
for char in chars:
    char_clear = char_drawer.create_pair(char)[1]
    char_noisy = noises.salt_and_pepper(char_clear, 0.2)

    # params = char_parameters(char_clear)
    params = char_parameters(char_noisy)
    print params
