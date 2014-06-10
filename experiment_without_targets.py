#!venv/bin/python
# -*- coding: utf-8 -*-
import run_chars as chars
from imaging.utils import render_image
from imaging.char_drawer import CharDrawer
import pickle

from imaging.utils import histogram, histogram_diff, ideal_histogram
from imaging.utils import max_histogram_diff

target_image = chars.get_binary_char(u'A')
source_image = chars.get_snp_noise_char(u'A')

target_hist = histogram(target_image)
source_hist = histogram(source_image)
template = ideal_histogram(target_image, 0.1)


actual_diff = histogram_diff(target_hist, template)
max_diff = max_histogram_diff(target_image)
print histogram_diff(target_hist, template)
print max_histogram_diff(target_image)
print (1 - float(actual_diff) / max_diff) / 1
# print histogram_diff(target_hist, source_hist)
# print histogram_diff(source_hist, template)

# import pdb; pdb.set_trace()
