#!venv/bin/python
# -*- coding: utf-8 -*-
from imaging.char_drawer import CharDrawer
# from imaging.noises import salt_and_pepper
import imaging.noises as noises
from imaging.utils import render_image

char_drawer = CharDrawer()
bw, color = char_drawer.create_pair('F')
# noisified = noises.salt_and_pepper(color, 0.1)
noisified = noises.gaussian(color, sigma=15.0)
# render_image(bw)
# render_image(color)
# render_image(noisified)
# import pdb; pdb.set_trace()

chars = u"AĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽ0123456789"
pairs = [
    char_drawer.create_pair(char)
    for char in chars
]
bws = [pair[0] for pair in pairs]
colors = [pair[1] for pair in pairs]
colors_sp = noises.salt_and_pepper_all(colors, 0.1)
colors_gaussian = noises.gaussian_all(colors, sigma=20.0)

# bw_mosaic = CharDrawer.create_mosaic(bws, 3, 2)
color_sp_mosaic = CharDrawer.create_mosaic(colors_sp, 10, 6)
color_gaussian_mosaic = CharDrawer.create_mosaic(colors_gaussian, 10, 6)
# import pdb; pdb.set_trace()
# render_image(bw_mosaic)
render_image(color_sp_mosaic)
render_image(color_gaussian_mosaic)
