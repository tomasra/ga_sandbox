#!venv/bin/python
# -*- coding: utf-8 -*-
import run_chars as chars
from imaging.utils import render_image
from imaging.char_drawer import CharDrawer

RUNS = 5

# Salt-and-pepper
source_image = chars.get_snp_noise_char(u'A')
target_image = chars.get_binary_char(u'A')

render_image(source_image)
solution_images = []
for i in xrange(RUNS):

    (
        average_fitnesses,
        best_fitnesses,
        best_solution,
        best_solution_image
    ) = chars.run(
        source_image,
        target_image=target_image,
        elitism=True)
    solution_images += [best_solution_image]

render_image(
    CharDrawer.create_mosaic(
        solution_images,
        5, 1
    )
)

# Gaussian
