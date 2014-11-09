#!venv/bin/python
# -*- coding: utf-8 -*-
from run_chars import ALL_CHARS
from run_chars import get_clear_char, apply_filters
from run_chars import get_snp_noise_char, get_gaussian_noise_char
from imaging.char_drawer import CharDrawer
from imaging.utils import render_image

WIDTH, HEIGHT = 9, 5

snp_with_target_best = [
    63, 8, 48, 64, 67, 21, 58, 13, 26, 4,
    51, 51, 22, 54, 12, 5, 33, 50, 46, 43,
    10, 12, 11, 2, 41, 67, 35, 33, 37, 57
]
gaussian_with_target_best = [
    66, 65, 14, 8, 69, 58, 24, 45, 58, 25,
    39, 29, 54, 23, 12, 32, 67, 27, 41, 30,
    10, 0, 55, 54, 32, 57, 53, 46, 35, 41
]
snp_without_target_best = [
    65, 1, 24, 42, 1, 36, 49, 53, 67, 59,
    65, 5, 52, 12, 55, 24, 2, 19, 34, 48,
    33, 65, 7, 45, 2, 32, 22, 10, 63, 10
]
gaussian_without_target_best = [
    63, 47, 31, 57, 30, 22, 45, 64, 34,
    63, 42, 2, 47, 44, 35, 17, 29, 59, 51,
    19, 1, 68, 25, 30, 54, 49, 55, 57, 67, 47
]

# img = get_clear_char(u'A')
# import pdb; pdb.set_trace()
# render_image(img)

# Original salt and pepper
render_image(
    CharDrawer.create_mosaic(
        [
            get_snp_noise_char(char)
            for char in ALL_CHARS],
        WIDTH, HEIGHT))

# Original gaussian
render_image(
    CharDrawer.create_mosaic(
        [
            get_gaussian_noise_char(char)
            for char in ALL_CHARS],
        WIDTH, HEIGHT))


# Salt and pepper with targets
render_image(
    CharDrawer.create_mosaic(
        [
            apply_filters(
                get_snp_noise_char(char),
                snp_with_target_best
            )
            for char in ALL_CHARS],
        WIDTH, HEIGHT))

# Gaussian with targets
render_image(
    CharDrawer.create_mosaic(
        [
            apply_filters(
                get_gaussian_noise_char(char),
                gaussian_with_target_best
            )
            for char in ALL_CHARS],
        WIDTH, HEIGHT))

# Salt and pepper WITHOUT targets
render_image(
    CharDrawer.create_mosaic(
        [
            apply_filters(
                get_snp_noise_char(char),
                snp_without_target_best
            )
            for char in ALL_CHARS],
        WIDTH, HEIGHT))

# Gaussian WITHOUT targets
render_image(
    CharDrawer.create_mosaic(
        [
            apply_filters(
                get_gaussian_noise_char(char),
                gaussian_without_target_best
            )
            for char in ALL_CHARS],
        WIDTH, HEIGHT))
