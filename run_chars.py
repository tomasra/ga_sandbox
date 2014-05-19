#!venv/bin/python
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

chars = "ABCDEF"
pairs = [
    char_drawer.create_pair(char)
    for char in chars
]
bws = [pair[0] for pair in pairs]
colors = [pair[1] for pair in pairs]

bw_mosaic = CharDrawer.create_mosaic(bws, 3, 2)
color_mosaic = CharDrawer.create_mosaic(colors, 2, 3)

# import pdb; pdb.set_trace()

render_image(bw_mosaic)
render_image(color_mosaic)
