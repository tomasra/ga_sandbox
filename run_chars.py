#!venv/bin/python
from imaging.char_drawer import CharDrawer
# from imaging.noises import salt_and_pepper
import imaging.noises as noises
from imaging.utils import render_image

chars = CharDrawer()
bw, color = chars.create_pair('K')
# noisified = noises.salt_and_pepper(color, 0.1)
noisified = noises.gaussian(color, sigma=15.0)
render_image(bw)
render_image(color)
render_image(noisified)
# import pdb; pdb.set_trace()
