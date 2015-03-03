
import numpy as np
import scipy
import matcompat
from skimage import io
from skimage import data
from skimage import util

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

from NoiseLevel import NoiseLevel

# img = io.imread('lena.png')
img = data.lena()

# img = img[99:150,99:150,:]
# img = img[49:100,49:100,:]

mskflg = 0.
#% if you want to produce the mask, put one for mskflg
# level = np.array(np.hstack((0, 5, 10, 20, 40)))
levels = [5, 10, 20, 40]
# levels = [5]
for level in levels:
    noisy_image = util.img_as_ubyte(
        util.random_noise(img, mode='gaussian', var=pow(float(level) / 256.0, 2))
    )
    nlevel, th, num = NoiseLevel(noisy_image)
    # nlevel, th, num = NoiseLevel(img)

    # fprintf('True: %5.2f  R:%5.2f G:%5.2f B:%5.2f\n', level[int(i)-1], nlevel[0], nlevel[1], nlevel[2])
    
    # true_sigma = float(level) / 256.0
    print 'True: %5.2f  R:%5.2f G:%5.2f B:%5.2f' % (level, nlevel[0], nlevel[1], nlevel[2])
    # if mskflg:
    #     msk = WeakTextureMask(noise, th)
    #     imwrite(np.uint8((msk*255.)), sprintf('msk%02d.png', level[int(i)-1]))
