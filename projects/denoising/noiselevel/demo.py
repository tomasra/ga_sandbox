
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

img = np.double(plt.imread('lena.png'))
mskflg = 0.
#% if you want to produce the mask, put one for mskflg
level = np.array(np.hstack((5., 10., 20., 40.)))
for i in np.arange(1., (matcompat.size(level, 2.))+1):
    noise = img+np.dot(plt.randn(matcompat.size(img)), level[int(i)-1])
    tic
    [nlevel, th] = NoiseLevel(noise)
    t = toc
    fprintf('True: %5.2f  R:%5.2f G:%5.2f B:%5.2f\n', level[int(i)-1], nlevel[0], nlevel[1], nlevel[2])
    fprintf('Calculation time: %5.2f [sec]\n\n', t)
    if mskflg:
        msk = WeakTextureMask(noise, th)
        imwrite(np.uint8((msk*255.)), sprintf('msk%02d.png', level[int(i)-1]))
    
    
    