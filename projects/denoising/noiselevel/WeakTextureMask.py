
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

#% Weak-Texture Mask generation
#%
#% msk = WeakTextureMask(img, patchsize, th)
#%
#%Output parameters
#% msk: weak-texture mask. 0 and 1 represent non-weak-texture and weak-texture regions, respectively
#%
#%
#%Input parameters
#% img: input single image
#% th: threshold which is output of NoiseLevel
#% patchsize (optional): patch size (default: 7)
#%
#%Example:
#% img = double(imread('img.png'));
#% patchsize = 7;
#% [nlevel th] = NoiseLevel(img, patchsize);
#% msk = WeakTextureMask(img, patchsize, th);
#% imwrite(uint8(msk*255), 'msk.png');
#% version: 20150203
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Noise Level Estimation:                                       %
#%                                                               %
#% Copyright (C) 2012-2015 Masayuki Tanaka. All rights reserved. %
#%                    mtanaka@ctrl.titech.ac.jp                  %
#%                                                               %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def WeakTextureMask(img, th, patchsize):

    # Local Variables: cha, Xh, imgh, Xtr, img, Xv, kh, m, col, p, s, msk, kv, th, ind, patchsize, imgv, row
    # Function calls: WeakTextureMask, im2col, sum, imfilter, vertcat, zeros, exist, size
    if not exist('patchsize', 'var'):
        patchsize = 7.
    
    
    kh = np.array(np.hstack((-1./2., 0., 1./2.)))
    imgh = imfilter(img, kh, 'replicate')
    imgh = imgh[:,1:matcompat.size(imgh, 2.)-1.,:]
    imgh = imgh*imgh
    kv = kh.conj().T
    imgv = imfilter(img, kv, 'replicate')
    imgv = imgv[1:matcompat.size(imgv, 1.)-1.,:,:]
    imgv = imgv*imgv
    s = matcompat.size(img)
    msk = np.zeros(s)
    for cha in np.arange(1., (s[2])+1):
        m = im2col(img[:,:,int(cha)-1], np.array(np.hstack((patchsize, patchsize))))
        m = np.zeros(matcompat.size(m))
        Xh = im2col(imgh[:,:,int(cha)-1], np.array(np.hstack((patchsize, patchsize-2.))))
        Xv = im2col(imgv[:,:,int(cha)-1], np.array(np.hstack((patchsize-2., patchsize))))
        Xtr = np.sum(vertcat(Xh, Xv))
        p = Xtr<th[int(cha)-1]
        ind = 1.
        for col in np.arange(1., (s[1]-patchsize+1.)+1):
            for row in np.arange(1., (s[0]-patchsize+1.)+1):
                if p[int(ind)-1] > 0.:
                    msk[int(row)-1:row+patchsize-1.,int(col)-1:col+patchsize-1.,int(cha)-1] = 1.
                
                
                ind = ind+1.
                
            
        
    return [msk]