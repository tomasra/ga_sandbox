
import numpy as np
import scipy
import scipy.ndimage
import scipy.ndimage.filters
import scipy.stats
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

#% NoiseLevel estimates noise level of input single noisy image.
#%
#% [nlevel th num] = NoiseLevel(img,patchsize,decim,conf,itr)
#%
#%Output parameters
#% nlevel: estimated noise levels. 
#% th: threshold to extract weak texture patches at the last iteration.
#% num: number of extracted weak texture patches at the last iteration.
#% 
#% The dimension output parameters is same to channels of the input image. 
#%
#%
#%Input parameters
#% img: input single image
#% patchsize (optional): patch size (default: 7)
#% decim (optional): decimation factor. If you put large number, the calculation will be accelerated. (default: 0)
#% conf (optional): confidence interval to determin the threshold for the weak texture. In this algorithm, this value is usually set the value very close to one. (default: 0.99)
#% itr (optional): number of iteration. (default: 3)
#%
#%Example:
#% img = double(imread('img.png'));
#% nlevel = NoiseLevel(img);
#%
#%Reference:
#% Xinhao Liu, Masayuki Tanaka and Masatoshi Okutomi
#% Noise Level Estimation Using Weak Textured Patches of a Single Noisy Image
#% IEEE International Conference on Image Processing (ICIP), 2012.
#%
#% Xinhao Liu, Masayuki Tanaka and Masatoshi Okutomi
#% Single-Image Noise Level Estimation for Blind Denoising Noisy Image
#% IEEE Transactions on Image Processing, Vol.22, No.12, pp.5226-5237, December, 2013.
#%
#% version: 20150203
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Noise Level Estimation:                                       %
#%                                                               %
#% Copyright (C) 2012-2015 Masayuki Tanaka. All rights reserved. %
#%                    mtanaka@ctrl.titech.ac.jp                  %
#%                                                               %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def patchify(img, patch_shape):
    # http://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape
    x, y = patch_shape
    shape = ((X-x+1), (Y-y+1), x, y) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
    #    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y, 1, Y, 1])
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


def im2col(img, patch_shape):
    """
    Matlab's im2col implementation
    """
    # Need to transpose image, otherwise result is different from matlab's im2col
    patch_shape = (patch_shape[1], patch_shape[0])
    patches = patchify(img.transpose(), patch_shape)
    num_patches = patches.shape[0] * patches.shape[1]
    patch_size = patch_shape[0] * patch_shape[1]
    # 4D to 3D array
    cols = patches.reshape((num_patches, patch_shape[0], patch_shape[1]))
    # Final array of cols
    cols = cols.reshape(num_patches, patch_size)
    return cols.transpose()


def imfilter_h(img, kernel):
    """
    Horizontal 1D convolution
    """
    if len(img.shape) == 3:
        # Need to iterate color channels
        channels = []
        for channel_idx in xrange(img.shape[2]):
            channel = np.array([
                scipy.ndimage.filters.correlate(row, kernel, mode='nearest')
                for row in img[:,:,channel_idx]
            ])
            channels.append(channel)
        result = np.dstack(tuple(channels))
    else:
        # Grayscale image?
        result = np.array([
            scipy.ndimage.filters.correlate(row, kernel, mode='nearest')
            for row in image
        ])
    return result


def imfilter_v(img, kernel):
    """
    Vertical 1D convolution
    """
    # Same as horizontal convolution, but done on transposed image
    return imfilter_h(img.swapaxes(0, 1), kernel).swapaxes(0, 1)


def NoiseLevel(img, patchsize=7, decim=0, conf=1-(1E-6), itr=3):
    # Image has a single channel?
    if len(img.shape) == 2:
        new_shape = list(img.shape) + [1]
        img = img.reshape(tuple(new_shape))

    img = img.astype(np.float64)
    nlevel, th, num = [], [], []
    
    kernel = [-0.5, 0, 0.5]
    imgh = imfilter_h(img, kernel)
    imgh = imgh[:,1:matcompat.size(imgh, 2.)-1.,:]
    imgh = imgh*imgh

    # imgv = imfilter(img, kv, 'replicate')
    # imgv = scipy.ndimage.correlate(img, kv, mode='nearest')
    imgv = imfilter_v(img, kernel)
    imgv = imgv[1:matcompat.size(imgv, 1.)-1.,:,:]
    imgv = imgv*imgv

    kh = np.array([[-1./2.], [0.], [1./2.]])
    kv = kh.T
    Dh = my_convmtx2(kh, patchsize)
    Dv = my_convmtx2(kv, patchsize)

    DD = np.dot(Dh.conj().T, Dh)+np.dot(Dv.conj().T, Dv)
    r = np.linalg.matrix_rank(DD)
    Dtr = np.trace(DD)

    tau0 = scipy.stats.gamma.ppf(
        conf,
        (np.double(r)/2.),
        scale=(2.0 * Dtr / np.double(r)))

    for cha in np.arange(1., (matcompat.size(img, 3.))+1):
        X = im2col(img[:, :, int(cha) - 1], (patchsize, patchsize))
        Xh = im2col(imgh[:, :, int(cha) - 1], (patchsize, patchsize - 2))
        Xv = im2col(imgv[:, :, int(cha) - 1], (patchsize - 2, patchsize))
        
        # Sum columns
        Xtr = np.sum(np.vstack((Xh, Xv)), 0)
        # import pdb; pdb.set_trace()

        if decim > 0:
            # This is not executed by default
            XtrX = vertcat(Xtr, X)
            XtrX = sortrows(XtrX.conj().T).conj().T
            p = np.floor(matdiv(matcompat.size(XtrX, 2.), decim+1.))
            p = np.dot(np.array(np.hstack((np.arange(1., (p)+1)))), decim+1.)
            Xtr = XtrX[0,int(p)-1]
            X = XtrX[1:matcompat.size(XtrX, 1.),int(p)-1]
        

        #%%%%% noise level estimation %%%%%
        tau = float('inf')
        if matcompat.size(X, 2.)<matcompat.size(X, 1.):
            sig2 = 0.
        else:
            cov = np.dot(X, X.T) / (matcompat.size(X, 2.)-1.)
            d = np.linalg.eigvals(cov)
            # MATLAB's 'eig' sorts values in asc order!
            # sig2 = d[0]
            sig2 = sorted(d)[0]
        
        for i in np.arange(2., (itr)+1):
            tau=sig2 * tau0
            p=(Xtr < tau)
            Xtr = Xtr[p]
            X = X.T[p].T
            if (matcompat.size(X,2) < matcompat.size(X,1)):
                break
            cov=np.dot(X, X.T) / (matcompat.size(X,2) - 1)
            d=np.linalg.eigvals(cov)
            sig2 = sorted(d)[0]

        nlevel.append(np.sqrt(sig2))
        th.append(tau)
        num.append(matcompat.size(X, 2.))

    return nlevel, th, num


def my_convmtx2(kernel, patch_size):
    """
    TODO: figure out what the code below actually does, lol
    """
    s = kernel.shape
    m = patch_size - s[0] + 1
    n = patch_size - s[1] + 1
    mtx = np.zeros((m * n, patch_size ** 2))

    k = 0
    for i in xrange(0, m):
        for j in xrange(0, n):
            for p in xrange(0, s[0]):
                l1 = (i + p) * patch_size + j
                l2 = l1 + s[1]
                mtx[k, l1:l2] = kernel[p, :]
            k += 1
    return mtx
