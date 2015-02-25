
import numpy as np
import scipy
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
def NoiseLevel(img, patchsize, decim, conf, itr):

    # Local Variables: tau, Xtr, DD, num, conf, Dtr, itr, nlevel, cha, img, decim, tau0, th, Dv, eg, imgh, XtrX, sig2, X, patchsize, imgv, d, Xh, Dh, cov, i, kh, p, r, kv, Xv
    # Function calls: vertcat, sortrows, trace, floor, im2col, double, NoiseLevel, imfilter, rank, gaminv, my_convmtx2, exist, sqrt, eig, Inf, sum, size
    if not exist('itr', 'var'):
        itr = 3.
    
    
    if not exist('conf', 'var'):
        conf = 0.0E-6
    
    
    if not exist('decim', 'var'):
        decim = 0.
    
    
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
    Dh = my_convmtx2(kh, patchsize, patchsize)
    Dv = my_convmtx2(kv, patchsize, patchsize)
    DD = np.dot(Dh.conj().T, Dh)+np.dot(Dv.conj().T, Dv)
    r = np.rank(DD)
    Dtr = np.trace(DD)
    tau0 = gaminv(conf, (np.double(r)/2.), matdiv(np.dot(2.0, Dtr), np.double(r)))
    #%{
    # eg = linalg.eig(DD)
    # tau0 = gaminv(conf, (np.double(r)/2.), np.dot(2.0, eg[int(np.dot(patchsize, patchsize))-1]))
    #%}
    for cha in np.arange(1., (matcompat.size(img, 3.))+1):
        X = im2col(img[:,:,int(cha)-1], np.array(np.hstack((patchsize, patchsize))))
        Xh = im2col(imgh[:,:,int(cha)-1], np.array(np.hstack((patchsize, patchsize-2.))))
        Xv = im2col(imgv[:,:,int(cha)-1], np.array(np.hstack((patchsize-2., patchsize))))
        Xtr = np.sum(vertcat(Xh, Xv))
        if decim > 0.:
            XtrX = vertcat(Xtr, X)
            XtrX = sortrows(XtrX.conj().T).conj().T
            p = np.floor(matdiv(matcompat.size(XtrX, 2.), decim+1.))
            p = np.dot(np.array(np.hstack((np.arange(1., (p)+1)))), decim+1.)
            Xtr = XtrX[0,int(p)-1]
            X = XtrX[1:matcompat.size(XtrX, 1.),int(p)-1]
        
        
        #%%%%% noise level estimation %%%%%
        tau = Inf
        if matcompat.size(X, 2.)<matcompat.size(X, 1.):
            sig2 = 0.
        else:
            cov = matdiv(np.dot(X, X.conj().T), matcompat.size(X, 2.)-1.)
            d = linalg.eig(cov)
            sig2 = d[0]
            
        
        for i in np.arange(2., (itr)+1):
            #%%%%% weak texture selectioin %%%%%
            
        nlevel[int(cha)-1] = np.sqrt(sig2)
        th[int(cha)-1] = tau
        num[int(cha)-1] = matcompat.size(X, 2.)
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    return [nlevel, th, num]
def my_convmtx2(H, m, n):

    # Local Variables: i, H, k, j, m, n, p, s, T
    # Function calls: my_convmtx2, zeros, size
    s = matcompat.size(H)
    T = np.zeros(np.dot(m-s[0]+1., n-s[1]+1.), np.dot(m, n))
    k = 1.
    for i in np.arange(1., (m-s[0]+1.)+1):
        for j in np.arange(1., (n-s[1]+1.)+1):
            for p in np.arange(1., (s[0])+1):
                T[int(k)-1,int(np.dot(i-1.+p-1., n)+j-1.+1.)-1:np.dot(i-1.+p-1., n)+j-1.+1.+s[1]-1.] = H[int(p)-1,:]
                
            k = k+1.
            
        
    return [T]