import skimage.filter as f
import skimage.filter.rank as fr
import skimage.morphology as fm
import numpy as np
import scipy.ndimage.filters as sf


# Single argument filters
def mean(img):
    return fr.mean(img)


def minimum(img):
    return fr.minimum(img)


def maximum(img):
    return fr.maximum(img)


def vsobel(img):
    return f.vsobel(img)


def hsobel(img):
    return f.hsobel(img)


def sobel(img):
    return f.sobel(img)


def lightedge(img):
    # http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    laplacian = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
    return sf.convolve(img, laplacian)


def darkedge(img):
    # http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    laplacian = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
    # laplacian + 255
    return inversion(sf.convolve(img, laplacian))


def erosion(img):
    selem = np.array([1, 1, 1], [1, 1, 1], [1, 1, 1])
    return fm.erosion(img, selem)


def dilation(img):
    selem = np.array([1, 1, 1], [1, 1, 1], [1, 1, 1])
    return fm.dilation(img, selem)


def inversion(img):
    it = np.nditer([img, None])
    for x, res in it:
        res[...] = 255 - x
    return it.operands[1]


# Two-argument filters

def logical_sum(img1, img2):
    """
    Maximum of two color planes
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        it = np.nditer([img1, img2, None])
        for x1, x2, res in it:
            res[...] = max([x1, x2])
        return it.operands[2]
    else:
        raise ValueError("Image size or dtype mismatch")


def logical_product(img1, img2):
    """
    Minimum of two color planes
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        it = np.nditer([img1, img2, None])
        for x1, x2, res in it:
            res[...] = min([x1, x2])
        return it.operands[2]
    else:
        raise ValueError("Image size or dtype mismatch")


def algebraic_sum(img1, img2):
    """
    Sum of two color planes - product / 255
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        it = np.nditer([img1, img2, None])
        for x1, x2, res in it:
            t = int(round(float(x1 * x2) / 255))
            res[...] = (x1 + x2) - t
        return it.operands[2]
    else:
        raise ValueError("Image size or dtype mismatch")


def algebraic_product(img1, img2):
    """
    Product of two color planes / 255
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        it = np.nditer([img1, img2, None])
        for x1, x2, res in it:
            res[...] = int(round(float(x1 * x2) / 255))
        return it.operands[2]
    else:
        raise ValueError("Image size or dtype mismatch")


def bounded_sum(img1, img2):
    """
    g = sum of two color planes
    if g > 255: g = 255
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        it = np.nditer([img1, img2, None])
        for x1, x2, res in it:
            s = x1 + x2
            if s > 255:
                s = 255
            res[...] = s
        return it.operands[2]
    else:
        raise ValueError("Image size or dtype mismatch")


def bounded_product(img1, img2):
    """
    g = product of two color planes - 255
    if g < 0: g = 0
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        it = np.nditer([img1, img2, None])
        for x1, x2, res in it:
            p = (x1 * x2) - 255
            if p < 0:
                p = 0
            res[...] = p
        return it.operands[2]
    else:
        raise ValueError("Image size or dtype mismatch")
