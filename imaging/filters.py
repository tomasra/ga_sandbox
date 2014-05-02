import skimage.filter as f
import skimage.filter.rank as fr
import skimage.morphology as fm
import numpy as np
import scipy.ndimage.filters as sf

_selem = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])


# Single argument filters
def mean(img):
    return fr.mean(img, fm.square(3))


def minimum(img):
    return fr.minimum(img, fm.square(3))


def maximum(img):
    return fr.maximum(img, fm.square(3))


def vsobel(img):
    # return (img * f.vsobel(img)).astype(img.dtype)
    return np.multiply(
        img,
        f.vsobel(img)
    ).astype(img.dtype)


def hsobel(img):
    # return (img * f.hsobel(img)).astype(img.dtype)
    return np.multiply(
        img,
        f.hsobel(img)
    ).astype(img.dtype)


def sobel(img):
    # return (img * f.sobel(img)).astype(img.dtype)
    return np.multiply(
        img,
        f.sobel(img)
    ).astype(img.dtype)


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
    return fm.erosion(img, fm.square(3))


def dilation(img):
    return fm.dilation(img, fm.square(3))


def inversion(img):
    return np.subtract(255, img)


# Two-argument filters

def logical_sum(img1, img2):
    """
    Maximum of two color planes
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        return np.maximum(img1, img2)
    else:
        raise ValueError("Image size or dtype mismatch")


def logical_product(img1, img2):
    """
    Minimum of two color planes
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        return np.minimum(img1, img2)
    else:
        raise ValueError("Image size or dtype mismatch")


def algebraic_sum(img1, img2):
    """
    Sum of two color planes - product / 255
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        return np.subtract(
            np.add(img1, img2),
            np.divide(
                np.multiply(img1, img2),
                255
            )
        )
    else:
        raise ValueError("Image size or dtype mismatch")


def algebraic_product(img1, img2):
    """
    Product of two color planes / 255
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        return np.divide(
            np.multiply(img1, img2),
            255
        )
    else:
        raise ValueError("Image size or dtype mismatch")


def bounded_sum(img1, img2):
    """
    g = sum of two color planes
    if g > 255: g = 255
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        return np.clip(
            np.add(img1, img2), 0, 255
        )
    else:
        raise ValueError("Image size or dtype mismatch")


def bounded_product(img1, img2):
    """
    g = product of two color planes - 255
    if g < 0: g = 0
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        return np.clip(
            np.subtract(
                np.multiply(img1, img2),
                255
            ),
            0, 255
        )
    else:
        raise ValueError("Image size or dtype mismatch")
