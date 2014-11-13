import skimage.filter.rank as fr
import skimage.morphology as fm
import numpy as np
import scipy.ndimage.filters as sf
import scipy.ndimage.morphology as sm


# Single argument filters
def mean(img):
    result = fr.mean(img, fm.square(3))
    return result


def minimum(img):
    result = sf.minimum_filter(img, size=3)
    return result


def maximum(img):
    return sf.maximum_filter(img, size=3)


def vsobel(img):
    result = sf.sobel(img.astype(np.int16), axis=1)
    # With normalization
    result /= 8
    result += 128
    result = result.astype(img.dtype)
    return result


def hsobel(img):
    # With normalization
    result = np.add(
        np.divide(
            sf.sobel(img.astype(np.int16), axis=0),
            8
        ),
        128
    ).astype(img.dtype)
    return result


def sobel(img):
    img_int16 = img.astype(np.int16)
    result = np.add(
        np.divide(
            np.hypot(
                sf.sobel(img_int16, axis=0),
                sf.sobel(img_int16, axis=1)
            ),
            8
        ),
        128
    ).astype(img.dtype)
    return result


def lightedge(img):
    result = inversion(darkedge(img))
    return result


def darkedge(img):
    result = sf.laplace(img.astype(np.int16))
    result /= 4
    result += 128
    result = result.astype(img.dtype)
    return result


def erosion(img):
    result = sm.grey_erosion(img, size=(3, 3))
    return result


def dilation(img):
    result = sm.grey_dilation(img, size=(3, 3))
    return result


def inversion(img):
    result = np.subtract(255, img)
    return result


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
                np.multiply(img1, img2).astype(np.float32),
                255
            ).round()
        )
    else:
        raise ValueError("Image size or dtype mismatch")


def algebraic_product(img1, img2):
    """
    Product of two color planes / 255
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        return np.divide(
            np.multiply(img1, img2).astype(np.float32),
            255
        ).round()
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
