import skimage.filter.rank as fr
import skimage.morphology as fm
import numpy as np
import scipy.ndimage.filters as sf
import scipy.ndimage.morphology as sm


one_argument_filters = []
two_argument_filters = []
other_filters = []
all_filters = one_argument_filters + two_argument_filters + other_filters


def filter(decorated_function):
    """
    Very simple decorator to mark functions
    which work as image filters
    """
    global one_argument_filters
    global two_argument_filters
    global other_filters

    # Determine argument count of the function
    argcount = decorated_function.func_code.co_argcount
    if argcount == 1:
        one_argument_filters.append(decorated_function)
    elif argcount == 2:
        two_argument_filters.append(decorated_function)
    else:
        other_filters.append(decorated_function)
    # Function is unchanged
    return decorated_function


# Single argument filters
@filter
def mean(img):
    result = fr.mean(img, fm.square(3)).astype(img.dtype)
    return result


@filter
def minimum(img):
    result = sf.minimum_filter(img, size=3).astype(img.dtype)
    return result


@filter
def maximum(img):
    return sf.maximum_filter(img, size=3).astype(img.dtype)


@filter
def vsobel(img):
    result = sf.sobel(img.astype(np.int16), axis=1)
    # With normalization
    result /= 8
    result += 128
    result = result.astype(img.dtype)
    return result


@filter
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


@filter
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


@filter
def lightedge(img):
    result = inversion(darkedge(img))
    return result


@filter
def darkedge(img):
    result = sf.laplace(img.astype(np.int16))
    result /= 4
    result += 128
    result = result.astype(img.dtype)
    return result


@filter
def erosion(img):
    result = sm.grey_erosion(img, size=(3, 3)).astype(img.dtype)
    return result


@filter
def dilation(img):
    result = sm.grey_dilation(img, size=(3, 3)).astype(img.dtype)
    return result


@filter
def inversion(img):
    result = np.subtract(255, img).astype(img.dtype)
    return result


# Two-argument filters
@filter
def logical_sum(img1, img2):
    """
    Maximum of two color planes
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        return np.maximum(img1, img2).astype(img1.dtype)
    else:
        raise ValueError("Image size or dtype mismatch")


@filter
def logical_product(img1, img2):
    """
    Minimum of two color planes
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        return np.minimum(img1, img2).astype(img1.dtype)
    else:
        raise ValueError("Image size or dtype mismatch")


@filter
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
        ).astype(img1.dtype)
    else:
        raise ValueError("Image size or dtype mismatch")


@filter
def algebraic_product(img1, img2):
    """
    Product of two color planes / 255
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        return np.divide(
            np.multiply(img1, img2).astype(np.float32),
            255
        ).round().astype(img1.dtype)
    else:
        raise ValueError("Image size or dtype mismatch")


@filter
def bounded_sum(img1, img2):
    """
    g = sum of two color planes
    if g > 255: g = 255
    """
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        return np.clip(
            np.add(img1, img2), 0, 255
        ).astype(img1.dtype)
    else:
        raise ValueError("Image size or dtype mismatch")


@filter
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
        ).astype(img1.dtype)
    else:
        raise ValueError("Image size or dtype mismatch")
