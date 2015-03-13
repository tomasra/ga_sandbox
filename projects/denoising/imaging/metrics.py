import numpy as np
from skimage import util
from skimage.measure import structural_similarity as ssim


# http://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
def mse(image_a, image_b):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    image_a = util.img_as_ubyte(image_a)
    image_b = util.img_as_ubyte(image_b)
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def psnr(image_a, image_b):
    pass


def absolute_error(image_a, image_b):
    """
    Sum of pixel differences
    Images - 2d numpy arrays
    """
    image_a = util.img_as_ubyte(image_a)
    image_b = util.img_as_ubyte(image_b)
    return np.sum(
        np.absolute(
            image_a.view(np.ndarray).astype(np.int16) -
            image_b.view(np.ndarray).astype(np.int16)
        )
    )
