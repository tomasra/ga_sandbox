import os
import math
import numpy as np
from skimage import util, io
from skimage.measure import structural_similarity as ssim
from skimage.filter.rank import median
from skimage.filter import hsobel, vsobel
import skimage.morphology as mph
from scipy.ndimage.filters import generic_filter
from scipy.ndimage import gaussian_filter
from projects.denoising.imaging.utils import render


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

def sharpness_test_image(blur_sigma=0.0):
    image = np.zeros((20, 20))
    for x in xrange(0, 20):
        for y in xrange(10, 20):
            image[x,y] = 1.0
    # Add gaussian blur
    image = gaussian_filter(image, sigma=blur_sigma)
    return image

def _render(image):
    io.imshow(image)
    io.show()

def sharpness(image, edge_threshold=0.0001, w=5, t=1.0):
    """
    Code implemented by the following article:
    Sharpness Estimation for Document and Scene Images,
    Kumar, Chen, Doermann
    """
    # Median filtering
    w_size = (w * 2) + 1
    image = util.img_as_float(image)
    image_m = util.img_as_float(median(image, mph.square(3)))

    # Window functions
    def dom_func(window):
        # import pdb; pdb.set_trace()
        return abs(
            abs(window[4] - window[2]) - abs(window[2] - window[0])
        )

    def contrast_func(window):
        # print window
        s = 0.0
        for i in xrange(0, len(window) - 1):
            # print i
            s += abs(window[i] - window[i+1])
        return s

    # Delta DoM in horizontal direction
    dom_x_values = generic_filter(image_m, dom_func,
        size=(1, 5),
        mode='reflect')
    # Delta DoM in vertical direction
    dom_y_values = generic_filter(image_m, dom_func,
        size=(5, 1),
        mode='reflect')

    dom_x = generic_filter(
        dom_x_values, lambda w: sum(w),
        size=(1, w_size), mode='reflect')
    dom_y = generic_filter(
        dom_y_values, lambda w: sum(w),
        size=(w_size, 1), mode='reflect')


    edges_x = vsobel(image)
    # Normalize
    edges_x *= (1.0 / edges_x.max())
    edges_x_pixels = len(edges_x[edges_x > edge_threshold].ravel())

    edges_y = hsobel(image)
    # Normalize
    edges_y *= (1.0 / edges_y.max())
    edges_y_pixels = len(edges_y[edges_y > edge_threshold].ravel())

    # Contrast in horizontal direction
    contrast_x = generic_filter(image, contrast_func,
        size=(1, w_size + 1),
        mode='reflect')
    # Contrast in vertical direction
    contrast_y = generic_filter(image, contrast_func,
        size=(w_size + 1, 1),
        mode='reflect')

    sharpness_x = dom_x / contrast_x
    sharpness_y = dom_y / contrast_y

    # import pdb; pdb.set_trace()

    sharp_x_pixels = len(np.where(
        sharpness_x[edges_x > edge_threshold] > t
    )[0])
    sharp_y_pixels = len(np.where(
        sharpness_y[edges_y > edge_threshold] > t
    )[0])

    # import pdb; pdb.set_trace()

    if edges_x_pixels > 0:
        rx = (float(sharp_x_pixels) / edges_x_pixels)
    else:
        rx = 1

    if edges_y_pixels > 0:
        ry = (float(sharp_y_pixels) / edges_y_pixels)
    else:
        ry = 1

    final_estimate = np.sqrt(
        (rx ** 2) + (ry ** 2)
    )    
    return final_estimate


# External matlab code imports
import os
import oct2py
from oct2py import octave
# addpath command hangs for some reason but path is actually added succesfully
# set some small timeout and later remove it
# oct = Oct2Py(timeout=0.01)
# oct = Oct2Py()
# BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# try:
# oct.addpath(os.path.join(BASE_PATH, 'matlab/q/MetricQ'))
# except oct2py.utils.Oct2PyError:
    # oct.timeout = None

def q(image, patch_size=8):
    aniso_set = octave.AnisoSetEst(image, patch_size)
    q_val = octave.MetricQ(image, patch_size, aniso_set)
    return q_val


def svd_coherence(grad_x, grad_y):
    gxvect = grad_x.ravel()
    gyvect = grad_y.ravel()
    grad = np.array([gxvect, gyvect])
    u, c, v = np.linalg.svd(grad)
    # single precision here?
    s1, s2 = c[0], c[1]
    co = np.abs((s1 - s2) / (s1 + s2))
    s1 = np.abs(s1)
    if np.isnan(co):
        co = 0
    return co, s1


def q_py(image, patch_size=8):
    """
    Q re-implementation in pure python
    """
    height, width = image.shape
    w = math.floor(float(width) / patch_size)
    h = math.floor(float(height) / patch_size)
    
    alpha = 0.001
    threshold = alpha ** (1.0 / (patch_size ** 2 - 1))
    threshold = math.sqrt((1 - threshold) / (1 + threshold))

    q = 0.0
    for m in xrange(int(h)):
        for n in xrange(int(w)):
            top, bottom = m * patch_size, (m + 1) * patch_size
            left, right = n * patch_size, (n + 1) * patch_size
            patch = image[top:bottom, left:right]
            grads = np.gradient(patch)
            grad_x, grad_y = grads[0], grads[1]

            coh = svd_coherence(grad_x, grad_y)
            if coh[0] > threshold:
                # Anisotropic patch found
                q += (coh[0] * coh[1])
    return q / (w * h)


# OCR
import pytesseract
import Levenshtein
from PIL import Image
def ocr_accuracy(image, text):
    """
    Run tesseract-ocr on provided image
    and compute levenshtein distance between 
    ocr'ed and original text.
    Accuracy percentage is returned
    """
    pil_image = Image.fromarray(util.img_as_ubyte(image))
    # Config has dictionaries disabled
    conf_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '_tesseract.config')
    ocr_text = pytesseract.image_to_string(pil_image, config=conf_path)
    # Distance
    l_dist = Levenshtein.distance(text, ocr_text)
    # Accuracy percentage
    # TODO: edge cases when ocr text is empty, etc
    # print l_dist
    # print ocr_text
    # print text
    acc = float(len(text) - l_dist) / len(text)
    # print acc
    return np.clip(acc, a_min=0.0, a_max=1.0)


def d_feature(image, tau=0.0):
    edges = canny(image)
    render(edges)
    for start_index, edge in np.ndenumerate(edges):
        if edge == True:
            # Edge gradient coordinates
            index, end_index = start_index, None
            previous_index = None
            g = 0.0

            visited_points = []
            while g < tau:
                # Pixels around the current edge pixel
                # max/min because of image (rectangle) edges
                window = np.zeros((3, 3))
                # Some negative value
                window.fill(-1 * np.inf)

                # Image window - source
                src_top, src_bottom = index[0] - 1, index[0] + 2
                src_left, src_right = index[1] - 1, index[1] + 2

                # Local window - destination
                dest_top, dest_bottom = 0, 3
                dest_left, dest_right = 0, 3

                if index[0] == 0:
                    # Top edge
                    src_top += 1
                    dest_top += 1
                if index[0] == image.shape[0] - 1:
                    # Bottom edge
                    src_bottom -= 1
                    dest_bottom -= 1
                if index[1] == 0:
                    # Left edge
                    src_left += 1
                    dest_left += 1
                if index[1] == image.shape[1] - 1:
                    # Very right
                    src_right -= 1
                    dest_right -= 1
                    
                src = image[src_top:src_bottom, src_left:src_right]
                window[dest_top:dest_bottom, dest_left:dest_right] = src

                # TODO: abs here???
                diffs = abs(window - image[index])

                # Center value
                diffs[1, 1] = -1 * np.inf
                # Over-the-edge values
                diffs[diffs == np.inf] = (-1 * np.inf)
                # The one which we just came from
                if previous_index is not None:
                    previous_center = (
                        1 + (previous_index[0] - index[0]),
                        1 + (previous_index[1] - index[1])
                    )
                    diffs[previous_center] = (-1 * np.inf)

                # print index
                # import pdb; pdb.set_trace()

                # print index
                # import pdb; pdb.set_trace()

                # diffs = image[index] - window
                # print top, bottom, left, right
                # print window
                # import pdb; pdb.set_trace()

                # HACK: replace central diff (zero) with some
                # very negative value

                # Index of next image pixel
                next_index = np.unravel_index(
                    np.argmax(diffs), window.shape)

                # # Translate to overall image coordinates
                next_index = (
                    index[0] + (next_index[0] - 1),
                    index[1] + (next_index[1] - 1)
                )

                g = abs(image[next_index] - image[start_index])
                print start_index, next_index, g

                if g >= tau:
                    # Stop iterating
                    end_index = next_index
                    break
                else:
                    previous_index = index
                    index = next_index
