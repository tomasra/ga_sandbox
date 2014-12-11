# from skimage.data import imread
import skimage.color as color
import skimage.morphology as mph
from PIL import Image, ImageOps
import projects.denoising.imaging.filters as flt
import scipy.ndimage.measurements as msr
import scipy.cluster.vq as vq
from pyemd import emd
# import cv

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('GTKAgg')
plt.switch_backend('TkAgg')
import matplotlib.cm as cm


# def read_image(filepath):
#     """
#     Reads a PNG image and returns three numpy arrays
#     (as RGB color planes)
#     """
#     input = imread(filepath)
#     # Take RGB planes (exclude alpha channel)
#     rgb = [
#         input[:, :, i]
#         for i in xrange(3)
#     ]
#     return rgb


def render_image(image, filename=None):
    """
    Render Image instance on screen or into file
    """
    fig = plt.figure()
    plt.imshow(image)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def render_grayscale(image):
    """
    Image - one numpy array
    """
    plt.imshow(image, cmap=cm.Greys_r)
    plt.show()


def rgb2gray(image):
    """
    Convert color image (list of 3 numpy arrays) to grayscale
    """
    scikit_img = np.dstack(image)
    gray_img = (color.rgb2gray(scikit_img) * 255).astype(np.uint8)
    return gray_img


def numpy2pil(image):
    stacked = np.dstack(image)
    return Image.fromarray(stacked)


def pil2numpy(image):
    return np.array(image)


def quantize(image, colors=2, return_colors=False):
    """
    Simple image color reduction
    """
    image_rgb = np.dstack(image)
    pixels = np.reshape(
        image_rgb,
        (image_rgb.shape[0] * image_rgb.shape[1], image_rgb.shape[2])
    )
    centroids, _ = vq.kmeans(pixels, colors)
    quantized, _ = vq.vq(pixels, centroids)
    quantized_idx = quantized.reshape(
        (image_rgb.shape[0], image_rgb.shape[1])
    )
    result = centroids[quantized_idx]
    final = [
        result[:, :, i]
        for i in xrange(3)
    ]

    # import pdb; pdb.set_trace()
    return final


def binarize(image):
    quantized = quantize(image, colors=2)
    # Now replace almost-black with completely black
    # Same with almost-white
    for plane in quantized:
        image_black = np.min(plane)
        image_white = np.max(plane)
        plane[plane == image_black] = 0     # True black
        plane[plane == image_white] = 255   # True white

    # import pdb; pdb.set_trace()
    return quantized


# def char_parameters(image):
#     """
#     Return image parameters for evaluating fitness function:
#     : text_color - self explanatory
#     : bg_color - self explanatory
#     : text_regions - count of separate connected regions with text_color
#     : thickness - average shape thickness, evaluated by
#                   running dilation until nothing remains
#     """
#     # Reduce to two colors
#     colors = 2

#     # Convert list to normal image object with RGB values as third dimension
#     image_rgb = np.dstack(image)

#     # Flat list of RGB pixels
#     pixels = np.reshape(
#         image_rgb,
#         (image_rgb.shape[0] * image_rgb.shape[1], image_rgb.shape[2])
#     )

#     # Color centers
#     centroids, _ = vq.kmeans(pixels, colors)
#     quantized, _ = vq.vq(pixels, centroids)
#     quantized_idx = quantized.reshape(
#         (image_rgb.shape[0], image_rgb.shape[1])
#     )
#     # image_result = centroids[quantized_idx]

#     if len(centroids) == 1:
#         # Image consists of just a single color
#         text_color, bg_color = centroids[0], centroids[0]
#         text_regions = 0
#         ttb_ratio = 0
#         # erosion_count = 0
#     else:
#         # Assuming that text color is the one having less pixels
#         text_idx = np.argmin(np.bincount(quantized))
#         text_color = centroids[text_idx]

#         bg_idx = np.argmax(np.bincount(quantized))
#         bg_color = centroids[bg_idx]

#         for_labeling = quantized_idx == text_idx
#         labeled, text_regions = msr.label(for_labeling)

#         # Text-to-background ratio
#         text_pixels = len(np.where(quantized == text_idx)[0])
#         bg_pixels = len(np.where(quantized == bg_idx)[0])
#         ttb_ratio = float(text_pixels) / float(bg_pixels)
#         # import pdb; pdb.set_trace()

#         # Estimate shape thickness: apply multiple erosions
#         # eroded = quantized_idx

#         # Check if either character or background is white.
#         # In case of white background - invert.
#         # if len(np.where(eroded == 1)[0]) > len(np.where(eroded == 0)[0]):
#         #     eroded = 1 - eroded

#         # erosion_count = 0
#         # while len(np.where(eroded == 1)[0]) > 0:
#         #     # Repeat erosion until no more white areas remain
#         #     eroded = mph.binary_erosion(
#         #         eroded,
#         #         mph.rectangle(3, 3))
#         #     erosion_count += 1


#     return text_color, bg_color, text_regions, ttb_ratio





# def histogram(image):
#     pil_image = numpy2pil(image)
#     # WARNING: hardcoded values
#     raw_hist = pil_image.histogram()
#     hist = [
#         np.array(raw_hist[:256]),
#         np.array(raw_hist[256:512]),
#         np.array(raw_hist[512:])
#     ]
#     return hist


# def histogram_diff(hist1, hist2):
#     n = len(hist1[0])

#     # distance = |x1-x2| + |y1-y2|
#     distance_matrix = np.array([
#         abs(x - y)
#         for x in xrange(n)
#         for y in xrange(n)
#     ]).reshape(n, n).astype(np.float)

#     result = sum([
#         emd(
#             pair[0].astype(np.float),
#             pair[1].astype(np.float),
#             distance_matrix
#         )
#         for pair in zip(hist1, hist2)
#     ])
#     # print result

#     # result = sum([
#     #     cv.CalcEMD2(
#     #         cv.fromarray(
#     #             np.dstack((
#     #                 pair[0][0],
#     #                 xrange(len(pair[0][0]))
#     #             ))[0].astype(np.float32)
#     #         ),
#     #         cv.fromarray(
#     #             np.dstack((
#     #                 pair[1][0],
#     #                 xrange(len(pair[1][0]))
#     #             ))[0].astype(np.float32)
#     #         ),
#     #         cv.CV_DIST_L1
#     #     )
#     #     for pair in zip(hist1, hist2)
#     # ])

#     # import pdb; pdb.set_trace()
#     # print result
#     return result


# def ideal_histogram(image, ttb_ratio):
#     # total = image[0].shape[0] * image[0].shape[1]
#     # # Not an error here, np.histogram leaves out one element
#     # pixels = np.zeros(255)#.astype(np.uint32)
#     # pixels[0] = total * ttb_ratio        # Black text pixels
#     # pixels[254] = total * (1.0 - ttb_ratio)  # White background pixels
#     # bins = np.array(xrange(256))
#     # return [(pixels, bins)] * 3

#     total = image[0].shape[0] * image[0].shape[1]
#     pixels = np.zeros(256)
#     pixels[0] = total * ttb_ratio        # Black text pixels
#     pixels[255] = total * (1.0 - ttb_ratio)  # White background pixels
#     return [pixels] * 3


# def max_histogram_diff(image):
#     total_pixels = image[0].shape[0] * image[0].shape[1]
#     np1 = np.zeros(256).astype(np.float)
#     np2 = np.zeros(256).astype(np.float)
#     np1[0] = total_pixels
#     np2[255] = total_pixels
#     hist1 = [np1, np1, np1]
#     hist2 = [np2, np2, np2]
#     return histogram_diff(hist1, hist2)
