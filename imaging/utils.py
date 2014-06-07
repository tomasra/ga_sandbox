from skimage.data import load, imread
import skimage.color as color
import skimage.morphology as mph
from PIL import Image
import imaging.filters as flt
import scipy.ndimage.measurements as msr
import scipy.cluster.vq as vq

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('GTKAgg')
import matplotlib.cm as cm


def read_image(filepath):
    """
    Reads a PNG image and returns three numpy arrays
    (as RGB color planes)
    """
    input = imread(filepath)
    # Take RGB planes (exclude alpha channel)
    rgb = [
        input[:, :, i]
        for i in xrange(3)
    ]
    return rgb


def render_image(image):
    """
    Image - list of three numpy arrays (color planes)
    """
    if isinstance(image, np.ndarray):
        full_image = [image, image, image]
        # import pdb; pdb.set_trace()
        # render_image([image, image, image])
        render_image(full_image)
    elif isinstance(image, list) and len(image) == 3:
        height, width = image[0].shape[0], image[0].shape[1]
        # Alpha channel plane with all 255's
        alpha = np.empty_like(image[0])
        alpha.fill(255)
        # Combine all channels into one image with third dimension
        rgb = [plane.astype(np.uint8) for plane in image]
        combined_image = np.dstack(tuple(rgb + [alpha]))
        plt.imshow(combined_image)
        plt.show()
    else:
        raise ValueError("Invalid image data")


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


    # pil_image = Image.fromarray(image)
    # # quantized = pil_image.quantize(colors, kmeans=1)
    # quantized = pil_image.convert('P', palette=Image.ADAPTIVE, colors=colors)
    # import pdb; pdb.set_trace()
    # final = np.array(quantized) * 255
    # return final


def char_parameters(image):
    """
    Return image parameters for evaluating fitness function:
    : text_color - self explanatory
    : bg_color - self explanatory
    : text_regions - count of separate connected regions with text_color
    : thickness - average shape thickness, evaluated by
                  running dilation until nothing remains
    """
    # Reduce to two colors
    colors = 2

    # Convert list to normal image object with RGB values as third dimension
    image_rgb = np.dstack(image)

    # Flat list of RGB pixels
    pixels = np.reshape(
        image_rgb,
        (image_rgb.shape[0] * image_rgb.shape[1], image_rgb.shape[2])
    )

    # Color centers
    centroids, _ = vq.kmeans(pixels, colors)
    quantized, _ = vq.vq(pixels, centroids)
    quantized_idx = quantized.reshape(
        (image_rgb.shape[0], image_rgb.shape[1])
    )
    # image_result = centroids[quantized_idx]

    if len(centroids) == 1:
        # Image consists of just a single color
        text_color, bg_color = centroids[0], centroids[0]
        text_regions = 0
        ttb_ratio = 0
        # erosion_count = 0
    else:
        # Assuming that text color is the one having less pixels
        text_idx = np.argmin(np.bincount(quantized))
        text_color = centroids[text_idx]

        bg_idx = np.argmax(np.bincount(quantized))
        bg_color = centroids[bg_idx]

        for_labeling = quantized_idx == text_idx
        labeled, text_regions = msr.label(for_labeling)

        # Text-to-background ratio
        text_pixels = len(np.where(quantized == text_idx)[0])
        bg_pixels = len(np.where(quantized == bg_idx)[0])
        ttb_ratio = float(text_pixels) / float(bg_pixels)
        # import pdb; pdb.set_trace()

        # Estimate shape thickness: apply multiple erosions
        # eroded = quantized_idx

        # Check if either character or background is white.
        # In case of white background - invert.
        # if len(np.where(eroded == 1)[0]) > len(np.where(eroded == 0)[0]):
        #     eroded = 1 - eroded

        # erosion_count = 0
        # while len(np.where(eroded == 1)[0]) > 0:
        #     # Repeat erosion until no more white areas remain
        #     eroded = mph.binary_erosion(
        #         eroded,
        #         mph.rectangle(3, 3))
        #     erosion_count += 1


    return text_color, bg_color, text_regions, ttb_ratio


def extract_boundaries(image):
    """
    Returns binary image with detected edges
    """
    gray = rgb2gray(image)
    edges = flt.sobel(gray)
    # eroded = flt.erosion(image, size=(3, 3))
    # boundaries = eroded - image
    quantized = quantize(edges)
    inverted = flt.inversion(quantized)
    return inverted


def region_sizes(image):
    """
    Labels binary image and returns list of labeled region sizes
    (how many pixels in each region)
    """
    labeled, count = msr.label(image)
    # import pdb; pdb.set_trace()
    results = [
        len(np.dstack(np.where(labeled == i))[0])
        for i in xrange(1, count + 1)
    ]
    return results


def connected_regions(image):
    """
    Converts image into grayscale, quantizes, counts connected regions
    """
    gray = rgb2gray(image)
    quantized = quantize(gray)
    # render_image(quantized)
    # inverted = flt.inversion(quantized)
    # regions = len(region_sizes(inverted))
    regions = len(region_sizes(quantized))
    if regions == 0:
        regions = image[0].shape[0] * image[0].shape[1]
    # print regions
    return regions
    # edges = extract_boundaries(filtered_image)
    # regions = len(region_sizes(edges))
