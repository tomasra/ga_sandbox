import numpy as np
from skimage.data import load, imread
import skimage.color as color
import skimage.morphology as mph
from projects.denoising.imaging.image import Image
from PIL import ImageOps
import projects.denoising.imaging.filters as flt
import scipy.ndimage.measurements as msr
import scipy.cluster.vq as vq


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
    # print count
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
    # render_image(image)

    colors = 2

    # Quantization into two colors
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

    if len(centroids) > 1:
        # for_render = (quantized_idx * 255).astype(np.uint8)
        # render_image(for_render)
        regions = len(region_sizes(quantized_idx))
        regions_inverted = len(region_sizes(1 - quantized_idx))
        # import pdb; pdb.set_trace()

        # if regions == 0:
        #     regions = image[0].shape[0] * image[0].shape[1]
        # print regions
        return max([regions, regions_inverted])
    else:
        return 0


def quantize(image, colors=2, return_colors=False):
    """
    Simple image color reduction
    """
    # Convert to flat list of pixels
    pixels = image.reshape((
        image.width * image.height,
        len(image.channels)
    ))
    centroids, _ = vq.kmeans(pixels, colors)
    quantized, _ = vq.vq(pixels, centroids)
    # Convert back to 2D image
    quantized_idx = quantized.reshape(
        (image.height, image.width)
    )
    result = centroids[quantized_idx]
    return Image(result)


def binarize(image):
    """
    Reduce all image colors to just black and white
    """
    quantized = quantize(image, colors=2)
    # Now replace almost-black with completely black
    # Same with almost-white
    quantized[quantized == np.min(quantized)] = 0     # True black
    quantized[quantized == np.max(quantized)] = 255   # True white
    return quantized
