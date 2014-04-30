from skimage.data import load
import numpy as np
import matplotlib.pyplot as plt


def read_image(filepath):
    """
    Reads a PNG image and returns three numpy arrays
    (as RGB color planes)
    """
    input = load(filepath)
    # Take RGB planes (exclude alpha channel)
    return [
        input[:, :, i]
        for i in xrange(0, 3)
    ]


def render_image(image):
    """
    Image - list of three numpy arrays (color planes)
    """
    height, width = image[0].shape[0], image[0].shape[1]
    # Alpha channel plane with all 255's
    alpha = np.empty_like(image[0])
    alpha.fill(255)
    # Combine all channels into one image with third dimension
    combined_image = np.dstack(tuple(image + alpha))
    plt.imshow(combined_image)
    plt.show()
