from skimage.data import load, imread
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
    height, width = image[0].shape[0], image[0].shape[1]
    # Alpha channel plane with all 255's
    alpha = np.empty_like(image[0])
    alpha.fill(255)
    # Combine all channels into one image with third dimension
    rgb = [plane.astype(np.uint8) for plane in image]
    combined_image = np.dstack(tuple(rgb + [alpha]))
    plt.imshow(combined_image)
    plt.show()


def render_grayscale(image):
    """
    Image - one numpy array
    """
    plt.imshow(image, cmap=cm.Greys_r)
    plt.show()
