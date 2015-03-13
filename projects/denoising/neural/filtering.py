import numpy as np
from scipy.ndimage import generic_filter
from pybrain.tools.shortcuts import buildNetwork
from fann2 import libfann


def filter_pybrain(image, net):
    """
    Run convolution on image via MLP neural net (pybrain)
    """
    if len(image.shape) != 2:
        raise ValueError("Only 2D grayscale images are supported")

    def _filter_func(window):
        return net.activate(window.flatten())[0]

    # ANN input window size - e.g. 3x3 or 5x5
    window_size = int(np.sqrt(net['in'].indim))
    footprint = np.ones((window_size, window_size))
    
    result = generic_filter(
        image, _filter_func, footprint=footprint)
    return result


def filter_fann(image, net):
    if len(image.shape) != 2:
        raise ValueError("Only 2D grayscale images are supported")

    def _filter_func(window):
        return net.run(window.flatten())[0]

    # ANN input window size - e.g. 3x3 or 5x5
    window_size = int(np.sqrt(net.get_num_input()))
    footprint = np.ones((window_size, window_size))
    
    result = generic_filter(
        image, _filter_func, footprint=footprint)
    return result


def init_fann():
    net = libfann.neural_net()
    net.create_standard_array([4, 2, 1])
    return net


def init_pybrain():
    net = buildNetwork(9, 20, 1)
    return net


def decode(string):
    # Return ANN...
    pass


def encode(net):
    # Return real/binary string...
    pass


if __name__ == "__main__":
    from skimage import data, io, util
    img = data.camera()[50:240, 170:350]
    img = util.img_as_float(img)
    
    # filtered = filter_neural(img, net)
    net = init_fann()
    filtered = filter_fann(img, net)

    # io.imshow(filtered)
    # io.show()
