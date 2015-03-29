import numpy as np
from scipy.ndimage import generic_filter
# from pybrain.tools.shortcuts import buildNetwork
from fann2 import libfann
from skimage.util import view_as_windows


def get_training_data(noisy_image, clear_image, patch_size=3):
    noisy_patches = view_as_windows(
        noisy_image, (patch_size, patch_size))
    training_inputs = [
        noisy_patches[index].ravel()
        for index in np.ndindex(
            noisy_patches.shape[0], noisy_patches.shape[1])
    ]
    patch_offset = (patch_size - 1) / 2
    training_outputs = clear_image[
        patch_offset:clear_image.shape[0] - patch_offset,
        patch_offset:clear_image.shape[1] - patch_offset
    ].ravel()
    return training_inputs, training_outputs


# def filter_pybrain(image, net):
#     """
#     Run convolution on image via MLP neural net (pybrain)
#     """
#     if len(image.shape) != 2:
#         raise ValueError("Only 2D grayscale images are supported")

#     def _filter_func(window):
#         return net.activate(window.flatten())[0]

#     # ANN input window size - e.g. 3x3 or 5x5
#     window_size = int(np.sqrt(net['in'].indim))
#     footprint = np.ones((window_size, window_size))
    
#     result = generic_filter(
#         image, _filter_func, footprint=footprint)
#     return result


def filter_fann(image, ann):
    """
    Run ANN-based filter through image in sliding-window fashion
    """
    def _filter_func(window):
        return ann.run(window.flatten())[0]
    
    window_size = int(np.sqrt(ann.get_num_input()))
    footprint = np.ones((window_size, window_size))
    filtered_image = generic_filter(
        image, _filter_func,
        footprint=footprint)
    return filtered_image


# def filter_fann(image, net):
#     if len(image.shape) != 2:
#         raise ValueError("Only 2D grayscale images are supported")

#     def _filter_func(window):
#         return net.run(window.flatten())[0]

#     # ANN input window size - e.g. 3x3 or 5x5
#     window_size = int(np.sqrt(net.get_num_input()))
#     footprint = np.ones((window_size, window_size))
    
#     result = generic_filter(
#         image, _filter_func, footprint=footprint)
#     return result
