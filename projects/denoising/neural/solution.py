import numpy as np
from fann2 import libfann
from core.individual import Individual
from core.chromosomes import RealChromosome
from scipy.ndimage import generic_filter
import sklearn.feature_extraction.image as skimg
import projects.denoising.imaging.metrics as metrics

from skimage import data, util

# TEMPORARY!
def get_phenotype(params):
    # Small fragment of stock 'camera man' test image
    source_image = None
    # target_image = data.camera()[50:240, 170:350]
    target_image = data.camera()[50:170, 170:280]
    target_image = util.img_as_float(target_image)
    if params['noise_type'] == 'gaussian':
        source_image = util.random_noise(
            target_image, mode='gaussian', var=params['noise_param'])
    else:
        raise ValueError("Only gaussian noise is supported")
    
    # Fixed for now.
    # network_shape = [9, 10, 10, 1]
    network_shape = [9, 20, 1]
    # network_shape = [25, 10, 10, 10, 1]
    # network_shape = [169, 511, 1]
    # network_shape = [49, 151, 1]
    # network_shape = [25, 75, 25]
    # network_shape = [9, 30, 9]
    return NeuralFilterMLP(source_image, target_image, network_shape)


# Copied from:
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/image.py
def reconstruct_from_patches_2d(patches, image_size):
    """Reconstruct the image from all of its patches.
    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.
    Parameters
    ----------
    patches: array, shape = (n_patches, patch_height, patch_width) or
        (n_patches, patch_height, patch_width, n_channels)
        The complete set of patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.
    image_size: tuple of ints (image_height, image_width) or
        (image_height, image_width, n_channels)
        the size of the image that will be reconstructed
    Returns
    -------
    image: array, shape = image_size
        the reconstructed image
    """
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1

    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i:i + p_h, j:j + p_w] += p

    import pdb; pdb.set_trace()

    for i in range(i_h):
        for j in range(i_w):
            # divide by the amount of overlap
            # XXX: is this the most efficient way? memory-wise yes, cpu wise?
            img[i, j] /= float(min(i + 1, p_h, i_h - i) *
                               min(j + 1, p_w, i_w - j))
    return img


def _filter(image, mlp, patch_size, stride_size=1):
    # Get patches as MLP inputs
    # max_patches = 1.0 / float(stride_size ** 2)
    # patches_initial = skimg.extract_patches_2d(image, patch_size, max_patches=max_patches)
    patches_initial = skimg.extract_patches_2d(image, patch_size)
    patches_filtered = np.array([
        np.array(mlp.run(patch.flatten())).astype(np.float).reshape(patch_size)
        for patch in patches_initial
    ])
    # import pdb; pdb.set_trace()
    filtered_image = reconstruct_from_patches_2d(patches_filtered, image.shape)
    return filtered_image


class NeuralFilterMLP(object):
    """
    Multi-layer perceptron
    """
    def __init__(self, source_image, target_image, network_shape):
        # if len(network_shape) != 3:
        #     raise ValueError("Only MLPs with one hidden layer are supported")
        # if network_shape[::-1][0] != 1:
        #     raise ValueError("MLP must have single output")

        self.source_image = source_image
        self.target_image = target_image
        self.network_shape = network_shape

    def __call__(self, *args, **kwargs):
        """
        Initialize individual
        """
        return NeuralFilterMLP._Individual(
            self, NeuralFilterMLP._Genotype(self),
            *args, **kwargs)

    """
    Initialize real-coded chromosome of specified weight
    """
    class _Genotype(object):
        def __init__(self, phenotype):
            # Chromosome length depends on network shape
            s = phenotype.network_shape
            # +1 because of bias neurons in input and hidden layers
            # self.length = ((s[0] + 1) * s[1]) + s[1] + 1

            self.length = 0
            for i in xrange(0, len(s) - 1):
                self.length += (s[i] + 1) * s[i + 1]

        def __call__(self, *args, **kwargs):
            return RealChromosome(self.length, -1.0, 1.0)

    """
    GA solution representing MLP-based image filter
    """
    class _Individual(Individual):
        def __init__(self, phenotype, *args, **kwargs):
            self.phenotype = phenotype
            self.mlp = None
            self.filtered_image = None
            super(NeuralFilterMLP._Individual, self).__init__(*args, **kwargs)

        def _decode(self, chromosome):
            """
            Reconstruct neural network from real-coded chromosome
            """
            # Initialize network
            self.mlp = libfann.neural_net()
            self.mlp.create_standard_array(self.phenotype.network_shape)

            # Set weights
            new_connections = [
                (
                    # Connection - from
                    pair[0][0],
                    # Connection - to
                    pair[0][1],
                    # New weight
                    pair[1]
                )
                for pair in zip(
                    self.mlp.get_connection_array(),
                    chromosome)
            ]
            self.mlp.set_weight_array(new_connections)

        def _calculate_fitness(self):
            """
            Run MLP filter on source image and
            calculate MSE between filtered and target
            """
            def _filter_func(window):
                return self.mlp.run(window.flatten())[0]

            window_size = int(np.sqrt(self.mlp.get_num_input()))
            footprint = np.ones((window_size, window_size))
            self.filtered_image = generic_filter(
                self.phenotype.source_image, _filter_func,
                footprint=footprint)
            # patch_size = (window_size, window_size)
            # self.filtered_image = _filter(self.phenotype.source_image, self.mlp, patch_size)

            max_diff = 255
            for dim in self.filtered_image.shape:
                max_diff *= dim

            # return 1.0 / (1.0 + mse(self.target_image, self.filtered_image))
            # return 1.0 / (1.0 + metrics.absolute_error(
            #     self.target_image, self.filtered_image))
            return 1.0 - float(metrics.absolute_error(
                self.phenotype.target_image, self.filtered_image)) / max_diff
