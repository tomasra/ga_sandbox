import os
import math
import json
import numpy as np
from fann2 import libfann
from core.individual import Individual
from core.chromosomes import RealChromosome, RealStatChromosome
from scipy.ndimage import generic_filter
import sklearn.feature_extraction.image as skimg
# import projects.denoising.imaging.metrics as metrics
from projects.denoising.imaging.char_drawer import get_test_image
from projects.denoising.imaging import metrics
from projects.denoising.neural.filtering import filter_fann

from skimage import data, util, io

# TEMPORARY!
def get_phenotype(params):
    source_image = util.img_as_float(io.imread(params['input_image']))
    # Fixed for now.
    network_shape = [25, 10, 10, 1]
    init_method = params['init_method']     # 'uniform' or 'normal'
    if init_method == 'normal':
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'gaussian-set-19-stats.json'   
        )
        with open(path, 'r') as fp:
            stats = json.load(fp)
    else:
        stats = None

    fitness_func = params['fitness_func']
    # fitness_func = 'ann'
    return NeuralFilterMLP(source_image, network_shape, init_method, stats, fitness_func)


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
    def __init__(self, source_image, network_shape, init_method, stats, fitness_func):
        self.source_image = source_image
        self.target_image = None
        self.network_shape = network_shape
        # Statistical chromosome initialization
        self.init_method = init_method
        self.stats = stats
        # Other
        self.fitness_func = fitness_func
        self.initial_q = metrics.q_py(source_image)

        self.trained_anns = []
        if fitness_func == 'ann':
            # Instantiate all neural nets
            ann_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'trained_anns'
            )
            for filename in os.listdir(ann_dir):
                if filename.endswith('.net'):
                    ann_path = os.path.join(ann_dir, filename)
                    ann = libfann.neural_net()
                    ann.create_from_file(ann_path)
                    self.trained_anns.append(ann)
        else:
            slope = 0.65722219398100967
            intercept = 0.099529774137723237
            self.ideal_q_guess = slope * self.initial_q + intercept
            # print "Initial Q guess: " + str(self.ideal_q_guess)

            exp_coefs = [
                6.58953834,
                29.54967305,
                6.00895362,
                -40.3269125
            ]
            exp_p = lambda x, a, b, c, d: -a * np.exp(-b * x + c) + d
            exp = lambda x: exp_p(x, *exp_coefs)
            self.parabola_coef = exp(self.ideal_q_guess)


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
            self.phenotype = phenotype
            s = phenotype.network_shape
            # +1 because of bias neurons in input and hidden layers
            # self.length = ((s[0] + 1) * s[1]) + s[1] + 1

            self.length = 0
            for i in xrange(0, len(s) - 1):
                self.length += (s[i] + 1) * s[i + 1]

        def __call__(self, *args, **kwargs):
            if self.phenotype.init_method == 'uniform':
                min_val = -1.0
                max_val = 1.0
                return RealChromosome(
                    self.length, min_val, max_val)
            elif self.phenotype.init_method == 'normal':
                return RealStatChromosome(
                    self.length,
                    self.phenotype.stats['mean'],
                    self.phenotype.stats['var'])

    """
    GA solution representing MLP-based image filter
    """
    class _Individual(Individual):
        def __init__(self, phenotype, *args, **kwargs):
            self.phenotype = phenotype
            # self.mlp = None
            
            # Initialize network
            self.mlp = libfann.neural_net()
            self.mlp.create_standard_array(self.phenotype.network_shape)

            self.filtered_image = None
            super(NeuralFilterMLP._Individual, self).__init__(*args, **kwargs)

        def _decode(self, chromosome):
            """
            Reconstruct neural network from real-coded chromosome
            """
            # Initialize network
            # self.mlp = libfann.neural_net()
            # self.mlp.create_standard_array(self.phenotype.network_shape)

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

            if self.phenotype.fitness_func == 'ann':
                # Average ann outputs
                self.filtered_image = filter_fann(self.phenotype.source_image, self.mlp)
                filtered_q = metrics.q_py(self.filtered_image)
                # print filtered_q
                return np.mean([
                    ann.run([self.phenotype.initial_q, filtered_q])
                    for ann in self.phenotype.trained_anns
                ])
            else:
                self.filtered_image = filter_fann(self.phenotype.source_image, self.mlp)
                filtered_q = metrics.q_py(self.filtered_image)
                # print self.phenotype.ideal_q_guess, filtered_q
            
                parabola_x = lambda x: self.phenotype.parabola_coef * (x - self.ideal_q_guess)**2 + 1.0
                fitness = self.phenotype.parabola_x(filtered_q)

                # delta_q = filtered_q - self.phenotype.initial_q
                # Sigmoid
                # fitness = 1.0 / (1.0 + math.exp(-1.0 * delta_q / self.phenotype.initial_q))
                # print delta_q
                # print "Initial Q: " + str(self.phenotype.initial_q) + ", filtered Q: " + str(filtered_q)
                return fitness
