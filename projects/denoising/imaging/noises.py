import numpy as np
from skimage.util import random_noise, img_as_ubyte
from projects.denoising.imaging.image import Image
from sklearn.decomposition import PCA
from skimage.filter import sobel, vsobel, hsobel
from skimage import data

# Override if needed
_rng_seed = None


def salt_and_pepper(image, amount=0.05):
    noisified = random_noise(
        image, mode='s&p', seed=_rng_seed, amount=amount)
    return Image(img_as_ubyte(noisified))


def gaussian(image, mean=0.0, var=0.01):
    noisified = random_noise(
        image, mode='gaussian', seed=_rng_seed,
        mean=mean, var=var)
    return Image(img_as_ubyte(noisified))


class GaussianEstimator(object):
    def __init__(self):
        pass


    def _conv_mtx2(self, kernel, patch_size):
        """
        TODO: figure out what the code below actually does, lol
        """
        s = kernel.shape
        m = patch_size - s[0] + 1
        n = patch_size - s[1] + 1
        mtx = np.zeros((m * n, patch_size ** 2))

        k = 0
        for i in xrange(0, m):
            for j in xrange(0, n):
                for p in xrange(0, s[0]):
                    l1 = (i + p) * patch_size + j
                    l2 = l1 + s[1]
                    mtx[k, l1:l2] = kernel[p, :]
                k += 1
        return mtx


    def run(self, image, patch_size=5, conf=1e-6):
        """
        Image - NxM numpy array (single color channel)
        """
        vector_length = patch_size ** 2

        # Gradient filter kernels
        kh = np.array([[-0.5, 0, 0.5]])
        kv = kh.T

        dh = self._conv_mtx2(kh, patch_size)
        dv = self._conv_mtx2(kv, patch_size)
        dd = dh.T.dot(dh) + dv.T.dot(dv)
        
        rank = np.linalg.matrix_rank(dd)
        trace = np.trace(dd)

        # Channels start here
        

        # Horizontal gradients
        
        return trace
        # for x0, y0 in self._patch_points(image, patch_size):
        #     patch = image[y0:(y0 + patch_size), x0:(x0 + patch_size)]
        #     # Remove last dimension
        #     patch = patch.reshape((patch.shape[0], patch.shape[1]))

        #     # Flatten by cols
        #     vector_cols = None
        #     # gradient = sobel(patch)
        #     # cov_grad = np.transpose(gradient).dot(gradient)
        #     dh = hsobel(patch)
        #     dv = vsobel(patch)
        #     cov_grad = np.cov(dh, dv)
        #     return cov_grad


        # pca = PCA()
        # data = np.array([
        #     # First slice the rows (y), then columns (x)
        #     image[y0:(y0 + patch_size), x0:(x0 + patch_size)].reshape(vector_length)
        #     for x0, y0 in self._patch_points(image, patch_size)
        # ])
        # return pca.fit(data)

    def _imfilter(img, kernel):
        """
        img - NxM numpy array
        """
        pass

    def _patch_points(self, image, patch_size):
        """
        Pixel-by-pixel iteration through image in NxN sized patches.
        """
        if len(image.channels) > 1:
            raise ValueError("Image must have a single color channel")

        if patch_size > image.width or patch_size > image.height:
            raise ValueError("Patch size too large")

        # Top-left points of all patches
        for x0 in xrange(0, image.width - patch_size + 1):
            for y0 in xrange(0, image.height - patch_size + 1):
                yield x0, y0


class SNPEstimator(object):
    def __init__(self):
        pass

    def entropy(self, ref_image, target_image):
        """
        Images must be grayscale and have equal dimensions
        """
        ref_hist = np.histogram(ref_image, bins=256)[0]
        target_hist = np.histogram(target_image, bins=256)[0]
        total_pixels = ref_image.shape[0] * ref_image.shape[1]
        entropy = 0.0
        for x in ref_hist:
            for y in target_hist:
                p_xy = (float(x) / total_pixels) * (float(y) / total_pixels)
                if p_xy > 0:
                    entropy += (p_xy * np.log2(p_xy))
        entropy = -entropy
        return entropy

    # def entropy(*X):
    #     """
    #     http://blog.biolab.si/2012/06/15/computing-joint-entropy-in-python/
    #     """
    #     return = np.sum(-p * np.log2(p) if p > 0 else 0 for p in
    #         (np.mean(reduce(np.logical_and, (predictions == c for predictions, c in zip(X, classes))))
    #             for classes in itertools.product(*[set(x) for x in X])))
