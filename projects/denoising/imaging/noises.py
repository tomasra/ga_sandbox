from skimage.util import random_noise, img_as_ubyte
from projects.denoising.imaging.image import Image

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
