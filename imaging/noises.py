import numpy as np
import random as rnd


def salt_and_pepper(image, probability):
    if isinstance(image, list):
        return [
            salt_and_pepper(color_plane, probability)
            for color_plane in image
        ]
    else:
        # Add salt and pepper
        new_image = np.empty_like(image)
        new_image[:] = image
        for i in xrange(0, image.shape[0]):
            for j in xrange(0, image.shape[1]):
                if rnd.random() < probability:
                    # Make the noise here
                    # either white or black pixel
                    bit = rnd.randint(0, 1)
                    new_image[i][j] = 255 * bit
        return new_image


def gaussian(image, mu=0.0, sigma=10.0):
    if isinstance(image, list):
        return [
            gaussian(color_plane, mu, sigma)
            for color_plane in image
        ]
    else:
        # Add gaussian noise
        new_image = np.empty_like(image)
        new_image[:] = image
        for i in xrange(0, image.shape[0]):
            for j in xrange(0, image.shape[1]):
                point = image[i][j] + rnd.gauss(mu, sigma)
                # Pixel value should be in [0; 255] interval
                if point > 255:
                    new_image[i][j] = 255
                elif point < 0:
                    new_image[i][j] = 0
                else:
                    new_image[i][j] = point
        return new_image


def poisson(image):
    pass


def speckle(image):
    pass
