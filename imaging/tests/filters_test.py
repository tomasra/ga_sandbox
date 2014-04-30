import unittest
import numpy as np
import imaging.filters as flt


class FiltersTests(unittest.TestCase):
    def test_inversion(self):
        """
        One argument filter - inversion
        """
        img = np.array([[100, 150], [200, 250]])
        expected = np.array([[155, 105], [55, 5]])
        actual = flt.inversion(img)
        self.assertTrue(np.allclose(actual, expected))

    def test_logical_sum(self):
        """
        Two argument filter - logical sum
        """
        img1 = np.array([[1, 2], [7, 8]])
        img2 = np.array([[5, 6], [3, 4]])
        # maximum of two color planes
        expected = np.array([[5, 6], [7, 8]])
        actual = flt.logical_sum(img1, img2)
        self.assertTrue(np.allclose(actual, expected))

    def test_logical_product(self):
        """
        Two argument filter - logical product
        """
        img1 = np.array([[1, 2], [7, 8]])
        img2 = np.array([[5, 6], [3, 4]])
        # minimum of two color planes
        expected = np.array([[1, 2], [3, 4]])
        actual = flt.logical_product(img1, img2)
        self.assertTrue(np.allclose(actual, expected))

    def test_algebraic_sum(self):
        """
        Two argument filter - algebraic sum
        """
        img1 = np.array([[10], [70]])
        img2 = np.array([[50], [30]])
        # sum of two color planes - their product / 255
        expected = np.array([[60 - 2], [100 - 8]])
        actual = flt.algebraic_sum(img1, img2)
        self.assertTrue(np.allclose(actual, expected))

    def test_algebraic_product(self):
        """
        Two argument filter - algebraic product
        """
        img1 = np.array([[10], [70]])
        img2 = np.array([[50], [30]])
        # product of two color planes / 255
        expected = np.array([[2], [8]])
        actual = flt.algebraic_product(img1, img2)
        self.assertTrue(np.allclose(actual, expected))

    def test_bounded_sum(self):
        """
        Two argument filter - bounded sum
        """
        img1 = np.array([[100, 150], [200, 250]])
        img2 = np.array([[100, 100], [100, 100]])
        # g = sum of two color planes
        # if g > 255: g = 255
        expected = np.array([[200, 250], [255, 255]])
        actual = flt.bounded_sum(img1, img2)
        self.assertTrue(np.allclose(actual, expected))

    def test_bounded_product(self):
        """
        Two argument filter - bounded sum
        """
        img1 = np.array([[10, 20], [30, 40]])
        img2 = np.array([[10, 10], [10, 10]])
        # product of two color planes - 255
        # if g < 0: g = 0
        expected = np.array([[0, 0], [45, 145]])
        actual = flt.bounded_product(img1, img2)
        self.assertTrue(np.allclose(actual, expected))
