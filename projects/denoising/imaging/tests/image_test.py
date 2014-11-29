import numpy as np
import numpy.testing as nptest
import unittest
from projects.denoising.imaging.image import Image, _Channels


class ImageTests(unittest.TestCase):
    def setUp(self):
        # 2x2x3
        self.image1 = Image(np.array([
            [[111, 112, 113], [121, 122, 123]],
            [[211, 212, 213], [221, 222, 223]]
        ]))
        self.image2 = Image(np.array([
            [[114, 115, 116], [124, 125, 126]],
            [[214, 215, 216], [224, 225, 226]]
        ]))

    def test_pixel_diff(self):
        """
        Image: pixel diff
        """
        self.assertEquals(
            self.image1.pixel_diff(self.image2), 3 * 12)


class ChannelTests(unittest.TestCase):
    def setUp(self):
        # Histogram (2D array)
        self.channels_2d = _Channels(np.array([
            [11, 12, 13],
            [21, 22, 23],
        ]))
        # Image (3D array)
        self.channels_3d = _Channels(np.array([
            [[111, 112, 113], [121, 122, 123]],
            [[211, 212, 213], [221, 222, 223]]
        ]))

    def test_getitem(self):
        """
        Channels: get single channel data
        """
        # 2D array
        nptest.assert_array_equal(
            self.channels_2d[2],
            np.array([13, 23]))
        # 3D array
        nptest.assert_array_equal(
            self.channels_3d[1],
            np.array([[112, 122], [212, 222]]))

    def test_setitem(self):
        """
        Channels: set single channel data
        """
        new_2d_channel = np.array([
            13000, 23000
        ])
        new_3d_channel = np.array([
            [11300, 12300],
            [21300, 22300],
        ])
        self.channels_2d[2] = new_2d_channel
        self.channels_3d[2] = new_3d_channel
        nptest.assert_array_equal(self.channels_2d[2], new_2d_channel)
        nptest.assert_array_equal(self.channels_3d[2], new_3d_channel)

    def test_iteration(self):
        """
        Channels: iterate channels
        """
        nptest.assert_array_equal(
            [channel for channel in self.channels_2d],
            np.array([
                [11, 21], [12, 22], [13, 23]
            ]))
        nptest.assert_array_equal(
            [channel for channel in self.channels_3d],
            np.array([
                [[111, 121], [211, 221]],
                [[112, 122], [212, 222]],
                [[113, 123], [213, 223]],
            ]))
