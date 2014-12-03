import copy
import numpy as np
from scipy import ndimage
from pyemd import emd


class Image(np.ndarray):
    """
    Wrapper around numpy.ndarray with additional image processing routines
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    """
    def __new__(cls, data):
        instance = np.asarray(data).view(cls)
        # Special case for images with only one channel (without 3rd dimension)
        if len(instance.shape) == 2:
            shape_3d = tuple(list(instance.shape) + [1])
            instance = instance.reshape(shape_3d)

        return instance

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # HACK!
        # Histogram creation needs channels, but iterating them creates new
        # arrays and calls this __array_finalize__, thus causing infinite
        # recursion. Workaround here is to create channels from raw np.array
        self.channels = _Channels(self.view(np.ndarray))
        self.histogram = Histogram(self)

    @staticmethod
    def from_channels(channels):
        """
        Stack channels and return a new image
        """
        stacked = None
        for channel in channels:
            if stacked is None:
                stacked = channel
            else:
                stacked = np.dstack([stacked, channel])
        return Image(stacked)

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    @property
    def pixels(self):
        return self.width * self.height

    @property
    def histogram(self):
        return self._histogram

    @histogram.setter
    def histogram(self, value):
        self._histogram = value

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, value):
        self._channels = value

    def run_filters(self, filter_calls):
        """
        Run a sequence of filters on current image
        and return a new image
        """
        image = copy.deepcopy(self)
        for filter_call in filter_calls:
            image = filter_call(image)
        return image

    def pixel_diff(self, other):
        """
        Sum of pixel differences
        Images - 2d numpy arrays
        """
        return np.sum(
            np.absolute(
                self.view(np.ndarray).astype(np.int16) -
                other.view(np.ndarray).astype(np.int16)
            )
        )

    @property
    def max_diff(self):
        """
        Maximum possible pixel difference between two images
        with dimensions of the current image
        """
        diff = 255
        for dim in self.shape:
            diff *= dim
        return diff


class Histogram(object):
    """
    Wrapper around 256x3 numpy array
    """
    BLACK_INDEX = 0
    WHITE_INDEX = 255
    LENGTH = 256

    # For EMD distance computation
    _L1_DISTANCE_MATRIX = np.array([
        abs(x - y)
        for x in xrange(LENGTH)
        for y in xrange(LENGTH)
    ]).reshape(LENGTH, LENGTH).astype(np.float)

    def __init__(self, image=None):
        if image is not None:
            if image.channels is not None:
                # Get histogram of each channel separately
                # and then stack them
                self.data = np.dstack([
                    ndimage.measurements.histogram(
                        channel, 0, 255, self.LENGTH)
                    for channel in image.channels
                ])[0]
            else:
                # No channels?
                self.data = ndimage.measurements.histogram(
                    image, 0, 255, self.LENGTH)
        else:
            # Initialize blank 256x3 array
            self.data = np.zeros((256, 3), dtype=np.int64)
        self._channels = _Channels(self.data, for_histogram=True)

    def __getitem__(self, index):
        # Values of all channels
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __sub__(self, other):
        """
        Earth-mover's distance (EMD) between two histograms.
        Calculated for channels separately and summed up.
        """
        result = sum([
            emd(
                pair[0].astype(np.float),
                pair[1].astype(np.float),
                Histogram._L1_DISTANCE_MATRIX
            )
            for pair in zip(self.channels, other.channels)
        ])
        return result

    @property
    def channels(self):
        return self._channels

    @staticmethod
    def binary(pixel_count, bw_ratio):
        """
        Returns 256x3 histogram with black and white levels
        according to specified black-to-white ratio ([0,1] interval)
        """
        hist = Histogram()
        black_pixels = pixel_count * bw_ratio
        white_pixels = pixel_count - black_pixels
        hist[Histogram.BLACK_INDEX] = [black_pixels] * 3
        hist[Histogram.WHITE_INDEX] = [white_pixels] * 3
        return hist

    @staticmethod
    def max_diff(pixel_count):
        """
        Maximum difference between two histograms of specified image size,
        representing completely white and black images.
        """
        white_hist = Histogram()
        white_hist[Histogram.WHITE_INDEX] = [pixel_count] * 3
        black_hist = Histogram()
        black_hist[Histogram.BLACK_INDEX] = [pixel_count] * 3
        return white_hist - black_hist


class _Channels(object):
    """
    Helper class for convenient access of separate color channels (ie. RGB)
    in images (3D array, i.e. 50x50x3) or histograms (2D array: 256x3)
    """
    def __init__(self, data, for_histogram=False):
        # See data setter
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        # Should be three dimensions for usual images
        self._dimensions = len(self._data.shape)
        self._channel_count = self._data.shape[self._dimensions - 1]

        # Same shape as original data minus the channel dimension
        shape = list(self._data.shape)
        shape.pop()
        self._shape_no_channel = tuple(shape)

    def __len__(self):
        return self._channel_count

    def __getitem__(self, key):
        """
        Return all data of a single channel
        """
        return np.split(
            self._data,
            self._channel_count,
            self._dimensions - 1)[key].reshape(self._shape_no_channel)

    def __setitem__(self, key, channel_data):
        """
        Overwrites specified channel
        """
        # ... is ellipsis notation
        self._data[..., key] = channel_data

    def __iter__(self):
        """
        Iterate through all channels
        """
        # If there is just one channel, yield the image itself
        if len(self) == 1:
            yield self._data
        else:
            channels = np.split(
                self._data,
                self._channel_count,
                self._dimensions - 1)
            for channel in channels:
                yield channel.reshape(self._shape_no_channel)
