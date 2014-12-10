from core.individual import Individual
from core.chromosomes import IntegerChromosome
from projects.denoising.imaging.filter_call import FilterCall
import projects.denoising.imaging.utils as iu
from projects.denoising.imaging.image import Histogram


class _FilterSequence(Individual):
    """
    Common code for any filter sequence phenotype
    """
    filter_calls = []

    def __init__(self, *args, **kwargs):
        # Decoded chromosome data
        self.filter_sequence = None

        # All available filter calls according to image channel count.
        # Only necessary to populate this once.
        # TODO: non ideal workaround, fix this.
        if not _FilterSequence.filter_calls:
            _FilterSequence.filter_calls = FilterCall.all(
                len(self.source_image.channels))

        super(_FilterSequence, self).__init__(*args, **kwargs)

    def __iter__(self):
        return self.filter_sequence.__iter__()

    def __len__(self):
        return len(self.filter_sequence)

    def _decode(self, chromosome):
        """
        Collect filter calls by indexes from integer chromosome
        """
        self.filter_sequence = [
            self.filter_calls[int(idx)]
            for idx in chromosome
        ]
        return self


class _FilterSequenceUnknownTarget(_FilterSequence):
    """
    Required arguments from FilterSequence construction:
    : sequence_length - filter count in solution
    : source_image - initial noisified image
    """
    # 10% black / 90% white pixels
    IDEAL_HIST_BW_RATIO = 0.1
    HISTOGRAM_WEIGHT = 0.5
    REGION_WEIGHT = 0.5

    def __init__(self, *args, **kwargs):
        # Target is a black/white histogram, instead of specific image
        self.target_histogram = Histogram.binary(
            self.source_image.pixels, self.IDEAL_HIST_BW_RATIO)
        # Maximum possible histogram difference for current image size
        self.max_histogram_diff = Histogram.max_diff(
            self.source_image.pixels)
        super(_FilterSequenceUnknownTarget, self).__init__(*args, **kwargs)

    def _calculate_fitness(self):
        """
        Fitness function value when the target is unknown
        Compares filtered image histogram with 'ideal' one
        defined by text-to-background pixel ratio.
        Also takes into account the number of connected regions in
        filtered image.
        """
        filtered_image = self.source_image.run_filters(self.filter_sequence)

        # Histogram comparison
        hist_diff = filtered_image.histogram - self.target_histogram
        fitness_val_hist = (
            self.max_histogram_diff - hist_diff) / self.max_histogram_diff

        # Connected regions
        region_count = iu.connected_regions(filtered_image)
        fitness_val_regions = 0.0 if region_count == 0 else 1.0 / region_count

        # Final fitness value from two components
        fitness_val = (
            fitness_val_hist * self.HISTOGRAM_WEIGHT +
            fitness_val_regions * self.REGION_WEIGHT
        )
        return fitness_val


class _FilterSequenceKnownTarget(_FilterSequence):
    """
    Required arguments from FilterSequence construction:
    : sequence_length - filter count in solution
    : source_image - initial nosified image
    : target_image - corresponding ideal binary image
    """
    def __init__(self, *args, **kwargs):
        super(_FilterSequenceKnownTarget, self).__init__(*args, **kwargs)

    def _calculate_fitness(self):
        """
        Pixel difference between target and filtered images
        translated into [0, 1] range.
        Higher fitness value corresponds to higher image similarity.
        """
        filtered_image = self.source_image.run_filters(self.filter_sequence)
        pixel_diff = self.target_image.pixel_diff(filtered_image)
        fitness = 1.0 - float(pixel_diff) / filtered_image.max_diff
        return fitness


class FilterSequence(object):
    """
    Here goes some black magic: instantiating this class does not return
    an instance but rather another class, depending on keyword arguments.
    This allows using different phenotypes in GA just by changing
    which args are passed in creation of FilterSequence.
    """
    def __new__(cls, *args, **kwargs):
        # Returns one of two previously defined classes
        if 'target_image' in kwargs and kwargs['target_image'] is not None:
            klass = _FilterSequenceKnownTarget

        else:
            klass = _FilterSequenceUnknownTarget

        # Add keyword arguments as class attributes
        for key, value in kwargs.iteritems():
            setattr(klass, key, value)
        return klass
