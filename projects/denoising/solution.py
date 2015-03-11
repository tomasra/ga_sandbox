from core.individual import Individual
from core.chromosomes import IntegerChromosome
import projects.denoising.imaging.analysis as analysis
import projects.denoising.imaging.noises as noises
from projects.denoising.imaging.filter_call import FilterCall
from projects.denoising.imaging.image import Histogram
from projects.denoising.imaging.char_drawer import CharDrawer


TEXT_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (255, 255, 255)


def generate_images(noise_type, noise_param):
    """
    Create source image (with noise) and clean target image
    """
    chars = CharDrawer(
        image_size=40,
        char_size=36,
        text_color=TEXT_COLOR,
        bg_color=BACKGROUND_COLOR)

    target_image = chars.create_colored_char(
        'A', TEXT_COLOR, BACKGROUND_COLOR)
    # Add noise
    if noise_type == 'snp':
        source_image = noises.salt_and_pepper(
            target_image, noise_param)
    elif noise_type == 'gaussian':
        source_image = noises.gaussian(
            target_image, var=noise_param)
    else:
        raise ValueError("Unknown noise type: %s" % noise_type)

    return source_image, target_image


def get_phenotype(params):
    source_image, target_image = generate_images(
        params['noise_type'], params['noise_param'])
    phenotype = FilterSequence(
        genotype=IntegerChromosome(
            length=params['chromosome_length'],
            min_val=0,
            max_val=len(FilterCall.all()) - 1),
        source_image=source_image,
        target_image=target_image)
    return phenotype


class _FilterSequence(Individual):
    """
    Common code for any filter sequence phenotype
    """
    def __init__(self, filter_calls, *args, **kwargs):
        # All available filter calls according to image channel count.
        self.filter_calls = filter_calls

        # Decoded chromosome data
        self.filter_sequence = None
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

    def __init__(self, source_image, *args, **kwargs):
        self.source_image = source_image

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
        region_count = analysis.connected_regions(filtered_image)
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
    : source_image - initial nosified image
    : target_image - corresponding ideal binary image
    """
    def __init__(self, source_image, target_image, *args, **kwargs):
        self.source_image = source_image
        self.target_image = target_image
        super(_FilterSequenceKnownTarget, self).__init__(*args, **kwargs)

    def _calculate_fitness(self):
        """
        Pixel difference between target and filtered images
        translated into [0, 1] range.
        Higher fitness value corresponds to higher image similarity.
        """
        filtered_image_channels = self.source_image.run_filters(
            self.filter_sequence,
            return_channels=True)
        pixel_diff = self.target_image.pixel_diff_channels(
            filtered_image_channels)
        fitness = 1.0 - float(pixel_diff) / self.target_image.max_diff
        return fitness


class FilterSequence(object):
    """
    Kind-of-a class factory for different types of solutions
    """
    def __init__(self, genotype, source_image, target_image=None):
        self.genotype = genotype
        self.source_image = source_image
        self.target_image = target_image
        self.filter_calls = FilterCall.all(
            len(self.source_image.channels))

    def __call__(self, *args, **kwargs):
        """
        Return one of two individual instances, depending on
        target image availability
        """
        if self.target_image is None:
            return _FilterSequenceUnknownTarget(
                self.source_image,
                self.filter_calls,
                self.genotype,
                *args, **kwargs)
        else:
            return _FilterSequenceKnownTarget(
                self.source_image,
                self.target_image,
                self.filter_calls,
                self.genotype,
                *args, **kwargs)
