from core.individual import Individual
from core.chromosomes import IntegerChromosome
from projects.denoising.imaging.filter_call import FilterCall
import projects.denoising.imaging.utils as iu
from projects.denoising.imaging.image import Histogram
from projects.denoising.imaging.analysis import connected_regions


class _FilterSequence(Individual):
    """
    Common code for any filter sequence phenotype
    """
    def __init__(self):
        # Decoded chromosome data
        self.filter_sequence = None

        # All available filter calls according to image channel count
        self.filter_calls = FilterCall.all(
            len(self.source_image.channels))

        super(_FilterSequence, self).__init__()

    def _decode(self, chromosome):
        """
        Collect filter calls by indexes from integer chromosome
        """
        self.filter_sequence = [
            self.filter_calls[int(idx)]
            for idx in chromosome
        ]
        return self

    def _initialize_chromosome(self):
        """
        Random array of integers, representing filter call indexes
        """
        return IntegerChromosome(
            min_val=0,
            max_val=len(self.filter_calls) - 1,
            initial_length=self.sequence_length)

    def _calculate_fitness(self):
        # Leave for subclasses
        raise NotImplementedError


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

    def __init__(self):
        # Target is a black/white histogram, instead of specific image
        self.target_histogram = Histogram.binary(
            self.source_image.pixels, self.IDEAL_HIST_BW_RATIO)
        # Maximum possible histogram difference for current image size
        self.max_histogram_diff = Histogram.max_diff(
            self.source_image.pixels)
        super(_FilterSequenceUnknownTarget, self).__init__()

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
    def __init__(self):
        super(_FilterSequenceKnownTarget, self).__init__()

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


# class _FilterSequenceSolution(Solution):
#     def __init__(
#             self,
#             evaluator,
#             filter_count,
#             seq_length,
#             sequence=[]):
#         self.evaluator = evaluator
#         # self.encoding_bits = encoding_bits
#         self.filter_count = filter_count
#         self.seq_length = seq_length
#         self.sequence = sequence
#         # Cached fitness value
#         self.cached_fitness = None

#     # def encode(self):
#     #     """
#     #     Represent each filter from the sequence as bit string
#     #     """
#     #     return IntegerChromosome(
#     #         1, self.filter_count,
#     #         content=self.sequence)

#     def decode(self, chromosome):
#         """
#         Parse filter sequence from binary chromosome
#         """
#         self.sequence = chromosome.content
#         # Reset cached fitness
#         self.cached_fitness = None
#         return self

#     @property
#     def fitness(self):
#         if not self.cached_fitness:
#             self.cached_fitness = self.evaluator.fitness(
#                 self.sequence
#             )
#         return self.cached_fitness

#     def initialize_chromosome(self):
#         return IntegerChromosome(
#             1, self.filter_count,
#             length=self.seq_length
#         )

#     def __repr__(self):
#         s = "["
#         s += ", ".join([
#             str(a)
#             for a in self.sequence
#             if a < self.filter_count
#         ])
#         s += "]"
#         return s


# class _FilterSequenceEvaluator(SolutionFactory):
#     # _text_color_expected = np.array([255, 255, 255])
#     # _bg_color_expected = np.array([0, 0, 0])
#     # _region_count_expected = 1
#     # _max_color_diff = np.linalg.norm(
#     #     _text_color_expected - _bg_color_expected
#     # )

#     def __init__(
#             self,
#             filter_calls,
#             input_images,
#             target_images,
#             sequence_length=20,
#             color_planes=3,
#             ttb_ratio=0.1):
#         # All available filters represented as functions
#         self.filter_calls = filter_calls
#         self.input_images = input_images    # Images before filtering
#         self.target_images = target_images  # Images expected after filtering
#         # Number of filters in the solution sequence
#         self.sequence_length = sequence_length
#         self.color_planes = color_planes

#         # Image count
#         # if len(input_images) != len(target_images):
#         #     raise ValueError("Input and target image counts do not match")
#         # else:
#         self.image_count = len(input_images)

#         self._ideal_histogram = iu.ideal_histogram(
#             input_images[0], ttb_ratio)
#         self._max_histogram_diff = iu.max_histogram_diff(
#             input_images[0])

#     # def create(self):
#     #     return FilterSequenceSolution(
#     #         self,
#     #         len(self.filter_calls),
#     #         self.sequence_length)

#     # def fitness(self, sequence):
#     #     """
#     #     Average fitness of all source/target image pairs
#     #     """
#     #     if self.target_images:
#     #         # Easy case - target image(s) is/are known
#     #         fitness_sum = sum([
#     #             self._fitness_one(
#     #                 self.input_images[i],
#     #                 self.target_images[i],
#     #                 sequence)
#     #             for i in xrange(self.image_count)
#     #         ])
#     #         return fitness_sum / self.image_count
#     #     else:
#     #         # Hard case - no target image to compare against
#     #         fitness_sum = sum([
#     #             self._fitness_unknown_target(source_image, sequence)
#     #             for source_image in self.input_images
#     #         ])
#     #         return fitness_sum / self.image_count

#     # def call_list(self, sequence):
#     #     """
#     #     Callable filter list by sequence indexes
#     #     """
#     #     return [
#     #         self.filter_calls[idx]
#     #         for idx in sequence
#     #         # HACK HACK HACK
#     #         if idx < len(self.filter_calls)
#     #     ]

#     # def _filter_image(self, image, sequence):
#     #     """
#     #     Executes filter sequence on input image
#     #     and returns result.
#     #     Image - list of numpy arrays (for each color plane).
#     #     """
#     #     return FilterCall.run_sequence(
#     #         image,
#     #         self.call_list(sequence)
#     #     )

#     # def _fitness_unknown_target(self, source_image, sequence):
#     #     """
#     #     Fitness function value when the target is unknown
#     #     Compares filtered image histogram with 'ideal' one
#     #     defined by text-to-background pixel ratio
#     #     """
#     #     filtered_image = self._filter_image(source_image, sequence)
#     #     actual_histogram = iu.histogram(filtered_image)

#     #     hist_diff = iu.histogram_diff(
#     #         self._ideal_histogram, actual_histogram)
#     #     # print hist_diff, self._max_histogram_diff
#     #     fitness_val_hist = (
#     #         self._max_histogram_diff - hist_diff) / self._max_histogram_diff
#     #     region_count = iu.connected_regions(filtered_image)
#     #     fitness_val_regions = 0.0 if region_count == 0 else 1.0 / region_count

#     #     # Equal weights for both parts
#     #     fitness_val = (
#     #         fitness_val_hist * 0.5 +
#     #         fitness_val_regions * 0.5
#     #     )
#     #     return fitness_val

#     # def _fitness_one(self, source_image, target_image, sequence):
#     #     """
#     #     Fitness function value for one image
#     #     """
#     #     filtered_image = self._filter_image(source_image, sequence)
#     #     diff = self._image_diff(target_image, filtered_image)
#     #     plane_count = len(source_image)
#     #     # REFACTOR THIS!
#     #     height, width = source_image[0].shape[0], source_image[0].shape[1]
#     #     fitness = 1.0 - float(diff) / (plane_count * width * height * 255)
#     #     return fitness

#     # def _image_diff(self, image1, image2):
#     #     """
#     #     Sum of pixel differences
#     #     Images - 2d numpy arrays
#     #     """
#     #     plane_count = len(image1)
#     #     return np.sum([
#     #         # Difference of each color plane
#     #         np.sum(
#     #             np.absolute(
#     #                 np.subtract(
#     #                     image1[i].astype(np.int16),
#     #                     image2[i].astype(np.int16)
#     #                 )
#     #             )
#     #         )
#     #         for i in xrange(plane_count)
#     #     ])
