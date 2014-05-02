from lib.chromosomes.binary import BinaryChromosome
from lib.solution import Solution, SolutionFactory
from lib.helpers import Helpers
from imaging.filter_call import FilterCall
import numpy as np


class FilterSequenceSolution(Solution):
    def __init__(
            self,
            evaluator,
            encoding_bits,
            seq_length,
            sequence=[]):
        self.evaluator = evaluator
        self.encoding_bits = encoding_bits
        self.seq_length = seq_length
        self.sequence = sequence

    def encode(self):
        """
        Represent each filter from the sequence as bit string
        """
        binary_string = [
            Helpers.int_to_bin(idx, self.encoding_bits)
            for idx in self.sequence
        ]
        return BinaryChromosome(content=binary_string)

    def decode(self, chromosome):
        """
        Parse filter sequence from binary chromosome
        """
        self.sequence = [
            Helpers.bin_to_int(encoded_filter)
            for encoded_filter in Helpers.enumerate_chunks(
                chromosome,
                self.encoding_bits)
        ]
        return self

    @property
    def fitness(self):
        return self.evaluator.fitness(self.sequence)

    def initialize_chromosome(self):
        return BinaryChromosome(
            length=self.encoding_bits * self.seq_length)


class FilterSequenceEvaluator(SolutionFactory):
    SEQUENCE_LENGTH = 30

    def __init__(
            self,
            filter_calls,
            source_images,
            target_images,
            color_planes=3):
        # All available filters represented as functions
        self.filter_calls = filter_calls

        # Nearest power of 2 for current filter call count
        # I.e. 68 filters need 7 bits (0-127) to be encoded
        self.encoding_bits = self._nearest_2_power(len(self.filter_calls))

        # Images before filtering
        self.source_images = source_images

        # Images expected after filtering
        self.target_images = target_images

    def create(self):
        return FilterSequenceSolution(
            self,
            self.encoding_bits,
            self.SEQUENCE_LENGTH)

    def fitness(self, sequence):
        """
        Average fitness of all source/target image pairs
        """
        image_count = len(self.source_images)
        fitness_sum = sum([
            self._fitness_one(
                self.source_images[i],
                self.target_images[i],
                sequence)
            for i in xrange(0, image_count)
        ])
        return fitness_sum / image_count

    def call_list(self, sequence):
        """
        Callable filter list by sequence indexes
        """
        return [
            self.filter_calls[idx]
            for idx in sequence
            # HACK HACK HACK
            if idx < len(self.filter_calls)
        ]

    def _filter_image(self, image, sequence):
        """
        Executes filter sequence on input image
        and returns result.
        Image - list of numpy arrays (for each color plane).
        """
        return FilterCall.run_sequence(
            image,
            self.call_list(sequence)
        )

    def _fitness_one(self, source_image, target_image, sequence):
        """
        Fitness function value for one image
        """
        filtered_image = self._filter_image(source_image, sequence)
        diff = self._image_diff(target_image, filtered_image)
        plane_count = len(source_image)
        # REFACTOR THIS!
        height, width = source_image[0].shape[0], source_image[0].shape[1]
        return 1.0 - float(diff) / (plane_count * width * height * 255)

    def _image_diff(self, image1, image2):
        """
        Sum of pixel differences
        Images - 2d numpy arrays
        """
        diff_sum = 0
        plane_count = len(image1)
        for i in xrange(0, plane_count):
            # Difference of each color plane
            diff_sum += np.sum(
                np.absolute(np.subtract(image1, image2))
            )
        return diff_sum

    def _nearest_2_power(self, integer):
        i = 1
        while 2**i < integer:
            i += 1
        return i
