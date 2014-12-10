import unittest
import numpy as np
from mock import Mock
from core.chromosomes import BinaryChromosome


class BinaryChromosomeTests(unittest.TestCase):
    def test_generate_random(self):
        """
        Binary chromosome - generate random string of specified length
        """
        chromo = BinaryChromosome(20)
        zeros_count = len([c for c in chromo if c == 0])
        ones_count = len([c for c in chromo if c == 1])
        self.assertEqual(len(chromo), 20)
        # Should have both ones and zeros
        self.assertNotEqual(zeros_count, 0)
        self.assertNotEqual(zeros_count, 20)
        self.assertNotEqual(ones_count, 0)
        self.assertNotEqual(ones_count, 20)

    def test_mutation(self):
        """
        Binary chromosome - mutation does happen to every odd bit
        """
        chromo = BinaryChromosome(0)
        # Normally content isn't supposed to be set this way
        chromo._content = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        class _FakeRandomizer(object):
            def __init__(self):
                self.call_count = 0

            def random_sample(self):
                # Alternate between two values: 0.0002 and 0.5
                self.call_count += 1
                return 0.0002 if self.call_count % 2 == 1 else 0.5

        chromo._randomizer = _FakeRandomizer()
        chromo.mutate(0.001)

        expected = np.array([1, 0, 1, 0, 0, 1, 0, 1])
        self.assertTrue((chromo._content == expected).all())

    def test_split_one_point(self):
        """
        Chromosome: split at one point
        """
        chromo = BinaryChromosome(0)
        chromo._content = np.array([1, 1, 1, 0, 0, 0])
        part1, part2 = chromo.split(2)
        self.assertTrue((part1._content == np.array([1, 1])).all())
        self.assertTrue((part2._content == np.array([1, 0, 0, 0])).all())

    def test_split_two_points(self):
        """
        Chromosome: split at two points
        """
        chromo = BinaryChromosome(0)
        chromo._content = np.array([1, 1, 0, 0, 1, 1])
        part1, part2, part3 = chromo.split([2, 4])
        self.assertTrue((part1._content == np.array([1, 1])).all())
        self.assertTrue((part2._content == np.array([0, 0])).all())
        self.assertTrue((part3._content == np.array([1, 1])).all())


class VarIntegerChromosomeTests(unittest.TestCase):
    pass
