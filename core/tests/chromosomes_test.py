import unittest
from mock import patch
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
        chromo = BinaryChromosome(content="00001111")
        random_vals = [0.0002, 0.5, 0.0002, 0.5, 0.0002, 0.5, 0.0002, 0.5]
        # Flip every second bit
        with patch('random.random', side_effect=random_vals):
            chromo.mutate(0.001)
        content_str = "".join([str(c) for c in chromo.content])
        self.assertEqual(content_str, "10100101")
