import unittest
from mock import patch
from core.crossovers import OnePointCrossover
from core.chromosomes import BinaryChromosome


class OnePointCrossoverTest(unittest.TestCase):
    def test_crossover_does_happen(self):
        """
        One point crossover - does happen
        """
        chromo1 = BinaryChromosome(content="00001111")
        chromo2 = BinaryChromosome(content="11110000")
        crossover = OnePointCrossover(0.9)

        # swap last two bits
        with patch('random.random', return_value=0.6):
            with patch('random.randint', return_value=6):
                # descendant = chromo1.crossover(chromo2, 0.9)
                desc1, desc2 = crossover.run(chromo1, chromo2)

        chromo1_str = "".join([str(c) for c in chromo1.content])
        chromo2_str = "".join([str(c) for c in chromo2.content])
        desc1_str = "".join([str(c) for c in desc1.content])
        desc2_str = "".join([str(c) for c in desc2.content])
        self.assertEqual(chromo1_str, "00001111")
        self.assertEqual(chromo2_str, "11110000")
        self.assertEqual(desc1_str, "00001100")
        self.assertEqual(desc2_str, "11110011")

    def test_crossover_does_not_happen(self):
        """
        One point crossover - does not happen
        """
        chromo1 = BinaryChromosome(content="00001111")
        chromo2 = BinaryChromosome(content="11110000")
        crossover = OnePointCrossover(0.3)

        # don't do crossover
        with patch('random.random', return_value=0.7):
            desc1, desc2 = crossover.run(chromo1, chromo2)

        chromo1_str = "".join([str(c) for c in chromo1.content])
        chromo2_str = "".join([str(c) for c in chromo2.content])
        desc1_str = "".join([str(c) for c in desc1.content])
        desc2_str = "".join([str(c) for c in desc2.content])
        self.assertEqual(chromo1_str, "00001111")
        self.assertEqual(chromo2_str, "11110000")
        self.assertEqual(desc1_str, "00001111")
        self.assertEqual(desc2_str, "11110000")
