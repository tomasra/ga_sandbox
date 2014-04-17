from lib.helpers import Helpers
import unittest


class HelpersTests(unittest.TestCase):
    def test_bin_to_int(self):
        """
        Helpers - convert binary string to integer
        """
        self.assertEqual(Helpers.bin_to_int("111"), 7)
        self.assertEqual(Helpers.bin_to_int("00111"), 7)
        self.assertEqual(Helpers.bin_to_int("i like turtles"), None)

    def test_char_index_to_bin(self):
        coding = "abcd"
        self.assertEqual(Helpers.char_index_to_bin(coding, "c"), "10")
        self.assertEqual(Helpers.char_index_to_bin(coding, "c", 4), "0010")
        self.assertEqual(Helpers.char_index_to_bin(coding, "E"), None)

    def test_char_index_to_bin_with_list(self):
        coding = "abcd$"
        # index of $ in binary
        self.assertEqual(
            Helpers.char_index_to_bin(coding, ['e', '$'], 4), "0100")
        self.assertEqual(Helpers.char_index_to_bin(coding, "ef"), None)

    def test_enumerate_chunks_with_string(self):
        actual1 = [chunk for chunk in Helpers.enumerate_chunks("abcd", 2)]
        actual2 = [chunk for chunk in Helpers.enumerate_chunks("abcde", 2)]
        actual3 = [chunk for chunk in Helpers.enumerate_chunks("abc")]
        self.assertSequenceEqual(actual1, ["ab", "cd"])
        self.assertSequenceEqual(actual2, ["ab", "cd", "e"])
        self.assertSequenceEqual(actual3, ["a", "b", "c"])

    def test_flip_bit(self):
        self.assertEqual(Helpers.flip_bit('0'), '1')
        self.assertEqual(Helpers.flip_bit(0), '1')
        self.assertEqual(Helpers.flip_bit('1'), '0')
        self.assertEqual(Helpers.flip_bit(1), '0')
        self.assertEqual(Helpers.flip_bit('$'), None)

    def test_random_bit_string(self):
        rand_seq = Helpers.random_bit_string(64)
        self.assertEqual(len(rand_seq), 64)
        # omg how do i test this???
        self.assertNotEqual(
            rand_seq,
            "0000000000000000000000000000000000000000000000000000000000000000")
        self.assertNotEqual(
            rand_seq,
            "1111111111111111111111111111111111111111111111111111111111111111")
        self.assertNotEqual(
            rand_seq,
            "1010101010101010101010101010101010101010101010101010101010101010")
        self.assertNotEqual(
            rand_seq,
            "0101010101010101010101010101010101010101010101010101010101010101")
