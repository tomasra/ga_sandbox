import random
import numpy as np


class Helpers(object):

    @staticmethod
    def int_to_bin(integer, length=None):
        """
        Converts integer to binary string of optionally specified length
        """
        binary = "{0:b}".format(integer)
        if length:
            if length < len(binary):
                raise ValueError("Invalid length")
            else:
                binary = binary.zfill(length)
        return binary

    @staticmethod
    def bin_to_int(binary):
        """
        Converts binary string to integer or returns none if binary is Invalid
        """
        try:
            # In case a binary numpy array gets passed
            if isinstance(binary, np.ndarray):
                binary = ''.join([str(char) for char in binary])
            return int(binary, 2)
        except ValueError:
            return None

    @staticmethod
    def char_index_to_bin(sequence, chars, length=None):
        """
        Returns binary representation of index of first char found in sequence,
        or None, if chars are not found
        """
        for char in chars:
            try:
                return Helpers.int_to_bin(sequence.index(char), length)
            except Exception:
                pass
        return None

    @staticmethod
    def enumerate_chunks(enumerable, length=1):
        """
        Enumerates list by chunks of specified length
        For example, enumerate_chunks("abcd", 2) would yield "ab" and "cd"
        Last chunk is returned regardless of its length
        (could be lower that length param value)
        """
        i = 0
        while i < len(enumerable):
            yield enumerable[i:i + length]
            i += length

    @staticmethod
    def flip_bit(value):
        """
        Flips bit (passed as character) value
        """
        if str(value) == "0":
            return "1"
        elif str(value) == "1":
            return "0"
        else:
            return None

    @staticmethod
    def random_bit_string(length):
        """
        Generates a random binary string of specified length
        """
        return "{0:b}".format(random.getrandbits(length)).zfill(length)
