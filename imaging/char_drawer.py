import os
import random as rnd
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class CharDrawer(object):
    def __init__(
            self,
            image_size=50,
            char_size=36,
            random_colors=True):
        font_path = os.path.join(os.getcwd(), "imaging/fonts/FreeSans.ttf")
        self.font = ImageFont.truetype(font_path, char_size)
        self.image_size = image_size
        self.char_size = char_size
        if random_colors:
            self.text_color = (
                rnd.randint(0, 255),
                rnd.randint(0, 255),
                rnd.randint(0, 255)
            )
            self.bg_color = (
                rnd.randint(0, 255),
                rnd.randint(0, 255),
                rnd.randint(0, 255)
            )

    def create_pair(self, character):
        """
        Generates a pair of black/white and colored
        charater images
        """
        char_blacknwhite = self._create_character(
            character,
            text_color=(0, 0, 0),
            bg_color=(255, 255, 255))
        char_color = self._create_character(
            character,
            text_color=self.text_color,
            bg_color=self.bg_color)
        return (
            self._image_to_numpy(char_blacknwhite),
            self._image_to_numpy(char_color)
        )

    def _create_character(self, character, text_color, bg_color):
        """
        Return character as Pillow Image object
        """
        image = Image.new(
            'RGB',
            (self.image_size, self.image_size),
            bg_color)
        draw = ImageDraw.Draw(image)
        position = (self.image_size - self.char_size) / 2
        draw.text(
            (position, position),
            character,
            # fill=(0, 0, 0),
            fill=text_color,
            font=self.font)
        return image

    @staticmethod
    def _image_to_numpy(image):
        """
        Converts Pillow Image object to numpy array list (RGB)
        """
        np_data = np.asarray(image, dtype=np.uint8)
        return [
            np_data[:, :, i]
            for i in xrange(np_data.shape[2])
        ]
