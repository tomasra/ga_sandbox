import os
import random as rnd
import numpy as np
from projects.denoising.imaging.analysis import binarize
# from PIL import Image, ImageDraw, ImageFont
import PIL
from PIL import ImageFont, ImageDraw
from projects.denoising.imaging.image import Image


class CharDrawer(object):
    def __init__(
            self,
            image_size=50,
            char_size=36,
            text_color=None,
            bg_color=None):
        font_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "fonts/FreeSans.ttf")
        self.font = ImageFont.truetype(font_path, char_size)
        self.image_size = image_size
        self.char_size = char_size

        if text_color and bg_color:
            self.text_color = text_color
            self.bg_color = bg_color
        elif not text_color and not bg_color:
            # Random colors?
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
        else:
            raise ValueError(
                "Both text and background color should be specified")

    def create_pair(self, character):
        """
        Generates a pair of black/white and colored
        charater images
        """
        bw_image = self._create_character(
            character,
            text_color=(0, 0, 0),
            bg_color=(255, 255, 255))
        color_image = self._create_character(
            character,
            text_color=self.text_color,
            bg_color=self.bg_color)

        # color_image = self._image_to_numpy(char_color)
        # bw_image = self._image_to_numpy(char_blacknwhite)
        bw_image = binarize(bw_image)

        return (bw_image, color_image)

    def _create_character(self, character, text_color, bg_color):
        """
        Return image of a single character
        """
        pil_image = PIL.Image.new(
            'RGB',
            (self.image_size, self.image_size),
            bg_color)
        draw = PIL.ImageDraw.Draw(pil_image)
        position = (self.image_size - self.char_size) / 2
        draw.text(
            (position, position),
            character,
            # fill=(0, 0, 0),
            fill=text_color,
            font=self.font)

        # Convert to custom image format (numpy array)
        image = Image(np.array(pil_image))
        return image

    @staticmethod
    def create_colored_char(character, text_color, bg_color):
        """
        Clear color image of specified character
        """
        char_drawer = CharDrawer(
            text_color=text_color,
            bg_color=bg_color
        )
        pair = char_drawer.create_pair(character)
        return pair[1]

    @staticmethod
    def create_binary_char(character):
        """
        Black and white image of specified character
        """
        char_drawer = CharDrawer(
            text_color=(0, 0, 0),
            bg_color=(255, 255, 255)
        )
        pair = char_drawer.create_pair(character)
        return pair[0]

    @staticmethod
    def create_mosaic(images, width, height, borders=True):
        """
        Makes a width x height mosaic
        from provided images (lists of numpy arrays)
        """
        shapes = set([
            plane.shape
            for image in images
            for plane in image
        ])
        planes = set([len(image) for image in images])
        if len(planes) > 1:
            raise ValueError(
                "Images should have the same number of color planes")
        if len(shapes) > 1:
            raise ValueError(
                "Images should of the same dimensions")
        elif len(images) > (width * height):
            raise ValueError(
                "Image count is larger than mosaic dimensions")
        else:
            shape, planes = list(shapes)[0], list(planes)[0]
            image_width, image_height = shape[1], shape[0]
            mosaic_width = image_width * width
            mosaic_height = image_height * height
            mosaic = [
                np.zeros((mosaic_height, mosaic_width), dtype=np.uint8)
                for i in xrange(planes)
            ]
            for idx, image in enumerate(images):
                # Image indexes (0-based) in the mosaic
                mosaic_x, mosaic_y = idx % width, idx / width
                top_left_x = mosaic_x * image_width
                top_left_y = mosaic_y * image_height
                # Copy all planes
                for plane_idx in xrange(planes):
                    mosaic[plane_idx][
                        top_left_y:top_left_y + image_height,
                        top_left_x:top_left_x + image_width
                    ] = image[plane_idx]
                if borders:
                    # Right and bottom border for each image
                    for plane_idx in xrange(planes):
                        # Right
                        mosaic[plane_idx][
                            top_left_y:top_left_y + image_height,
                            # "Index out of bounds" happens without this one
                            top_left_x + image_width - 1
                        ] = 0
                        # Bottom
                        mosaic[plane_idx][
                            top_left_y + image_height - 1,
                            top_left_x:top_left_x + image_width
                        ] = 0
            # Top and left border for whole mosaic
            if borders:
                for plane_idx in xrange(planes):
                    mosaic[plane_idx][0, 0:mosaic_width] = 0
                    mosaic[plane_idx][0:mosaic_height, 0] = 0
            return mosaic
