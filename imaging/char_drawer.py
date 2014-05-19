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
