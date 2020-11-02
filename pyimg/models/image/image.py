import copy

import numpy as np
from PIL import Image

from pyimg.config import constants
from pyimg.models import Matrix


class ImageImpl(Matrix):
    def __init__(self, image):
        super(ImageImpl, self).__init__(image)
        self.width = self.array.shape[0]
        self.height = self.array.shape[1]
        self.channels = self.array.shape[2]

    def convert_to_pil(self) -> Image:
        if self.channels == 1:
            return Image.fromarray(self.get_array().reshape((self.height, self.width)))
        return Image.fromarray(self.get_array())

    def get_array(self) -> np.ndarray:
        return np.uint8(np.round(self.array))

    def __copy__(self):
        cpy = copy.deepcopy(self.array)
        a_copy = ImageImpl(cpy)
        return a_copy

    def __len__(self):
        return len(self.array)

    def df(self):
        # to make a histogram (count distribution frequency)
        hists = []
        array = self.get_array()
        for c in range(self.channels):
            hists.append(ImageImpl._df_image(array[:, :, c], self.width, self.height))
        return hists

    def cdf(self):
        # cumulative distribution frequency
        cdf = []
        hists = self.df()
        for c in range(self.channels):
            cdf.append(ImageImpl._cdf_hist(hists[c]))
        return cdf

    def equalize_image(self):
        # Returns the equalized image in a new ImageImpl instance.
        image_cdf = self.cdf()
        array = self.get_array()
        image_equalized = np.zeros(array.shape)

        for c in range(self.channels):
            image_equalized[:, :, c] = ImageImpl._equalize_cdf(
                array[:, :, c], image_cdf[c]
            )
        return ImageImpl.from_array(image_equalized)

    def to_gray(self):
        if self.channels == 3:
            return ImageImpl.from_array(np.dot(self.array[..., :3], [0.2989, 0.5870, 0.1140])[:, :, np.newaxis])
        return self

    def to_rgb(self):
        if self.channels == 1:
            return ImageImpl.from_array(np.repeat(self.array, 3, axis=2))
        return self

    @staticmethod
    def from_array(array):
        return ImageImpl(array)

    @staticmethod
    def _df_image(img: np.ndarray, width: int, height: int):
        # count distribution frequency for an array
        hist = [0] * constants.PIXEL_RANGE
        for i in range(width):
            for j in range(height):
                hist[img[i, j]] += 1

        return hist

    @staticmethod
    def _cdf_hist(hist):
        # cumulative distribution frequency
        cdf = [0] * constants.PIXEL_RANGE
        cdf[0] = hist[0]
        for i in range(1, constants.PIXEL_RANGE):
            cdf[i] = cdf[i - 1] + hist[i]
        # Now we normalize the histogram
        cdf = [ele * constants.MAX_PIXEL_VALUE / cdf[-1] for ele in cdf]
        return cdf

    @staticmethod
    def _equalize_cdf(img: np.ndarray, image_cdf) -> np.ndarray:
        # use linear interpolation of cdf to find new pixel values. Scipy alternative exists
        return np.interp(img, range(0, constants.PIXEL_RANGE), image_cdf)
