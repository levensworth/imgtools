import math

import numpy as np
from PIL import Image

from pyimg.models import Matrix


class ImageImpl(Matrix):
    def __init__(self, image):
        super(ImageImpl, self).__init__(image)
        self.wight = self.array.shape[0]
        self.height = self.array.shape[1]
        self.channels = self.array.shape[2]

    def convert_to_pil(self) -> Image:
        return Image.fromarray(self.get_array())

    def get_histogram_dict(self):
        hists = []
        for i in range(self.channels):
            unique, counts = np.unique(self.get_array()[i], return_counts=True)
            hists.append(dict(zip(unique, counts)))
        return hists

    def get_array(self) -> np.ndarray:
        return np.uint8(np.round(self.array))

    @staticmethod
    def from_array(array):
        return ImageImpl(array)
