import numpy as np
from PIL import Image

from pyimg.models import Matrix


class ImageImpl(Matrix):
    def __init__(self, image: Image):
        super(ImageImpl, self).__init__(image)
        self.w = self.array.shape[0]
        self.h = self.array.shape[1]
        self.c = self.array.shape[2]

    def convert_to_pil(self) -> Image:
        return Image.fromarray(self.get_array())

    def get_histogram_dict(self):
        unique, counts = np.unique(self.get_array(), return_counts=True)
        return dict(zip(unique, counts))

    def get_array(self):
        return np.uint8(np.round(self.array))
