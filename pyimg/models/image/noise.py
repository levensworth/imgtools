import numpy as np

from pyimg.config import constants
from pyimg.models.image import ImageImpl
from pyimg.models.random_number.generator import Generator


def apply_noise(image: ImageImpl, random_generator: Generator,
                is_additive: bool, threshold: float) -> ImageImpl:

    noise = random_generator.generate()
    noise = np.array(noise).reshape(image.array.shape)

    mask = np.random.uniform(low=0.0, high=1.0, size=image.array.shape)
    mask = np.where(mask > threshold, 1.0, 0.0)

    if is_additive:
        result = image.array + mask * noise
    else:
        result = image.array * mask * noise

    return ImageImpl.from_array(result)


def salt_and_pepper_apply(image: ImageImpl, p0: float, p1=None) -> ImageImpl:
    if p1 is None:
        p1 = 1 - p0

    mask = np.random.uniform(low=0.0, high=1.0, size=image.array.shape)
    mask_p0 = np.where(mask > p0, 1.0, 0.0)
    mask_p1 = np.where(mask < p1, 1.0, 0.0)
    mask_p1_add = np.where(mask > p1, 1.0, 0.0)

    result = image.array * mask_p0
    result = result * mask_p1 + mask_p1_add * constants.MAX_PIXEL_VALUE

    return ImageImpl.from_array(result)
