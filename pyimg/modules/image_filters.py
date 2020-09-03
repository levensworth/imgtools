import numpy as np

from pyimg.config.constants import MAX_PIXEL_VALUE


def threshold_filter(a_img: np.ndarray, threshold: float) -> np.ndarray:
    """
    Given a matrix image representation, apply treshold transformation
    and return a binary matrix which follows this transformation:
    F(img[i,j]):| max_pixel_value if img[i, j] >= threshold
                | 0 else
    :param a_img: matrix image representation
    :param threshold: float between [0, max pixel value]
    :return: transformed matrix
    """
    # applies if element wise
    a_img = a_img >= threshold
    # transform a boolean matrix binary and scales
    return a_img.astype(int) * MAX_PIXEL_VALUE
