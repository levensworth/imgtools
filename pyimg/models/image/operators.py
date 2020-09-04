import math

import numpy as np

from pyimg.config.constants import MAX_PIXEL_VALUE, MIN_PIXEL_VALUE
from pyimg.models.image import ImageImpl


def gamma_fun(a_img: ImageImpl, gamma: float) -> ImageImpl:
    """
    Given a matrix image representation, apply gamma filter
    :param a_img: image matrix representation
    :param gamma: gamma value, should be positive value
    :return: image matrix after filter transformation
    """
    # remember T(img) = C^(1 - gamma) * img ^ (gamma)
    # calculate C value
    c = MAX_PIXEL_VALUE
    # element wise power function
    img = a_img.apply(lambda x: np.power(x, gamma))
    return img.apply(lambda x: x.dot(c ** (1 - gamma)))


def negative_img_fun(a_img: ImageImpl) -> ImageImpl:
    """
    Given an image matrix representation, invert pixel values.
    Following the function:
    F: PixelDomain -> PixelDomain/
    F(r) = -r + Max_Pixel_value
    :param a_img: matrix image representation
    :return: transformed matrix
    """

    return a_img.mul_scalar(-1).add_scalar(MAX_PIXEL_VALUE)


def dynamic_compression_image(a_img):
    c = (MAX_PIXEL_VALUE - 1) / math.log10(1 + a_img.max_value())
    return a_img.add_scalar(1).apply(np.log10).mul_scalar(c)


def histogram_equalization(a_img: ImageImpl) -> ImageImpl:
    """
    Given a matrix representation of an image, apply histogram equalization as given by:

    T(Rk) = sum from 0 to k of Nj/N
    where:
        - Rk: k-th grey value in the scale from o - max pixel value
        - Nj: number of pixel with j-th grey value in the matrix
        - N: total number of pixels.
    :param a_img: image matrix representation
    :return: transformed matrix
    """

    if a_img.channels == 3:
        for i in range(a_img.channels):
            a_img.array[:, :, i] = _equalize_single_scale(a_img.array[:, :, i])
    elif a_img.channels == 2:
        a_img = _equalize_single_scale(a_img.array)
    else:
        raise ArithmeticError(
            "matrix with {} channels can't be processed".format(a_img.channels)
        )
    return a_img


def _equalize_single_scale(a_matrix: np.ndarray) -> np.ndarray:
    """
    Given a matrix representation of pixel values with shape (n,m)
    apply histogram equalization
    :param a_matrix: pixel matrix of shape (n, m)
    :return: transformed matrix
    """

    local_max_pixel_value = np.amax(a_matrix)
    local_min_pixel_value = np.amin(a_matrix)

    total_pixels = np.count_nonzero(a_matrix > 0)
    # for each value in the pixel scale equalize matrix
    for grey_value in range(MAX_PIXEL_VALUE):
        # TODO: this is too slow!
        new_grey_value = 0
        for i in range(grey_value):
            # to calculate new grey iterate over all prev values
            indices = np.argwhere(a_matrix == i)
            new_grey_value += len(indices) / total_pixels

        # new_grey is a value in range [0, 1) , we transform it to [0, max pixel val]
        new_grey_value = int(
            new_grey_value * (local_max_pixel_value - local_min_pixel_value)
            + local_min_pixel_value
        )

        # get indeces of all pixels with specified grey value
        # https://stackoverflow.com/questions/4588628/find-indices-of-elements-equal-to-zero-in-a-numpy-array
        indices = np.argwhere(a_matrix == grey_value)

        # transform those values with new grey val
        for y, x in indices:
            a_matrix[y, x] = new_grey_value

    return a_matrix


def linear_adjustment(a_img: ImageImpl) -> ImageImpl:
    """
    Given a matrix image representation apply, if necessary, linear transformation
    to bring values in the pixel value range (0, 255).
    :param a_img: numpy array of 2 or 3 dimensions.
    :return: np.ndarray of same shape with values in range
    """

    min_value = a_img.min_value()
    max_value = a_img.max_value()

    if MAX_PIXEL_VALUE >= max_value and MIN_PIXEL_VALUE <= min_value:
        # values are in range
        return a_img  # pixels should be ints no floats
    # if values are out of range, adjust based on current values

    if max_value == min_value:
        # a transformation should only shift values in this case
        slope = 0
    else:
        slope = (MAX_PIXEL_VALUE - MIN_PIXEL_VALUE) / (max_value - min_value)
    if max_value == min_value:
        if max_value > MAX_PIXEL_VALUE:
            constant = MAX_PIXEL_VALUE
        elif min_value < MIN_PIXEL_VALUE:
            constant = MIN_PIXEL_VALUE
        else:
            constant = max_value
    else:
        # as we want the tranformation to map MIN_PIXEL_VALUE to the min_value found
        # we just solve y = mx + b for known x, y and m
        constant = MIN_PIXEL_VALUE - slope * min_value

    return a_img.mul_scalar(slope).add_scalar(constant)
