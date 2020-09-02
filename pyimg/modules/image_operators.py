import numpy as np

from pyimg.config.constants import MAX_PIXEL_VALUE, MIN_PIXEL_VALUE


def img_sum(a_img: np.ndarray, another_img: np.ndarray) -> np.ndarray:
    """
    Given 2 images apply matrix sum. Both matrices should be of same size
    :param a_img: ndarray of shape (..)
    :param another_img: ndarray of same shape as first param
    :return: np.ndarray of same shape
    """

    return a_img + another_img


def img_mul(a_img: np.ndarray, another_img: np.ndarray) -> np.ndarray:
    """
    Given 2 image matrices apply matrix multiplication (dot product).
    This method expects 2d arrays.
    Future implementations may support higher dimensions.
    :param a_img: ndarray of shape (n, m)
    :param another_img: ndarray of shape (m, s)
    :return: ndarray of shape (n, s)
    """

    return np.matmul(a_img, another_img)


def img_dif(a_img: np.ndarray, another_img: np.ndarray) -> np.ndarray:
    """
    Given 2 images apply matrix dif. Both matrices should be of same size
    :param a_img: ndarray of shape (..)
    :param another_img: ndarray of same shape as first param
    :return: np.ndarray of same shape
    """

    return img_sum(a_img, -1 * another_img)


def order_img_by_size(a_img: np.ndarray, another_img: np.ndarray) -> [np.ndarray]:
    """
    Given 2 arrays, return them in order of size. If none of them fit inside the other,
    raise ArithmeticError
    :param a_img: ndarray of shape (..)
    :param another_img: ndarray of  shape (..)
    :return: list of matrices order by size

    :raise ArithmeticError
    """
    shape_1 = a_img.shape
    shape_2 = another_img.shape

    # this problem reduces to finding the vector whose values are equal or greater in each index

    a_img_is_greater = True
    another_img_is_greater = True
    for x1, x2 in zip(shape_1, shape_2):
        if x1 < x2:
            a_img_is_greater = False
        # notice we don't use else cause of the x1 == x2 clause
        if x1 > x2:
            another_img_is_greater = False

    # check if indeed it was greater
    if a_img_is_greater:
        return [a_img, another_img]
    elif another_img_is_greater:
        return [another_img, a_img]
    else:
        raise ArithmeticError("Images can't fit together")


def linear_adjustment(a_img: np.ndarray) -> np.ndarray:
    """
    Given a matrix image representation apply, if necessary, linear transformation
    to bring values in the pixel value range (0, 255).
    :param a_img: numpy array of 2 or 3 dimensions.
    :return: np.ndarray of same shape with values in range
    """

    min_value = a_img.min()
    max_value = a_img.max()

    if MAX_PIXEL_VALUE >= max_value and MIN_PIXEL_VALUE <= min_value:
        # values are in range
        return a_img
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

    a_img = a_img * slope + constant
    return a_img.astype(int)  # pixels should be ints no floats
