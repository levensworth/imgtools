from copy import copy

import numpy as np

from pyimg.config.constants import MAX_PIXEL_VALUE, MIN_PIXEL_VALUE
from pyimg.models.image import ImageImpl

from .filters import gaussian_filter_fast


def prewitt_detector(a_img: ImageImpl) -> ImageImpl:
    """
    Detect borders within an image based on prewitt algorithm.
    Basically, we look for gradient variations using finite difference analysis.
    :param a_img: This is the img over which we want to obtain results.

    :return ImageImpl the boolean image representation where border pixels have a MAX_PIXEL value, all else is ZERO.
    """
    # this will calculate the horizontal gradient
    horizontal_kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    # calcualte vertical gradient
    vertical_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    horizontal_grad = copy(a_img)
    horizontal_grad.convolution_fast(horizontal_kernel)

    vertical_grad = copy(a_img)
    vertical_grad.convolution_fast(vertical_kernel)

    # finally the module of the value is calculated as the sqrt([dFx ** 2 + dFy ** 2])
    array = np.sqrt(horizontal_grad.array ** 2 + vertical_grad.array ** 2)
    grad_img = ImageImpl(array)

    return grad_img


def sobel_detector(a_img: ImageImpl) -> ImageImpl:
    """
    Given an image object, return the gradient filter (border detection) as
    denoted by the sobel operator.
    :param a_img: image matrix representation
    :return: image gradient filter matrix representation
    """

    # this will calculate the horizontal gradient
    horizontal_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # calcualte vertical gradient
    vertical_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    horizontal_grad = copy(a_img)
    horizontal_grad.convolution_fast(horizontal_kernel)

    vertical_grad = copy(a_img)
    vertical_grad.convolution_fast(vertical_kernel)

    # finally the module of the value is calculated as the sqrt([dFx ** 2 + dFy ** 2])
    grad_img = ImageImpl(np.sqrt(horizontal_grad.array ** 2 + vertical_grad.array ** 2))
    return grad_img


def laplacian_border_detection(a_img: ImageImpl, threshold: int) -> ImageImpl:
    """
    Detec borders based on the laplacian algorithm
    :param a_img: matrix image representation
    :return: image border mask
    """

    laplacian_operator = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # compute mask
    grad_img = copy(a_img)
    grad_img.convolution_fast(laplacian_operator)

    # search for gradient changes
    threshold = int(threshold)
    grad_img.apply_laplacian_change(threshold, MAX_PIXEL_VALUE)

    return grad_img


def gaussian_laplacian_detection(
    a_img: ImageImpl, threshold: int, sigma: float, kernel_size: int
) -> ImageImpl:
    """
    apply gaussian laplacian mask and border detection as explained by (Marr-Hildreth)
    :param a_img:
    :param threshold:
    :param sigma:
    :return:
    """

    grad_img = gaussian_filter_fast(a_img, kernel_size, sigma)

    return laplacian_border_detection(grad_img, threshold)
