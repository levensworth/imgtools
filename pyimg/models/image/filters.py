import math

import numpy as np
import scipy.stats as st

from pyimg.config.constants import MAX_PIXEL_VALUE, MIN_PIXEL_VALUE
from pyimg.models.image import ImageImpl


def mean_filter(a_img: ImageImpl, kernel_size: int) -> ImageImpl:
    """
    Given an Image instance, apply the mean filter using a square kernel of size
    kernel_size.
    :param a_img: Image instance
    :param kernel_size: kernel size int
    :return: transformed image
    """
    a_img.convolution(
        kernel_size,
        lambda window: np.mean(
            window.reshape(
                -1,
            )
        ),
    )

    return a_img


def mean_filter_fast(a_img: ImageImpl, kernel_size: int) -> ImageImpl:
    """
    Given an Image instance, apply the mean filter using a square kernel of size
    kernel_size.
    :param a_img: Image instance
    :param kernel_size: kernel size int
    :return: transformed image
    """
    kernel_size = int(kernel_size)
    a_img.convolution_fast(
        np.ones((kernel_size, kernel_size, a_img.channels)),
    )

    return a_img


def median_filter(a_img: ImageImpl, kernel_size: int) -> ImageImpl:
    """
    Given an Image instance, apply the median filter using a square kernel of size
    kernel_size.
    :param a_img: Image instance
    :param kernel_size: kernel size int
    :return: transformed image
    """
    a_img.convolution(
        kernel_size,
        lambda window: np.median(
            window.reshape(
                -1,
            )
        ),
    )

    return a_img


def median_filter(a_img: ImageImpl, kernel_size: int) -> ImageImpl:
    """
    Given an Image instance, apply the median filter using a square kernel of size
    kernel_size.
    :param a_img: Image instance
    :param kernel_size: kernel size int
    :return: transformed image
    """
    a_img.convolution(
        kernel_size,
        lambda window: np.median(
            window.reshape(
                -1,
            )
        ),
    )

    return a_img


def weighted_median_filter(a_img: ImageImpl, kernel_size: int) -> ImageImpl:
    """
    Given an Image instance, apply the weighted median filter using a square kernel of size
    kernel_size.
    :param a_img: Image instance
    :param kernel_size: kernel size int
    :return: transformed image
    """
    a_img.convolution(
        kernel_size,
        lambda window: _weighted_median(window, kernel_size),
    )

    return a_img


def _weighted_median(array: np.ndarray, kernel_size: int):

    radius_matrix = np.zeros((int(kernel_size), int(kernel_size)))
    height = array.shape[0]
    width = array.shape[1]
    centroid = np.array([int(height / 2), int(width / 2)])

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            # calculate radius
            radius_matrix[i, j] = np.linalg.norm(np.array([i, j]) - centroid)

    radius_matrix *= len(set(radius_matrix.reshape((-1,)).tolist()))

    radius_matrix = radius_matrix.astype(int)

    radius_matrix = np.abs(radius_matrix - radius_matrix.max() - 1)

    radius_matrix = radius_matrix.reshape((-1,))
    # reshaped into a single vector
    array = array.reshape((-1,))

    weighted_vector = []

    for i in range(len(array)):

        weighted_vector += [array[i] for _ in range(radius_matrix[i])]

    return np.median(np.array(weighted_vector))


def gaussian_filter(a_img: ImageImpl, kernel_size: int, sigma: float) -> ImageImpl:
    """
    Given an Image instance, apply the weighted gaussian filter using a square kernel of size
    kernel_size.
    :param a_img: Image instance
    :param kernel_size: kernel size int
    :param sigma: sigma value
    :return: transformed image
    """
    a_img.convolution(
        kernel_size,
        lambda window: _apply_gaussian_filter(window, int(kernel_size), sigma),
    )

    return a_img


def _apply_gaussian_filter(window: np.ndarray, kernerl_size: int, sigma: float):

    weights_matrix = gaussian_kernel(kernerl_size, sigma) * MAX_PIXEL_VALUE
    weights_matrix = weights_matrix.astype(int).reshape((-1,))
    return np.average(window.reshape((-1,)), weights=weights_matrix)


def gaussian_kernel(kernlen=5, nsig=1):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


def gaussian_filter_fast(a_img: ImageImpl, kernel_size: int, sigma: float) -> ImageImpl:
    """
    Given an Image instance, apply the weighted gaussian filter using a square kernel of size
    kernel_size.
    :param a_img: Image instance
    :param kernel_size: kernel size int
    :param sigma: sigma value
    :return: transformed image
    """
    kernel_size = int(kernel_size)
    kernel = (
        np.expand_dims(gaussian_kernel(kernel_size, sigma), axis=2) * MAX_PIXEL_VALUE
    )
    a_img.convolution_fast(
        (1 / kernel_size ** 2) * kernel,
    )

    return a_img


def high_filter(a_img: ImageImpl, kernel_size: int) -> ImageImpl:
    """
    Given a Image instance, apply high value filter.
    :param a_img:
    :param kernel_size:
    :return: transformed Image instance
    """

    a_img.convolution(
        kernel_size,
        lambda window: apply_high_filter(window, int(kernel_size)),
    )

    return a_img


def apply_high_filter(a_img: np.ndarray, kernel_size: int):
    kernel = np.ones((kernel_size, kernel_size)) * -1
    # augment the center value
    kernel[int(kernel_size / 2), int(kernel_size / 2)] = kernel_size
    # element wise multiplication
    after_kernel_weights = a_img * kernel

    return np.mean(after_kernel_weights)
