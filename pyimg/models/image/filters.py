import math

import numpy as np
import scipy.stats as st
from scipy import ndimage

from pyimg.config.constants import MAX_PIXEL_VALUE
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
        np.ones((kernel_size, kernel_size)),
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


def median_filter_fast(a_img: ImageImpl, kernel_size: int) -> ImageImpl:
    """
    Given an Image instance, apply the median filter using a square kernel of size
    kernel_size.
    :param a_img: Image instance
    :param kernel_size: kernel size int
    :return: transformed image
    """
    # build kernel
    kernel_size = int(kernel_size)

    filtered = ndimage.median_filter(a_img.array, size=kernel_size)
    a_img.array = filtered
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
    kernel = gaussian_kernel(kernel_size, sigma) * MAX_PIXEL_VALUE
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


def high_filter_fast(a_img: ImageImpl, kernel_size: int) -> ImageImpl:
    """
    Given a Image instance, apply high value filter.
    :param a_img:
    :param kernel_size:
    :return: transformed Image instance
    """
    kernel_size = int(kernel_size)
    kernel = np.ones((kernel_size, kernel_size)) * -1
    # augment the center value
    kernel[int(kernel_size / 2), int(kernel_size / 2)] = kernel_size

    a_img.convolution_fast(
        kernel,
    )
    return a_img


def threshold_filter(a_img: ImageImpl, threshold: float) -> ImageImpl:
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
    a_img.array = a_img.array >= threshold
    # transform a boolean matrix binary and scales
    a_img.array = a_img.array.astype(int) * MAX_PIXEL_VALUE

    return a_img


def bilateral_filter(a_img: ImageImpl, kernel_size: int, sigma_s: float, sigma_r: float) -> ImageImpl:
    """
    Given an Image instance, apply the bilateral filter
    kernel_size.
    :param a_img: Image instance
    :param kernel_size: kernel size int
    :param sigma_s: sigma_s value
    :param sigma_r: sigma_r value
    :return: transformed image
    """
    a_img.convolution(
        kernel_size,
        lambda window: _apply_bilateral_filter(window, int(kernel_size), sigma_s, sigma_r),
    )

    return a_img


def _apply_bilateral_filter(window: np.ndarray, kernerl_size: int, sigma_s: float, sigma_r: float):
    sliding_window = np.zeros((kernerl_size, kernerl_size))
    wp = 0
    for i in range(0, kernerl_size):
        for j in range(0, kernerl_size):
            y_center = int(kernerl_size / 2)
            x_center = int(kernerl_size / 2)
            k = i - y_center
            l = j - x_center
            gaussian_value = -(pow(y_center - k, 2) + pow(x_center - l, 2)) / (2 * sigma_s * sigma_s)
            r_value = -(pow(abs(int(window[y_center, x_center]) - int(window[i, j])), 2) /
                        (2 * sigma_r * sigma_r))
            sliding_window[i, j] = math.exp(gaussian_value + r_value)
            wp += sliding_window[i, j]
    sliding_window = sliding_window / wp
    return np.sum(window * sliding_window)
