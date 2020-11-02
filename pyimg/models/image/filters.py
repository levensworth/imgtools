import math
from copy import copy

import numpy as np
import scipy.stats as st
from medpy.filter.smoothing import anisotropic_diffusion as ansio_dif
from scipy import ndimage

from pyimg.config.constants import MAX_PIXEL_VALUE
from pyimg.models.image import ImageImpl

import cv2


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


def circular_kernel(kernel_size: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))


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


def isotropic_diffusion(a_img: ImageImpl, max_scale) -> ImageImpl:
    sigma = 3
    new_img = copy(a_img)
    for scale in range(0, int(max_scale)):
        kernel_size = int(sigma * 2 + 1)
        new_img = new_img.add(gaussian_filter_fast(a_img, kernel_size, sigma))
    return new_img


def leclerc_coef(lam, b):
    return np.exp(-1 * (np.power(lam, 2)) / (np.power(b, 2)))


def anisodiff(a_img: ImageImpl, steps, b) -> ImageImpl:
    """
    Given an image implementation, calculates the ansiotropic filter as described in the paper by Perona- Malik et al.

    :param a_img: image matrix representation
    :param steps: the number of iterations to apply the mask
    :param b: the leclerc coefficient hyper parameter
    :return: ImageImpl
    """

    lam = 0.25
    # convert to int
    steps = int(steps)
    im = a_img.array

    im_new = np.zeros(im.shape, dtype=im.dtype)
    for t in range(steps):
        # calculate the gradient with respect to each cardinal direction
        # the sneaky way is to get get the far right sub matrix (without the last
        # row and subtract the far left one, and the do the same for all other directions.
        dn = im[:-2, 1:-1] - im[1:-1, 1:-1]
        ds = im[2:, 1:-1] - im[1:-1, 1:-1]
        de = im[1:-1, 2:] - im[1:-1, 1:-1]
        dw = im[1:-1, :-2] - im[1:-1, 1:-1]
        im_new[1:-1, 1:-1] = im[1:-1, 1:-1] + lam * (
            leclerc_coef(dn, b) * dn
            + leclerc_coef(ds, b) * ds
            + leclerc_coef(de, b) * de
            + leclerc_coef(dw, b) * dw
        )
        im = im_new
    return ImageImpl(im)


def bilateral_filter(
    a_img: ImageImpl, kernel_size: int, sigma_s: float, sigma_r: float
) -> ImageImpl:
    """
    Given an Image instance, apply the bilateral filter
    kernel_size.
    :param a_img: Image instance
    :param kernel_size: kernel size int
    :param sigma_s: sigma_s value
    :param sigma_r: sigma_r value
    :return: transformed image
    """

    img_in = a_img.get_array()

    if a_img.channels == 1:
        img_in = img_in[:, :, 0]

    gaussian = (
        lambda r2, sigma: (np.exp(-0.5 * r2 / sigma ** 2) * 3).astype(int) * 1.0 / 3.0
    )

    # define the window width to be the 3 time the spatial std. dev. to
    # be sure that most of the spatial kernel is actually captured
    win_width = int(kernel_size)

    # initialize the results and sum of weights to very small values for
    # numerical stability. not strictly necessary but helpful to avoid
    # wild values with pathological choices of parameters
    reg_constant = 1e-8
    wgt_sum = np.ones(img_in.shape) * reg_constant
    result = img_in * reg_constant

    # accumulate the result by circularly shifting the image across the
    # window in the horizontal and vertical directions. within the inner
    # loop, calculate the two weights and accumulate the weight sum and
    # the unnormalized result image
    for shft_x in range(-win_width, win_width + 1):
        for shft_y in range(-win_width, win_width + 1):
            # compute the spatial weight
            w = gaussian(shft_x ** 2 + shft_y ** 2, sigma_s)

            # shift by the offsets
            off = np.roll(img_in, [shft_y, shft_x], axis=[0, 1])

            # compute the value weight
            tw = w * gaussian((off - img_in) ** 2, sigma_r)

            # accumulate the results
            result += off * tw
            wgt_sum += tw

    # normalize the result and return

    out = result / wgt_sum
    dims = len(out.shape)
    out = np.expand_dims(out, axis=dims) if dims == 2 else out

    return ImageImpl.from_array(out)


"""

    a_img.convolution(
        kernel_size,
        lambda window: _apply_bilateral_filter(window, int(kernel_size), sigma_s, sigma_r),
    )

    return a_img
"""


def _apply_bilateral_filter(
    window: np.ndarray, kernel_size: int, sigma_s: float, sigma_r: float
):
    sliding_window = np.zeros((kernel_size, kernel_size))
    wp = 0
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            y_center = int(kernel_size / 2)
            x_center = int(kernel_size / 2)
            k = i - y_center
            l = j - x_center
            gaussian_value = -(pow(y_center - k, 2) + pow(x_center - l, 2)) / (
                2 * sigma_s * sigma_s
            )
            r_value = -(
                pow(abs(int(window[y_center, x_center]) - int(window[i, j])), 2)
                / (2 * sigma_r * sigma_r)
            )
            sliding_window[i, j] = math.exp(gaussian_value + r_value)
            wp += sliding_window[i, j]
    sliding_window = sliding_window / wp
    return np.sum(window * sliding_window)
