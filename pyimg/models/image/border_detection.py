from copy import copy
from typing import List

import numpy as np

from pyimg.config import constants
from pyimg.config.constants import MAX_PIXEL_VALUE, MIN_PIXEL_VALUE
from pyimg.models.image import ImageImpl, linear_adjustment, operators
from pyimg.models.image.thresholding import umbralization_with_two_thresholds

from .filters import bilateral_filter, circular_kernel, gaussian_filter_fast


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


def sobel_detector(a_img: ImageImpl, return_all: bool = False) -> List[ImageImpl]:
    """
    Given an image object, return the gradient filter (border detection) as
    denoted by the sobel operator.
    :param return_all:
    :param a_img: image matrix representation
    :return: image gradient filter matrix representation
    """

    horizontal_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # calcualte vertical gradient
    vertical_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    horizontal_grad = copy(a_img)
    horizontal_grad.convolution_fast(horizontal_kernel)

    vertical_grad = copy(a_img)
    vertical_grad.convolution_fast(vertical_kernel)

    # finally the module of the value is calculated as the sqrt([dFx ** 2 + dFy ** 2])
    grad_img = ImageImpl(np.sqrt(horizontal_grad.array ** 2 + vertical_grad.array ** 2))

    result = [grad_img]
    if return_all:
        result.append(horizontal_grad)
        result.append(vertical_grad)

    return result


def laplacian_border_detection(a_img: ImageImpl) -> ImageImpl:
    """
    Detec borders based on the laplacian algorithm
    :param a_img: matrix image representation
    :return: image border mask
    """

    laplacian_operator = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # compute mask
    grad_img = copy(a_img)
    grad_img.convolution_fast(laplacian_operator)

    return grad_img


def gaussian_laplacian_detection(
    a_img: ImageImpl, threshold: int, sigma: float, kernel_size: int
) -> ImageImpl:
    """
    apply gaussian laplacian mask and border detection as explained by (Marr-Hildreth)
    :param kernel_size:
    :param a_img:
    :param threshold:
    :param sigma:
    :return:
    """

    grad_img = gaussian_filter_fast(a_img, kernel_size, sigma)

    img = laplacian_border_detection(grad_img)

    img.apply_laplacian_change(threshold, MAX_PIXEL_VALUE)

    return img


def canny_detection(
    a_img: ImageImpl,
    kernel_size: int,
    sigma_s: float,
    sigma_r: float,
    four_neighbours: bool = True,
) -> ImageImpl:
    """
    apply caddy mask and border detection
    :param four_neighbours:
    :param a_img:
    :param kernel_size:
    :param sigma_s:
    :param sigma_r:
    :return:
    """

    gray_image = a_img.to_gray()

    filtered_image = bilateral_filter(gray_image, kernel_size, sigma_s, sigma_r)

    border_images = sobel_detector(filtered_image, True)

    synthesized_image = linear_adjustment(border_images[0])
    horizontal_image = border_images[1]
    vertical_image = border_images[2]

    # this calcualtes the direction in radians
    angle_matrix = np.arctan2(
        horizontal_image.array[..., 0], vertical_image.array[..., 0]
    )
    # now we convert to degrees
    angle_matrix = np.rad2deg(angle_matrix)
    angle_matrix[angle_matrix < 0] += 180

    suppressed_image = suppress_false_maximums2(synthesized_image, angle_matrix)

    high_threshold = np.amax(suppressed_image.array) * 0.15
    low_threshold = np.amax(suppressed_image.array) * 0.05

    umbralized_image = umbralization_with_two_thresholds(
        suppressed_image, high_threshold, low_threshold
    )

    border_image = hysteresis(umbralized_image, bool(four_neighbours))

    # import cv2
    # edges = cv2.Canny(filtered_image.get_array()[..., 0], 100, 200)
    # border_image = ImageImpl.from_array(edges[:, :, np.newaxis])

    return linear_adjustment(border_image)


def suppress_false_maximums2(
    synthesized_image: ImageImpl, angle: np.ndarray
) -> ImageImpl:
    synthesized_array = synthesized_image.array[..., 0]
    result = np.zeros_like(synthesized_array)

    for i in range(1, result.shape[0] - 1):
        for j in range(1, result.shape[1] - 1):

            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                after_pixel = synthesized_array[i, j + 1]
                before_pixel = synthesized_array[i, j - 1]

            # angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                after_pixel = synthesized_array[i + 1, j - 1]
                before_pixel = synthesized_array[i - 1, j + 1]

            # angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                after_pixel = synthesized_array[i + 1, j]
                before_pixel = synthesized_array[i - 1, j]

            # angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                after_pixel = synthesized_array[i - 1, j - 1]
                before_pixel = synthesized_array[i + 1, j + 1]

            else:
                ValueError("Angle not valid")

            if (
                synthesized_array[i, j] >= after_pixel
                and synthesized_array[i, j] >= before_pixel
            ):
                result[i, j] = synthesized_array[i, j]
            else:
                result[i, j] = 0
    return ImageImpl.from_array(result[:, :, np.newaxis])


def has_border_neighbours_without_thresholds(
    image: ImageImpl, i: int, j: int, four_neighbours: bool
):
    height, width = image.height, image.width
    image_array = image.get_array()[..., 0]

    if four_neighbours:
        increments = [[0, -1], [1, 0], [0, 1], [-1, 0]]
    else:
        increments = [
            [-1, -1],
            [0, -1],
            [1, -1],
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
        ]
    for k in range(0, len(increments)):
        new_i = i + increments[k][0]
        new_j = j + increments[k][1]
        if (
            0 <= new_i < width
            and 0 <= new_j < height
            and image_array[new_i, new_j] == constants.MAX_PIXEL_VALUE
        ):
            return True
    return False


def hysteresis(
    image: ImageImpl,
    four_neighbours: bool = True,
    weak: int = int(constants.MAX_PIXEL_VALUE / 2),
    strong: int = constants.MAX_PIXEL_VALUE,
) -> ImageImpl:
    height, width = image.height, image.width
    image_array = image.get_array()[..., 0]

    border_image = np.array(image_array)

    for i in range(1, width - 1):
        for j in range(1, height - 1):
            if border_image[i, j] == weak:
                if has_border_neighbours_without_thresholds(
                    image, i, j, four_neighbours
                ):
                    border_image[i, j] = strong
                else:
                    border_image[i, j] = 0

    return ImageImpl(border_image[:, :, np.newaxis])


def susan_detection(a_img: ImageImpl, threshold: int) -> ImageImpl:
    """

    :param a_img:
    :param kernel_size:
    :param threshold:
    :return:
    """
    circ_kernel = circular_kernel(7)
    a_img.apply_filter(circ_kernel, lambda m: _calculate_c_for_susan(m, threshold))
    a_img.array = np.uint8(a_img.array > 0.75)
    return a_img.mul_scalar(255)


def _calculate_c_for_susan(matrix, threshold):
    c = 0

    center_val = int(matrix.shape[0] / 2)
    center_val = matrix[center_val, center_val]

    matrix = np.abs(matrix - center_val)
    matrix = matrix < threshold

    try:
        v = sum(np.sum(matrix)) - 12 if center_val < threshold else sum(np.sum(matrix))
        return 1 - v / 37
    except TypeError:
        v = np.sum(matrix) - 12 if center_val < threshold else np.sum(matrix)
        return 1 - v / 37


def harris_detection(a_img: ImageImpl, threshold: int, k: float = 0.04,
                     kernel_size: int = 7, sigma: float = 2) -> ImageImpl:

    gray_image = a_img.to_gray()

    border_images = sobel_detector(gray_image, True)

    horizontal_image = border_images[1]
    vertical_image = border_images[2]

    ix_squared = gaussian_filter_fast(horizontal_image.mul(horizontal_image), kernel_size, sigma)
    iy_squared = gaussian_filter_fast(vertical_image.mul(vertical_image), kernel_size, sigma)

    ixy = gaussian_filter_fast(horizontal_image.mul(vertical_image), kernel_size, sigma)
    ixy_squared = ixy.mul(ixy)

    trace = ix_squared.add(iy_squared)
    det = ix_squared.mul(iy_squared).sub(ixy_squared)
    r = det.sub(trace.mul(trace).mul_scalar(k))
    r = linear_adjustment(r)

    min_value = int(np.max(r.get_array()) * threshold)
    result = a_img.to_rgb()

    for y in range(0, result.height):
        for x in range(0, result.width):
            if r.array[y, x, 0] >= min_value:
                result.array[y, x, 0] = constants.MAX_PIXEL_VALUE
                result.array[y, x, 1] = 0
                result.array[y, x, 2] = 0

    return result
