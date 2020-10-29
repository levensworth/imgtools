from copy import copy
from typing import List

import numpy as np

from pyimg.config import constants
from pyimg.config.constants import MAX_PIXEL_VALUE, MIN_PIXEL_VALUE
from pyimg.models.image import ImageImpl, operators, linear_adjustment
from pyimg.models.image.thresholding import umbralization_with_two_thresholds

from .filters import gaussian_filter_fast, bilateral_filter, circular_kernel


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
    a_img: ImageImpl, kernel_size: int, sigma_s: float, sigma_r: float, four_neighbours: bool,
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
    horizontal_image = linear_adjustment(border_images[1])
    vertical_image = linear_adjustment(border_images[2])

    angle_matrix = get_angle(horizontal_image, vertical_image)
    suppressed_image = suppress_false_maximums(synthesized_image, angle_matrix)

    low_threshold = np.amax(suppressed_image.array) * 0.06
    high_threshold = np.amax(suppressed_image.array) * 0.14

    import cv2
    edges = cv2.Canny(filtered_image.get_array()[..., 0], 100, 200)

    umbralized_image = umbralization_with_two_thresholds(suppressed_image, high_threshold, low_threshold)

    border_image = hysteresis(umbralized_image, bool(four_neighbours))

    border_image = ImageImpl.from_array(edges[:, :, np.newaxis])
    return border_image


def get_angle(horizontal_image: ImageImpl, vertical_image: ImageImpl) -> np.ndarray:
    img1, img2 = horizontal_image.array[..., 0], vertical_image.array[..., 0]
    u1 = 22.5
    u2 = 67.5
    u3 = 112.5
    u4 = 157.5
    img3 = np.arctan2(img1, img2) * 180 / np.pi
    for i in range(0, img1.shape[0]):
        for j in range(0, img1.shape[1]):
            if img2[i, j] == 0:
                img3[i, j] = 90
            else:
                val = img3[i, j]
                if val < 0:
                    val = val + 180
                if u1 <= val < u2:
                    img3[i, j] = 45
                elif u2 <= val < u3:
                    img3[i, j] = 90
                elif u3 <= val < u4:
                    img3[i, j] = 135
                else:
                    img3[i, j] = 0
    return img3


def suppress_false_maximums(synthesized_image: ImageImpl, angle_matrix: np.ndarray) -> ImageImpl:
    synthesized_array = synthesized_image.array[..., 0]

    result = np.array(synthesized_array)
    for i in range(1, result.shape[0] - 1):
        for j in range(1, result.shape[1] - 1):
            if angle_matrix[i, j] == 0:
                pix1 = synthesized_array[i, j - 1]
                pix2 = synthesized_array[i, j + 1]
            elif angle_matrix[i, j] == 45:
                pix1 = synthesized_array[i - 1, j + 1]
                pix2 = synthesized_array[i + 1, j - 1]
            elif angle_matrix[i, j] == 135:
                pix1 = synthesized_array[i - 1, j - 1]
                pix2 = synthesized_array[i + 1, j + 1]
            elif angle_matrix[i, j] == 90:
                pix1 = synthesized_array[i - 1, j]
                pix2 = synthesized_array[i + 1, j]
            else:
                ValueError("Invalid angle in matrix")

            if pix1 > synthesized_array[i, j] or pix2 > synthesized_array[i, j]:
                result[i, j] = 0
    return ImageImpl.from_array(result[:, :, np.newaxis])


def has_border_neighbours_without_thresholds(image: ImageImpl, x: int, y: int, four_neighbours: bool):
    height, width = image.height, image.width
    image_array = image.get_array()[..., 0]

    if four_neighbours:
        increments = [[0, -1], [1, 0], [0, 1], [-1, 0]]
    else:
        increments = [[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]]
    for i in range(0, len(increments)):
        new_x = x + increments[i][1]
        new_y = y + increments[i][0]
        if 0 <= new_x < width and 0 <= new_y < height and image_array[new_y, new_x] == constants.MAX_PIXEL_VALUE:
            return True
    return False


def hysteresis(image: ImageImpl, four_neighbours: bool = True) -> ImageImpl:
    height, width = image.height, image.width
    image_array = image.get_array()[..., 0]

    border_image = np.zeros((height, width))
    for y in range(0, height):
        for x in range(0, width):
            if image_array[y, x] == constants.MAX_PIXEL_VALUE:
                border_image[y, x] = constants.MAX_PIXEL_VALUE
            elif image_array[y, x] == constants.MAX_PIXEL_VALUE / 2 and \
                has_border_neighbours_without_thresholds(image, x, y, four_neighbours):
                border_image[y, x] = constants.MAX_PIXEL_VALUE

    return ImageImpl(border_image[:, :, np.newaxis])


def susan_detection(
    a_img: ImageImpl, threshold: int
) -> ImageImpl:
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
