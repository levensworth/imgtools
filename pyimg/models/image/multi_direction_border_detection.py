from copy import copy

from scipy.ndimage.interpolation import rotate

from pyimg.models.image import ImageImpl

from .kernels import *

ROTATIONS = [0, 45, 90]


def itba_border_detection(a_img: ImageImpl) -> ImageImpl:
    return rotate_border_detection(a_img, ITBA, [0, 90])


def prewitt_border_detection(a_img: ImageImpl) -> ImageImpl:
    return rotate_border_detection(a_img, PREWITT, ROTATIONS)


def sobel_border_detection(a_img: ImageImpl) -> ImageImpl:
    return rotate_border_detection(a_img, SOBEL, [0, 90])


def kirish_border_detection(a_img: ImageImpl) -> ImageImpl:
    return rotate_border_detection(a_img, KIRISH, ROTATIONS)


def rotate_border_detection(a_img: ImageImpl, kernel, rotations) -> ImageImpl:
    mask_images = []
    for rotation in rotations:
        angle = rotation
        kernel = rotate(kernel, angle, reshape=False, mode="reflect")
        new_img = copy(a_img)
        new_img.convolution_fast(kernel)
        mask_images.append(new_img.array)

    result_array = np.zeros(a_img.array.shape)
    for img in mask_images:
        result_array += img ** 2

    result_array = np.sqrt(result_array)

    return ImageImpl(result_array)
