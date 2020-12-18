from typing import Union
import matplotlib.pyplot as plt
import cv2
import numpy as np

from pyimg.models.image import ImageImpl, object_recognition
from pyimg.models.image.noise import apply_noise
from pyimg.models.random_number.generator import GaussianGenerator

def get_result(
    image: ImageImpl,
    acepted: bool,
    descriptors1_qty: int,
    descriptors2_qty: int,
    matches_qty: int,
    matches_mean: float,
    matches_std: float,
):
    return matches_qty / min(descriptors1_qty, descriptors2_qty)


def adjust_gamma(image: ImageImpl, gamma: Union[float, int]) -> ImageImpl:
    invGamma = 1.0 / gamma
    table = np.array([
      ((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)])
    return ImageImpl.from_array(cv2.LUT(image.array.astype(np.uint8), table.astype(np.uint8)))


def scale_transform(image: ImageImpl, scale: Union[float, int]) -> ImageImpl:
    """ Resize an image maintaining its proportions

    Args:
        image: image
        scale (Union[float, int]): Percent as whole number of original image. eg. 53

    Returns:
        image (np.ndarray): Scaled image
    """
    _scale = lambda dim, s: int(dim * s)
    im = image.array
    width, height, channels = im.shape
    new_width: int = _scale(width, scale)
    new_height: int = _scale(height, scale)
    new_dim: tuple = (new_width, new_height)
    return ImageImpl.from_array(cv2.resize(src=im, dsize=new_dim, interpolation=cv2.INTER_LINEAR))


def rotate_transform(image: ImageImpl, angle: Union[float, int]) -> ImageImpl:
    image_center = tuple(np.array(image.array.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image.array, rot_mat, image.array.shape[1::-1], flags=cv2.INTER_LINEAR)
    return ImageImpl.from_array(result)


def gaussian_transform(image: ImageImpl, threshold: Union[float, int]) -> ImageImpl:
    return apply_noise(image, GaussianGenerator(mean=100, std=20, size=image.array.size), True, threshold)


def automated_result(
    image1: ImageImpl,
    image2: ImageImpl,
    threshold: float,
    transform_type: str,
    validate_second_min: bool,
    validate_second_threshold: float,
    acceptance: float = 0.2,
):
    similarities = [1, 2, -1, 'cos']
    similarities_names = ['Manhattan', 'Euclidean', 'Chebyshev', 'Cosine']
    similarities_color = ['r', 'g', 'b', 'y']

    illumination_range = [x * 0.2 for x in range(1, 12)]
    scaling_range = [x * 0.1 for x in range(5, 15)]
    rotation_range = range(0, 330, 30)
    gaussian_range = [x * 0.1 for x in range(1, 9)]

    fig = plt.figure()
    plt.ylabel("Percentage of key-point matched")

    if transform_type == 'i':
        transform_range = illumination_range
        transform_func = adjust_gamma
        plt.xlabel("Illumination variation")
    elif transform_type == 's':
        transform_range = scaling_range
        transform_func = scale_transform
        plt.xlabel("Percentage of scaling")
    elif transform_type == 'r':
        transform_range = rotation_range
        transform_func = rotate_transform
        plt.xlabel("Angle of rotation")
    elif transform_type == 'n':
        transform_range = gaussian_range
        transform_func = gaussian_transform
        plt.xlabel("Sigma for Gaussian Additive noise")
    else:
        return

    percentages_lists = []
    plt.grid(True)
    ax = fig.gca()
    for similarity in similarities:
        percentages_lists.append([])
        for transform in transform_range:
            image_temp = transform_func(image2, transform)
            percentages_lists[-1].append(get_result(*object_recognition.compare_images_sift(
                image1, image_temp, threshold, acceptance, similarity, validate_second_min,
                validate_second_threshold)))

    for percentages, name, color, in zip(percentages_lists, similarities_names, similarities_color):
        plt.plot(transform_range, percentages, 'o-', color=color, label=name)


    ax.set_yticks(np.arange(0, 1., 0.1))
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.legend()
    plt.show()

