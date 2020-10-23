# dummy file for now
# the import problem was solved formm:
# https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named-yadayadayada
import os
import sys

test_module_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, test_module_path + "/../")

import numpy as np

from pyimg.config.constants import MAX_PIXEL_VALUE
from pyimg.models.image.operators import linear_adjustment

IMG_SHAPE = (256, 256, 3)


def given_an_identity_matrix() -> np.ndarray:
    """
    Returns a matrix of shape IMG_SHAPE with values 1 above max pixel value
    :return: np.ndarray
    """
    return np.ones(IMG_SHAPE).dot(MAX_PIXEL_VALUE + 1)


def test_linear_adjustment_upper_limit() -> bool:
    """
    Given a matrix image representation with all values 1 above max pixel value
    check if the ratio of diference is consistent
    :return: bool
    """

    a_img = given_an_identity_matrix()
    adjusted = linear_adjustment(a_img)
    # check if values are all equals

    assert then_all_values_equal(adjusted)
    assert then_pixel_ratio_consistent(a_img, adjusted)
    return True


def then_all_values_equal(a_img: np.ndarray) -> bool:
    """
    Given an image check if all pixels are equals
    :param a_img: np.ndarray
    :return: true if all values are equal
    """

    return a_img.min() == a_img.max()


def then_pixel_ratio_consistent(a_img: np.ndarray, adjusted: np.ndarray) -> bool:
    """
    Given an image and it's transformation, check if the ratio between all pixels is the same.
    AKA linear.
    :param a_img: Original image matrix representation
    :param adjusted: transformed imae matrix representation
    :return: True if constant pixel ratio is applied.
    """

    dif_matrix = a_img - adjusted
    return then_all_values_equal(dif_matrix)
