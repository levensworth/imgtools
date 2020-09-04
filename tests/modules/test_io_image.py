import pathlib

import numpy as np
from PIL import Image

from pyimg.modules.image_io import load_unformatted_raw_image

TEST_IMG = pathlib.PurePosixPath("/Users/levensworth/pyimg-tools/generated/elena.RAW")
CONTROL_IMG = pathlib.PurePosixPath(
    "/Users/levensworth/pyimg-tools/generated/elena.jpeg"
)


def given_raw_file_path():
    return TEST_IMG


def test_load_unformatted_raw() -> bool:
    """
    test loading raw images with .info metadata files
    :return: bool
    """

    img_path = given_raw_file_path()
    img = load_unformatted_raw_image(img_path)
    assert then_equal_elena(img)


def then_equal_elena(a_img: np.ndarray) -> bool:
    """
    Test whether or not a matrix image representation
    is equal to the B&W elena image
    :param a_img:
    :return:
    """
    control = Image.open(str(CONTROL_IMG))
    control_matrix = np.asarray(control)
    return np.allclose(control_matrix, a_img, rtol=2, atol=10)


test_load_unformatted_raw()
