import numpy as np

from pyimg.models.matrix import Matrix


def given_a_2d_matrix():
    return Matrix(np.ones((10, 10)))


def given_a_3d_matrix():
    return Matrix(np.ones((10, 10, 3)))


def given_a_filter():
    return lambda window: np.sum(np.sum(window))


def given_a_kernel_size():
    return 3


# This test is deprecated because we
# shifted towards 3 dimensional tensors as defacto standard for all images (B&W as well as RGB)
"""
def test_2d_conv():

    matrix = given_a_2d_matrix()
    fn = given_a_filter()
    kernel_size = given_a_kernel_size()

    matrix.convolution(kernel_size, fn)
    assert then_check_2d_aplication(matrix.array)
"""


def test_3d_conv():

    matrix = given_a_3d_matrix()
    fn = given_a_filter()
    kernel_size = given_a_kernel_size()

    matrix.convolution(kernel_size, fn)
    assert then_check_3d_aplication(matrix)


def then_check_2d_aplication(transformed: Matrix) -> bool:
    """
    Checks whether the transformed 2d matrix does equal transformation
    :param transformed:
    :return:
    """

    real_transformed = np.array(
        [
            [4, 6, 6, 6, 6, 6, 6, 6, 6, 4],
            [6, 9, 9, 9, 9, 9, 9, 9, 9, 6],
            [6, 9, 9, 9, 9, 9, 9, 9, 9, 6],
            [6, 9, 9, 9, 9, 9, 9, 9, 9, 6],
            [6, 9, 9, 9, 9, 9, 9, 9, 9, 6],
            [6, 9, 9, 9, 9, 9, 9, 9, 9, 6],
            [6, 9, 9, 9, 9, 9, 9, 9, 9, 6],
            [6, 9, 9, 9, 9, 9, 9, 9, 9, 6],
            [6, 9, 9, 9, 9, 9, 9, 9, 9, 6],
            [4, 6, 6, 6, 6, 6, 6, 6, 6, 4],
        ]
    )
    return np.equal(transformed, real_transformed).astype(int).min() == 1


def then_check_3d_aplication(transformed: Matrix) -> bool:

    for i in range(3):
        if not then_check_2d_aplication(transformed.array[:, :, i]):
            return False

    return True
