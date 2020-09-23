from pyimg.config.constants import MAX_PIXEL_VALUE, MIN_PIXEL_VALUE
from pyimg.models.image import ImageImpl
import numpy as np
from copy import copy

def prewitt_detector(a_img: ImageImpl) -> ImageImpl:
    '''
    Detect borders within an image based on prewitt algorithm.
    Basically, we look for gradient variations using finite difference analysis.
    :param a_img: This is the img over which we want to obtain results.

    :return ImageImpl the boolean image representation where border pixels have a MAX_PIXEL value, all else is ZERO.
    '''
    # this will calculate the horizontal gradient
    horizontal_kernel = np.array([[ 1,  1,  1],
                                  [ 0,  0,  0],
                                  [-1, -1, -1]])
    # calcualte vertical gradient
    vertical_kernel = np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]])
    horizontal_grad = copy(a_img)
    horizontal_grad.convolution_fast(horizontal_kernel)

    vertical_grad = copy(a_img)
    vertical_grad.convolution_fast(vertical_kernel)


    # finally the module of the value is calculated as the sqrt([dFx ** 2 + dFy ** 2])
    grad_img = ImageImpl(np.sqrt(horizontal_grad.array ** 2 + vertical_grad.array ** 2))
    return grad_img



