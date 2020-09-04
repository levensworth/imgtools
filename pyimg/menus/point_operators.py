import os
from tkinter import Menu

from pyimg.config import constants
from pyimg.menus.operation_interface import (BinaryImageOperation,
                                             UnaryImageOperation,
                                             UnaryWithParamsImageOperation)
from pyimg.models.image import ImageImpl, operators
from pyimg.modules.image_io import display_img, save_img


def linear_adj_image_wrapper(image: ImageImpl):
    adjusted_img = operators.linear_adjustment(image)
    img = adjusted_img.convert_to_pil()
    display_img(img)
    save_img(img, os.path.join(constants.SAVE_PATH, "result_img.jpg"))


class PointOperatorMenu:
    """
    This class is a simple wrapper around tkinter menu object.
    it's self display as a menu in the top bar.
    """

    def __init__(self, menubar, image_io):
        operation_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Operations", menu=operation_menu)

        binary_operators_menu = Menu(operation_menu, tearoff=0)
        operation_menu.add_cascade(label="Binary operators", menu=binary_operators_menu)

        binary_operators_menu.add_command(
            label="Add",
            command=BinaryImageOperation(
                image_io,
                "Add",
                lambda x, y: linear_adj_image_wrapper(ImageImpl.add_matrix(x, y)),
            ).generate_interface,
        )

        binary_operators_menu.add_command(
            label="Subtract",
            command=BinaryImageOperation(
                image_io,
                "Sub",
                lambda x, y: linear_adj_image_wrapper(ImageImpl.sub_matrix(x, y)),
            ).generate_interface,
        )

        binary_operators_menu.add_command(
            label="Multiply",
            command=BinaryImageOperation(
                image_io,
                "Mul",
                lambda x, y: linear_adj_image_wrapper(ImageImpl.mul_matrix(x, y)),
            ).generate_interface,
        )

        single_img_menu = Menu(operation_menu, tearoff=0)
        operation_menu.add_cascade(label="Single operators", menu=single_img_menu)

        single_img_menu.add_command(
            label="CRD",
            command=UnaryImageOperation(
                image_io,
                "CRD",
                lambda x: linear_adj_image_wrapper(
                    operators.dynamic_compression_image(x)
                ),
            ).generate_interface,
        )

        single_img_menu.add_command(
            label="Multiply scalar",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Mul",
                lambda m1, scalar: linear_adj_image_wrapper(
                    ImageImpl.mul_scalar_matrix(m1, scalar)
                ),
                params=["scalar"],
            ).generate_interface,
        )
        single_img_menu.add_command(
            label="Gamma",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Gamma",
                lambda m1, c: linear_adj_image_wrapper(operators.gamma_fun(m1, c)),
                params=["c"],
            ).generate_interface,
        )
        single_img_menu.add_command(
            label="Negative",
            command=UnaryImageOperation(
                image_io,
                "Neg",
                lambda x: linear_adj_image_wrapper(operators.negative_img_fun(x)),
            ).generate_interface,
        )
        single_img_menu.add_command(
            label="Equalize Histogram",
            command=UnaryImageOperation(
                image_io,
                "Equ",
                lambda x: linear_adj_image_wrapper(operators.histogram_equalization(x)),
            ).generate_interface,
        )
        # subtract_menu.add_command(label="B&W", command=generate_subtract_grey_operation_input)
        # multiply_menu = Menu(operation_menu, tearoff=0)
        # operation_menu.add_cascade(label="Multiply", menu=multiply_menu)
        # multiply_menu.add_command(label="By scalar", command=generate_multiply_by_scalar_input)
        # multiply_menu.add_command(label="Two images", command=generate_multiply_images_operation_input)
        # operation_menu.add_command(label="Copy", command=generate_copy_sub_image_input)
        # negative_menu = Menu(operation_menu, tearoff=0)
        # operation_menu.add_cascade(label="Negative", menu=negative_menu)
        # negative_menu.add_command(label="Colored Negative", command=lambda:
        #                           colored_negative_wrapper(interface.current_image, constants.WIDTH, constants.HEIGHT))
        # negative_menu.add_command(label="Grey Negative", command=lambda:
        #                           grey_negative_wrapper(interface.current_image, constants.WIDTH, constants.HEIGHT))
        #
