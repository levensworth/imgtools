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


class FilterMenu:
    """
    This class is a simple wrapper around tkinter menu object.
    it's self display as a menu in the top bar.
    """

    def __init__(self, menubar, image_io):
        filter_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Filters", menu=filter_menu)

        filter_menu.add_command(
            label="Mean Filter",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Mean",
                lambda m1, kernel_size: linear_adj_image_wrapper(
                    operators.mean_filter(m1, kernel_size)
                ),
                params=["kernel_size"],
            ).generate_interface,
        )
