from tkinter import Menu

from pyimg.menus.operation_interface import (UnaryWithParamsAndRegionOperation)
from pyimg.menus.point_operators import display_linear_adj_image_wrapper
from pyimg.models.image import line_detection


class LineMenu:
    """
    This class is a simple wrapper around tkinter menu object.
    it's self display as a menu in the top bar.
    """

    def __init__(self, menubar, image_io):
        line_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Line Detection", menu=line_menu)

        line_menu.add_command(
            label="Pixel exchange",
            command=UnaryWithParamsAndRegionOperation(
                image_io,
                "Apply",
                lambda image, region, epsilon, max_iterations: display_linear_adj_image_wrapper(
                    line_detection.pixel_exchange(image, region.start_x, region.start_y, region.end_x, region.end_y,
                                                  epsilon, int(max_iterations))

                ),
                params=["epsilon", "max_iterations"],
            ).generate_interface,
        )
