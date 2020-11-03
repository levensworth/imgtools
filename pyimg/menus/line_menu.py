from tkinter import Menu

from pyimg.menus.operation_interface import (UnaryWithParamsAndRegionOperation,
                                             UnaryWithParamsImageOperation)
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

        pixel_exchange_menu = Menu(line_menu, tearoff=0)
        line_menu.add_cascade(label="Pixel exchange", menu=pixel_exchange_menu)

        pixel_exchange_menu.add_command(
            label="Image",
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

        pixel_exchange_menu.add_command(
            label="Sequence",
            command=UnaryWithParamsAndRegionOperation(
                image_io,
                "Apply",
                lambda image, region, epsilon, max_iterations, quantity: display_linear_adj_image_wrapper(
                    line_detection.pixel_exchange_in_sequence(image, image_io.file_name, region.start_x, region.start_y,
                                                              region.end_x, region.end_y, epsilon, int(max_iterations),
                                                              int(quantity))

                ),
                params=["epsilon", "max_iterations", "quantity"],
            ).generate_interface,
        )

        hough_detection_menu = Menu(line_menu, tearoff=0)
        line_menu.add_cascade(label="Hough", menu=hough_detection_menu)

        hough_detection_menu.add_command(
            label="Line",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Apply",
                lambda image, epsilon, threshold: display_linear_adj_image_wrapper(
                    line_detection.hough_line_detector(image, epsilon, int(threshold))

                ),
                params=["epsilon", "threshold"],
            ).generate_interface,
        )

        hough_detection_menu.add_command(
            label="Circle",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Apply",
                lambda image, epsilon, threshold: display_linear_adj_image_wrapper(
                    line_detection.hough_circle_detector(image, epsilon, int(threshold))

                ),
                params=["epsilon", "threshold"],
            ).generate_interface,
        )
