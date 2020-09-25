from tkinter import Menu

from pyimg.menus.operation_interface import (BinaryImageOperation,
                                             UnaryImageOperation,
                                             UnaryWithParamsImageOperation)
from pyimg.menus.point_operators import display_linear_adj_image_wrapper
from pyimg.models.image import (border_detection,
                                multi_direction_border_detection)


class BorderMenu:
    """
    This class is a simple wrapper around tkinter menu object.
    it's self display as a menu in the top bar.
    """

    def __init__(self, menubar, image_io):
        filter_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Border Detection", menu=filter_menu)

        filter_menu.add_command(
            label="Prewitt operator",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Prewitt",
                lambda image: display_linear_adj_image_wrapper(
                    border_detection.prewitt_detector(image)
                ),
                params=[],
            ).generate_interface,
        )

        filter_menu.add_command(
            label="Sobel operator",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Sobel",
                lambda image: display_linear_adj_image_wrapper(
                    border_detection.sobel_detector(image)
                ),
                params=[],
            ).generate_interface,
        )

        filter_menu.add_command(
            label="Laplacian operator",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Laplace",
                lambda image, threshold: display_linear_adj_image_wrapper(
                    border_detection.laplacian_border_detection(image, threshold)
                ),
                params=["threshold"],
            ).generate_interface,
        )

        filter_menu.add_command(
            label="Gaussian laplacian operator",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Gauss",
                lambda image, threshold, sigma, kernel_size: display_linear_adj_image_wrapper(
                    border_detection.gaussian_laplacian_detection(
                        image, threshold, sigma, kernel_size
                    )
                ),
                params=["threshold", "sigma", "kernel_size"],
            ).generate_interface,
        )

        filter_menu.add_command(
            label="Prewitt multi dir operator",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Prewitt",
                lambda image, rotation_angle: display_linear_adj_image_wrapper(
                    multi_direction_border_detection.prewitt_border_detection(image, rotation_angle)),
                params=['rotation_angle'],
            ).generate_interface,
        )

        filter_menu.add_command(
            label="Sobel multi dir operator",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Sobel",
                lambda image, rotation_angle: display_linear_adj_image_wrapper(
                    multi_direction_border_detection.sobel_border_detection(image, rotation_angle)),
                params=['rotation_angle'],
            ).generate_interface,
        )

        filter_menu.add_command(
            label="ITBA multi dir operator",
            command=UnaryWithParamsImageOperation(
                image_io,
                "ITBA",
                lambda image, rotation_angle: display_linear_adj_image_wrapper(
                    multi_direction_border_detection.itba_border_detection(image, rotation_angle)),
                params=['rotation_angle'],
            ).generate_interface,
        )

        filter_menu.add_command(
            label="Kirish multi dir operator",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Kirish",
                lambda image, rotation_angle: display_linear_adj_image_wrapper(
                    multi_direction_border_detection.kirish_border_detection(image, rotation_angle)),
                params=['rotation_angle'],
            ).generate_interface,
        )
