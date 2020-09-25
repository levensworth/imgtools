from tkinter import Menu

from pyimg.menus.operation_interface import (BinaryImageOperation,
                                             UnaryImageOperation,
                                             UnaryWithParamsImageOperation)
from pyimg.menus.point_operators import display_linear_adj_image_wrapper
from pyimg.models.image import filters


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
                lambda image, kernel_size: display_linear_adj_image_wrapper(
                    filters.mean_filter_fast(image, kernel_size)
                ),
                params=["kernel_size"],
            ).generate_interface,
        )
        filter_menu.add_command(
            label="Median Filter",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Median",
                lambda image, kernel_size: display_linear_adj_image_wrapper(
                    filters.median_filter_fast(image, kernel_size)
                ),
                params=["kernel_size"],
            ).generate_interface,
        )
        filter_menu.add_command(
            label="Weighted Median Filter",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Median",
                lambda image, kernel_size: display_linear_adj_image_wrapper(
                    filters.weighted_median_filter(image, kernel_size)
                ),
                params=["kernel_size"],
            ).generate_interface,
        )
        filter_menu.add_command(
            label="Gaussian filter",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Gaussian",
                lambda image, sigma, kernel_size: display_linear_adj_image_wrapper(
                    filters.gaussian_filter_fast(image, kernel_size, sigma)
                ),
                params=["sigma", "kernel_size"],
            ).generate_interface,
        )
        filter_menu.add_command(
            label="High Filter",
            command=UnaryWithParamsImageOperation(
                image_io,
                "High",
                lambda image, kernel_size: display_linear_adj_image_wrapper(
                    filters.high_filter_fast(image, kernel_size)
                ),
                params=["kernel_size"],
            ).generate_interface,
        )
        filter_menu.add_command(
            label="Threshold Filter",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Threshold",
                lambda image, threshold: display_linear_adj_image_wrapper(
                    filters.threshold_filter(image, threshold)
                ),
                params=["threshold"],
            ).generate_interface,
        )
