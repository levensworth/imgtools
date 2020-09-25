import tkinter as tk
from tkinter import Menu

from pyimg.menus.operation_interface import (BinaryImageOperation,
                                             UnaryImageOperation,
                                             UnaryWithParamsImageOperation)
from pyimg.menus.point_operators import display_linear_adj_image_wrapper
from pyimg.models.image import thresholding, ImageImpl


def display_result(image: ImageImpl):
    img, thresholds = thresholding.otsu_thresholding(image)
    display_linear_adj_image_wrapper(img)

    text_window = tk.Tk()
    message = tk.Text(text_window, height=4, width=50, font=("Helvetica", 20))
    message.pack()

    if image.channels == 3:
        text = 'Computed Thresholds: R: {}, G: {}, B: {}'.format(thresholds[0],
                                                                 thresholds[1],
                                                                 thresholds[2])
    else:
        text = 'Computed Threshold: {}'. format(thresholds[0])
    message.insert(tk.END, text)
    text_window.mainloop()


class ThresholdMenu:
    """
    This class is a simple wrapper around tkinter menu object.
    it's self display as a menu in the top bar.
    """

    def __init__(self, menubar, image_io):
        filter_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Thresholding", menu=filter_menu)

        filter_menu.add_command(
            label="Global",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Global",
                display_result,
                params=[],
            ).generate_interface,
        )

        filter_menu.add_command(
            label="Otsu",
            command=UnaryWithParamsImageOperation(
                image_io,
                "Otsu",
                display_result,
                params=[],
            ).generate_interface,
        )
