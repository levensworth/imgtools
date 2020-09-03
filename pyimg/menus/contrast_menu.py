import os
from tkinter import Entry, Menu, messagebox, ttk

from pyimg.config import constants
from pyimg.config.interface_info import InterfaceInfo
from pyimg.modules.image_filters import *
from pyimg.modules.image_io import convert_array_to_img, display_img, save_img
from pyimg.modules.image_operators import linear_adjustment


class FunctionMenu:
    def __init__(self, menubar):
        interface = InterfaceInfo.get_instance()
        function_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Filters", menu=function_menu)
        function_menu.add_command(
            label="B&W Threshold Filter", command=b_w_threshold_view
        )
        function_menu.add_command(
            label="Threshold Filter", command=color_threshold_view
        )


def b_w_threshold_view():
    interface = InterfaceInfo.get_instance()
    interface.delete_widgets(interface.buttons_frame)
    if interface.current_image is not None:
        ttk.Label(
            interface.buttons_frame, text="Threshold", background=constants.TOP_COLOR
        ).grid(row=0, column=0)
        threshold = Entry(interface.buttons_frame)
        threshold.grid(row=0, column=1)
        apply_threshold = ttk.Button(
            interface.buttons_frame,
            text="Apply",
            command=lambda: threshold_wrapper(
                interface.current_image.convert("L"), float(threshold.get())
            ),
        )
        apply_threshold.grid(row=1, column=0)
    else:
        messagebox.showerror(
            title="Error", message="You must upload an image to apply a threshold"
        )


def color_threshold_view():
    interface = InterfaceInfo.get_instance()
    interface.delete_widgets(interface.buttons_frame)
    if interface.current_image is not None:
        ttk.Label(
            interface.buttons_frame, text="Threshold", background=constants.TOP_COLOR
        ).grid(row=0, column=0)
        threshold = Entry(interface.buttons_frame)
        threshold.grid(row=0, column=1)
        apply_threshold = ttk.Button(
            interface.buttons_frame,
            text="Apply",
            command=lambda: threshold_wrapper(
                interface.current_image, float(threshold.get())
            ),
        )
        apply_threshold.grid(row=1, column=0)
    else:
        messagebox.showerror(
            title="Error", message="You must upload an image to apply a threshold"
        )


def threshold_wrapper(a_img, threshold):
    filtered = threshold_filter(np.array(a_img).astype(float), threshold)
    adjusted = linear_adjustment(filtered)
    img = convert_array_to_img(adjusted)
    display_img(img)
    save_img(adjusted, os.path.join(constants.SAVE_PATH, "threshold_img.jpg"))
