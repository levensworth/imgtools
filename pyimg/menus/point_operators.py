import os
from tkinter import Menu, messagebox, ttk

from pyimg.config import constants as constants
from pyimg.config.interface_info import InterfaceInfo
from pyimg.menus.io_menu import load_image
from pyimg.modules.image_operators import *
from pyimg.modules.imageio import convert_array_to_img, display_img, save_img


def apply_op(a_image: np.ndarray, another_image: np.ndarray, op) -> np.ndarray:
    """
    Given 2 matrices (image representations) apply the supplied matrix operator.
    For consistency, op will receive a_image as first param and another_image as second.


    :param a_image: ndarray representing an image
    :param another_image: ndarray representing an image
    :param op: function operation to apply
    :return: np.ndarray result of applying the operator
    """

    return op(a_image, another_image)


def load_left_image(interface):
    loaded_image = load_image(0, 0)
    if loaded_image is not None:
        interface.left_image = loaded_image


def load_right_image(interface):
    loaded_image = load_image(0, 1)
    if loaded_image is not None:
        interface.right_image = loaded_image


def generate_binary_operations_input(interface):
    if interface.current_image is not None or interface.image_to_copy is not None:
        interface.reset_parameters()
    else:
        interface.delete_widgets(interface.buttons_frame)
    image_1_button = ttk.Button(
        interface.buttons_frame,
        text="Load Image 1",
        command=lambda: load_left_image(interface),
    )
    image_2_button = ttk.Button(
        interface.buttons_frame,
        text="Load Image 2",
        command=lambda: load_right_image(interface),
    )
    image_1_button.grid(row=0, column=0)
    image_2_button.grid(row=0, column=1)


def binary_operation_validator(image_1, image_2):
    if image_1 is None or image_2 is None:
        return False
    else:
        return True


def generate_add_operation_input():
    interface = InterfaceInfo.get_instance()
    generate_binary_operations_input(interface)
    add_button = ttk.Button(
        interface.buttons_frame,
        text="Add",
        command=lambda: add_grey_image_wrapper(
            constants.WIDTH,
            constants.HEIGHT,
            interface.left_image,
            constants.WIDTH,
            constants.HEIGHT,
            interface.right_image,
        ),
    )
    add_button.grid(row=1, column=0)


def add_grey_image_wrapper(width_1, height_1, image_1, width_2, height_2, image_2):
    if binary_operation_validator(image_1, image_2):
        image_1, image_2 = order_img_by_size(np.array(image_1), np.array(image_2))
        result_img = apply_op(image_1, image_2, lambda x, y: x + y)
        # bring values to pixel range
        adjusted_img = linear_adjustment(result_img)
        img = convert_array_to_img(adjusted_img)
        display_img(img)
        save_img(adjusted_img, os.path.join(constants.SAVE_PATH, "added_img.jpg"))
    else:
        messagebox.showerror(
            title="Error", message="You need to upload image 1 and 2 to add"
        )


class PointOperatorMenu:
    """
    This class is a simple wrapper around tkinter menu object.
    it's self display as a menu in the top bar.
    """

    def __init__(self, menubar):
        interface = InterfaceInfo.get_instance()
        operation_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Operations", menu=operation_menu)

        operation_menu.add_command(label="Add", command=generate_add_operation_input)
        # subtract_menu = Menu(operation_menu, tearoff=0)
        # operation_menu.add_cascade(label="Subtract", menu=subtract_menu)
        # subtract_menu.add_command(label="Color", command=generate_subtract_colored_operation_input)
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
