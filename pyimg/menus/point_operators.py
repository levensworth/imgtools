import math
import os
from tkinter import Entry, Menu, messagebox, ttk

from pyimg.config import constants as constants
from pyimg.config.interface_info import InterfaceInfo
from pyimg.menus.io_menu import load_image
from pyimg.modules.image_io import convert_array_to_img, display_img, save_img
from pyimg.modules.image_operators import *
from pyimg.modules.image_binary_operators import *


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


def apply_unary_op(a_image: np.ndarray, op) -> np.ndarray:
    """
    Given 1 matrix (image representations) apply the supplied matrix operator.
    For consistency, op will receive a_image as first param and another_image as second.


    :param a_image: ndarray representing an image
    :param op: function operation to apply
    :return: np.ndarray result of applying the operator
    """

    return op(a_image)


def load_left_image(interface):
    loaded_image = load_image()
    if loaded_image is not None:
        interface.left_image = loaded_image


def load_right_image(interface):
    loaded_image = load_image()
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


def generate_unary_operations_input(interface):
    if interface.current_image is not None or interface.image_to_copy is not None:
        interface.reset_parameters()
    else:
        interface.delete_widgets(interface.buttons_frame)
    image_button = ttk.Button(
        interface.buttons_frame,
        text="Load Image",
        command=lambda: load_left_image(interface),
    )
    image_button.grid(row=0, column=0)


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
        command=lambda: add_image_wrapper(
            interface.left_image,
            interface.right_image,
        ),
    )
    add_button.grid(row=1, column=0)


def generate_compress_operation_input():
    interface = InterfaceInfo.get_instance()
    generate_unary_operations_input(interface)
    add_button = ttk.Button(
        interface.buttons_frame,
        text="CDR",
        command=lambda: dynamic_compression_image_wrapper(
            interface.left_image,
        ),
    )
    add_button.grid(row=1, column=0)


def add_image_wrapper(image_1, image_2):
    if binary_operation_validator(image_1, image_2):
        image_1, image_2 = order_img_by_size(
            np.array(image_1).astype(float), np.array(image_2).astype(float)
        )
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


def dynamic_compression_image_wrapper(image):
    img_array = np.array(image).astype(float)
    c = (constants.MAX_PIXEL_VALUE - 1) / math.log10(1 + np.amax(img_array))
    result_img = apply_unary_op(img_array, lambda x: c * np.log10(1 + x))
    adjusted_img = linear_adjustment(result_img)

    img = convert_array_to_img(adjusted_img)
    display_img(img)
    save_img(adjusted_img, os.path.join(constants.SAVE_PATH + "compressed_image.ppm"))


def generate_dif_operation_input():
    interface = InterfaceInfo.get_instance()
    generate_binary_operations_input(interface)
    add_button = ttk.Button(
        interface.buttons_frame,
        text="Dif",
        command=lambda: dif_image_wrapper(
            interface.left_image,
            interface.right_image,
        ),
    )
    add_button.grid(row=1, column=0)


def dif_image_wrapper(image_1, image_2):
    if binary_operation_validator(image_1, image_2):

        image_1 = np.array(image_1).astype(float)
        image_2 = np.array(image_2).astype(float)

        result_img = apply_op(image_1, image_2, lambda x, y: x - y)
        # bring values to pixel range
        adjusted_img = linear_adjustment(result_img)
        img = convert_array_to_img(adjusted_img)
        display_img(img)
        save_img(adjusted_img, os.path.join(constants.SAVE_PATH, "subtract_img.jpg"))
    else:
        messagebox.showerror(
            title="Error", message="You need to upload image 1 and 2 to subtract"
        )


def generate_scalar_multiplication():
    interface = InterfaceInfo.get_instance()
    interface.reset_parameters()
    load_image_button = ttk.Button(
        interface.buttons_frame,
        text="Load Image",
        command=lambda: load_left_image(interface),
    )
    load_image_button.grid(row=0, column=0)
    ttk.Label(
        interface.buttons_frame, text="Scalar", background=constants.TOP_COLOR
    ).grid(row=1, column=0)
    scalar = Entry(interface.buttons_frame)
    scalar.grid(row=1, column=1)
    multiply_button = ttk.Button(
        interface.buttons_frame,
        text="Multiply",
        command=lambda: mul_img_wrapper(interface.left_image, scalar.get()),
    )
    multiply_button.grid(row=2, column=0)


def mul_img_wrapper(image_1, scalar):
    image_1 = np.array(image_1).astype(float)

    result_img = image_1.dot(float(scalar))
    # bring values to pixel range
    adjusted_img = linear_adjustment(result_img)
    img = convert_array_to_img(adjusted_img)
    display_img(img)
    save_img(adjusted_img, os.path.join(constants.SAVE_PATH, "stretch_img.jpg"))


def generate_gamma_operation():
    interface = InterfaceInfo.get_instance()
    interface.reset_parameters()
    load_image_button = ttk.Button(
        interface.buttons_frame,
        text="Load Image",
        command=lambda: load_left_image(interface),
    )
    load_image_button.grid(row=0, column=0)
    ttk.Label(
        interface.buttons_frame, text="Scalar", background=constants.TOP_COLOR
    ).grid(row=1, column=0)
    scalar = Entry(interface.buttons_frame)
    scalar.grid(row=1, column=1)
    multiply_button = ttk.Button(
        interface.buttons_frame,
        text="Multiply",
        command=lambda: gamma_img_wrapper(interface.left_image, scalar.get()),
    )
    multiply_button.grid(row=2, column=0)


def gamma_img_wrapper(a_img, c_value):
    a_img = np.array(a_img).astype(float)

    result_img = gamma_fun(a_img, float(c_value))
    # bring values to pixel range
    adjusted_img = linear_adjustment(result_img)
    img = convert_array_to_img(adjusted_img)
    display_img(img)
    save_img(adjusted_img, os.path.join(constants.SAVE_PATH, "gamma_img.jpg"))


def generate_negative():
    interface = InterfaceInfo.get_instance()
    interface.reset_parameters()
    load_image_button = ttk.Button(
        interface.buttons_frame,
        text="Load Image",
        command=lambda: load_left_image(interface),
    )
    load_image_button.grid(row=0, column=0)

    negative_button = ttk.Button(
        interface.buttons_frame,
        text="Negative",
        command=lambda: negative_img_wrapper(interface.left_image),
    )
    negative_button.grid(row=2, column=0)


def negative_img_wrapper(a_img):
    negative = negative_img_fun(np.array(a_img))
    adjusted = linear_adjustment(negative)
    img = convert_array_to_img(adjusted)
    display_img(img)
    save_img(adjusted, os.path.join(constants.SAVE_PATH, "negative_img.jpg"))


def generate_equalized_image_input():
    interface = InterfaceInfo.get_instance()
    generate_unary_operations_input(interface)
    add_button = ttk.Button(
        interface.buttons_frame,
        text="EQ",
        command=lambda: equalize_image_wrapper(
            interface.left_image,
        ),
    )
    add_button.grid(row=1, column=0)


def equalize_image_wrapper(a_img):
    equalized = histogram_equalization(np.array(a_img))
    adjusted = linear_adjustment(equalized)
    img = convert_array_to_img(adjusted)
    display_img(img)
    save_img(adjusted, os.path.join(constants.SAVE_PATH, "equalized_img.jpg"))


class PointOperatorMenu:
    """
    This class is a simple wrapper around tkinter menu object.
    it's self display as a menu in the top bar.
    """

    def __init__(self, menubar):
        interface = InterfaceInfo.get_instance()
        operation_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Operations", menu=operation_menu)

        binary_operators_menu = Menu(operation_menu, tearoff=0)
        operation_menu.add_cascade(label="Binary operators", menu=binary_operators_menu)

        binary_operators_menu.add_command(
            label="Add", command=generate_add_operation_input
        )

        binary_operators_menu.add_command(
            label="Subtract", command=generate_dif_operation_input
        )

        single_img_menu = Menu(operation_menu, tearoff=0)
        operation_menu.add_cascade(label="Single operators", menu=single_img_menu)

        single_img_menu.add_command(
            label="CRD", command=generate_compress_operation_input
        )

        single_img_menu.add_command(
            label="Multiply", command=generate_scalar_multiplication
        )
        single_img_menu.add_command(label="gamma", command=generate_gamma_operation)
        single_img_menu.add_command(label="negative", command=generate_negative)
        single_img_menu.add_command(label="equalize histogram", command=generate_equalized_image_input)
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
