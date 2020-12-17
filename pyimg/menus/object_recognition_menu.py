import tkinter as tk
from tkinter import Menu

from pyimg.menus.auto_object_recognition_menu import automated_result
from pyimg.menus.operation_interface import BinaryWithBoolAndStringParamsOperation
from pyimg.menus.point_operators import display_linear_adj_image_wrapper
from pyimg.models.image import ImageImpl, object_recognition


def display_result(
    image: ImageImpl,
    acepted: bool,
    descriptors1_qty: int,
    descriptors2_qty: int,
    matches_qty: int,
    matches_mean: float,
    matches_std: float,
):
    display_linear_adj_image_wrapper(image)

    text_window = tk.Tk()
    message = tk.Text(text_window, height=7, width=100, font=("Helvetica", 20))
    message.pack()

    text = (
        "Results:\nQuantity of descriptors in image one: {}\n"
        "Quantity of descriptors in image two: {}\n"
        "Quantity matched descriptors: {}\n"
        "Mean of all normalized valid distances between descriptors: {:.3f} (0 is when a descriptor is more similar to "
        "another)\n"
        "Standard deviation of all normalized valid distances between descriptors: {:.3f}\n".format(
            descriptors1_qty, descriptors2_qty, matches_qty, matches_mean, matches_std
        )
    )

    if acepted:
        text += "Conclusion: They are the same image.\n"
    else:
        text += "Conclusion: They are not the same image.\n"

    message.insert(tk.END, text)
    text_window.mainloop()


class ObjectRecognitionMenu:
    """
    This class is a simple wrapper around tkinter menu object.
    it's self display as a menu in the top bar.
    """

    def __init__(self, menubar, image_io):
        filter_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Object recognition", menu=filter_menu)

        filter_menu.add_command(
            label="SIFT",
            command=BinaryWithBoolAndStringParamsOperation(
                image_io,
                "Apply",
                lambda image1, image2, threshold, acceptance, similarity, validate_second_min,
                       validate_second_threshold:
                display_result(
                    *object_recognition.compare_images_sift(
                        image1, image2, threshold, acceptance, similarity, validate_second_min,
                        validate_second_threshold
                    )
                ),
                params=["threshold", "acceptance", "validate_second_threshold"],
                bool_params=[("validate_second_min", "not_validate_second_min")],
                str_params=["similarity"],
            ).generate_interface,
        )

        filter_menu.add_command(
            label="Automated SIFT",
            command=BinaryWithBoolAndStringParamsOperation(
                image_io,
                "Apply",
                lambda image1, image2, threshold, transform_type, validate_second_min,
                       validate_second_threshold:
                automated_result(
                    image1, image2, threshold, transform_type, validate_second_min,
                    validate_second_threshold
                )
                ,
                params=["threshold", "validate_second_threshold"],
                bool_params=[("validate_second_min", "not_validate_second_min")],
                str_params=["transform_type"],
            ).generate_interface,
        )
