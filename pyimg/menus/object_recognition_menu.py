import tkinter as tk
from tkinter import Menu

from pyimg.menus.operation_interface import BinaryWithParamsImageOperation
from pyimg.menus.point_operators import display_linear_adj_image_wrapper
from pyimg.models.image import ImageImpl, object_recognition


def display_result(image: ImageImpl, acepted: bool, descriptors1_qty: int, descriptors2_qty: int, matches_qty: int):
    display_linear_adj_image_wrapper(image)

    text_window = tk.Tk()
    message = tk.Text(text_window, height=5, width=50, font=("Helvetica", 20))
    message.pack()

    text = "Results:\nQuantity of descriptors in image one: {}\nQuantity of descriptors in image two: {}\n" \
           "Quantity matched descriptors: {}\n".format(descriptors1_qty, descriptors2_qty, matches_qty)

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
            command=BinaryWithParamsImageOperation(
                image_io,
                "Apply",
                lambda image1, image2, threshold, acceptance: display_result(
                    *object_recognition.compare_images_sift(image1, image2, threshold, acceptance)
                ),
                params=["threshold", "acceptance"],
            ).generate_interface,
        )

