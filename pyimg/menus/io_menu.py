import os
from pathlib import Path
from tkinter import Menu, filedialog, messagebox, ttk
from tkinter.filedialog import asksaveasfilename

import imageio
import numpy as np
from PIL import Image, ImageTk

from pyimg.config import constants as constants
from pyimg.config.constants import IMG_EXTENSIONS
from pyimg.config.interface_info import InterfaceInfo
from pyimg.modules import image_io


def open_file_name() -> str:
    """
    Display a finder for the file selection.
    :return
        str: file path
    """
    file_name = filedialog.askopenfilename(
        title="Choose Image", filetypes=IMG_EXTENSIONS
    )
    if file_name:
        return file_name
    else:
        return ""


def load_image():
    interface = InterfaceInfo.get_instance()
    file_name = open_file_name()
    if file_name:
        interface.current_image_name = file_name
        if file_name.endswith(".RAW"):
            raw_image = image_io.load_raw_image(Path(file_name))
            Image.fromarray(raw_image)
        else:
            # opens the image
            image = Image.open(file_name)
        # resize the image and apply a high-quality down sampling filter
        image = image.resize((constants.WIDTH, constants.HEIGHT), Image.ANTIALIAS)
        image_instance = image
        # PhotoImage class is used to add image to widgets, icons etc
        image = ImageTk.PhotoImage(image)
        # create a label
        # panel = ttk.Label(interface.image_frame, image=image)
        interface.generate_canvas()
        interface.canvas.create_image(0, 0, image=image, anchor="nw")
        # set the image as img
        interface.canvas.image = image
        return image_instance


def load_image_wrapper():
    interface = InterfaceInfo.get_instance()
    interface.remove_images()
    if interface.current_image is None:
        interface.current_image = load_image()
        # harris_method(interface.current_image, constants.HEIGHT, constants.WIDTH, 0.8)
        # sift_method(interface.current_image, constants.HEIGHT, constants.WIDTH)
        # compare_images(interface.current_image, constants.HEIGHT, constants.WIDTH, interface.current_image, constants.HEIGHT, constants.WIDTH, 400)
    elif interface.image_to_copy is None:
        interface.image_to_copy = load_image(0, 1)
    else:
        messagebox.showerror(
            title="Error",
            message="You can't upload more than two images. If you want to change"
            ' one click on the "Clean image" button first',
        )


def save_image():
    interface = InterfaceInfo.get_instance()
    if interface.current_image is None:
        messagebox.showerror(
            title="Error", message="You must upload an image to save it"
        )
    else:
        image = interface.current_image
        image_info = image.filename = asksaveasfilename(
            initialdir=os.getcwd(),
            title="Select file",
            filetypes=IMG_EXTENSIONS,
        )

        # image.convert("I")
        # image.save(image_info)
        image_content = image.convert("L")
        image_matrix = np.array(image_content)
        imageio.imwrite(image_info, image_matrix)


class ImageMenu:
    """
    This class is a simple wrapper around tkinter menu object.
    it's self display as a menu in the top bar.
    """

    def __init__(self, menubar):
        interface = InterfaceInfo.get_instance()
        image_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Image", menu=image_menu)
        image_menu.add_command(label="Open", command=load_image_wrapper)
        image_menu.add_command(label="Save", command=save_image)
        image_menu.add_separator()
        image_menu.add_command(label="Exit", command=interface.root.quit)
