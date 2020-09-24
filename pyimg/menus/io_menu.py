from pathlib import Path
from tkinter import Menu, filedialog, messagebox

import imageio
import numpy as np
from PIL import Image, ImageTk

from pyimg.config import constants as constants
from pyimg.config.constants import IMG_EXTENSIONS
from pyimg.models.image import ImageImpl
from pyimg.modules import image_io


class ImageIO:
    def __init__(self, interface):
        self.interface = interface

    def full_load_image(self):
        file_name = self.choose_file_name("Select file")
        image = self.load_image(file_name)

        # PhotoImage class is used to add image to widgets, icons etc
        image_disp = ImageTk.PhotoImage(image)
        self.interface.generate_canvas()
        self.interface.canvas.create_image(0, 0, image=image_disp, anchor="nw")
        self.interface.canvas.image = image_disp
        image_matrix = np.array(image)
        dims = len(image_matrix.shape)
        image_matrix = (
            np.expand_dims(image_matrix, axis=dims) if dims == 2 else image_matrix
        )

        self.interface.images.append(ImageImpl(image_matrix))

        return ImageImpl(image_matrix)

    def full_save_image(self):
        if self.interface.result_image is None:
            messagebox.showerror(title="Error", message="There is no image to save.")
        else:
            image = self.interface.result_image
            image_name = self.choose_file_name("Save as")
            self.save_image(image, image_name)

    def reset(self):
        self.interface.reset_parameters()

    @staticmethod
    def load_image(file_name):
        if file_name.endswith(".RAW"):
            raw_image = image_io.load_raw_image(Path(file_name))
            image = Image.fromarray(raw_image)
        else:
            image = Image.open(file_name)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
        # resize the image and apply a high-quality down sampling filter
        image = image.resize((constants.WIDTH, constants.HEIGHT), Image.ANTIALIAS)
        return image

    @staticmethod
    def save_image(image: Image, image_info):
        image_content = image.convert("L")
        image_matrix = np.array(image_content)
        return imageio.imwrite(image_info, image_matrix)

    @staticmethod
    def choose_file_name(title: str) -> str:
        """
        Display a finder for the file selection.
        :return
            str: file path
        """
        if title == "Save as":
            op = filedialog.asksaveasfilename
        else:
            op = filedialog.askopenfilename

        file_name = op(title=title, filetypes=IMG_EXTENSIONS)
        if file_name:
            return file_name
        else:
            return ""


class ImageMenu:
    """
    This class is a simple wrapper around tkinter menu object.
    it's self display as a menu in the top bar.
    """

    def __init__(self, menubar, interface):
        self.image_menu = Menu(menubar, tearoff=0)
        self.image_io = ImageIO(interface)
        menubar.add_cascade(label="Image", menu=self.image_menu)
        self.image_menu.add_command(
            label="Open Image", command=self.image_io.full_load_image
        )
        self.image_menu.add_command(
            label="Save Image As", command=self.image_io.full_save_image
        )
        self.image_menu.add_separator()
        self.image_menu.add_command(label="Exit", command=interface.root.quit)
