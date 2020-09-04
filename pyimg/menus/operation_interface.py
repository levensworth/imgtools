from abc import ABC, abstractmethod
from tkinter import Entry, messagebox, ttk, BooleanVar

from pyimg.config import constants as constants
from pyimg.menus.io_menu import ImageIO


class ImageOperationInput:
    def __init__(self, image_io: ImageIO, cols: int):
        self.image_io = image_io
        self.cols = cols
        self.interface = image_io.interface

    def generate_inputs(self, qty):
        self.interface.remove_buttons()
        for i in range(qty):
            row, column = 0, i
            image_button = ttk.Button(
                self.interface.buttons_frame,
                text="Load Image " + str(i + 1),
                command=self.image_io.full_load_image,
            )
            image_button.grid(row=row, column=column)

    def get_input(self):
        return tuple(self.interface.images)


class ImageOperation(ABC):
    def __init__(self, image_io: ImageIO, operands: int, button_text: str):
        self.image_input = ImageOperationInput(image_io, constants.IMAGE_COLS)
        self.operands = operands
        self.button_text = button_text
        self.extra_params = {}
        self.extra_bool_params = {}

    @abstractmethod
    def command_apply(self):
        pass

    def is_ready(self) -> bool:
        if len(self.image_input.get_input()) == self.operands:
            return True
        else:
            return False

    def generate_interface(self):
        self.image_input.generate_inputs(self.operands)
        for i, param_text in enumerate(self.extra_params.keys()):
            ttk.Label(
                self.image_input.interface.buttons_frame,
                text=param_text,
                background=constants.TOP_COLOR,
            ).grid(row=1, column=2 * i)
            param = Entry(self.image_input.interface.buttons_frame)
            param.grid(row=1, column=1 + 2 * i)
            self.extra_params[param_text] = param

        count = 0
        for param_texts in self.extra_bool_params.keys():
            for i, param_text in enumerate(param_texts):
                ttk.Radiobutton(
                    self.image_input.interface.buttons_frame,
                    text=param_text,
                    value=i == 0,
                    variable=self.extra_bool_params[param_texts],
                ).grid(row=2, column=count)
                count += 1

        add_button = ttk.Button(
            self.image_input.interface.buttons_frame,
            text=self.button_text,
            command=self.command_apply,
        )
        extra_row = int(len(self.extra_params) > 0)
        extra_row = extra_row + int(len(self.extra_bool_params) > 0)
        add_button.grid(row=1 + extra_row, column=0)

    def add_button_input(self, param_text):
        self.extra_params[param_text] = None
        return self

    def add_radio_button_input(self, param_text):
        self.extra_bool_params[param_text] = BooleanVar()
        self.extra_bool_params[param_text].set(True)
        return self

    def get_params(self):
        return {
            **{k: float(v.get())
               for k, v in self.extra_params.items()
               if self.extra_params[k].get() != ''},
            **{k[0]: float(v.get())
               for k, v in self.extra_bool_params.items()}
        }


class UnaryImageOperation(ImageOperation):
    def __init__(self, image_io: ImageIO, button_text: str, func):
        super(UnaryImageOperation, self).__init__(image_io, 1, button_text)
        self.func = func

    def command_apply(self):
        if self.is_ready():
            image = self.image_input.get_input()[0]
            return self.func(image)
        else:
            messagebox.showerror(
                title="Error",
                message="You need to upload one image for this operation.",
            )


class UnaryWithParamsImageOperation(UnaryImageOperation):
    def __init__(self, image_io: ImageIO, button_text: str, func, params):
        super(UnaryImageOperation, self).__init__(image_io, 1, button_text)
        self.func = func
        self.params = params
        for param in self.params:
            self.add_button_input(param)

    def is_ready(self) -> bool:
        if super().is_ready():
            if len(self.params) == len(self.get_params()):
                return True
        return False

    def command_apply(self):
        if self.is_ready():
            image = self.image_input.get_input()[0]
            return self.func(**{**{"image": image}, **self.get_params()})
        else:
            messagebox.showerror(
                title="Error",
                message="You need to upload one image for this operation and set the parameters.",
            )


class BinaryImageOperation(ImageOperation):
    def __init__(self, image_io: ImageIO, button_text: str, func):
        super(BinaryImageOperation, self).__init__(image_io, 2, button_text)
        self.func = func

    def command_apply(self):
        if self.is_ready():
            image1, image2 = self.image_input.get_input()

            return self.func(image1, image2)
        else:
            messagebox.showerror(
                title="Error",
                message="You need to upload two images for this operation.",
            )
