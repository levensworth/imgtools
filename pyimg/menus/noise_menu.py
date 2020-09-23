import datetime
import os
from tkinter import Menu, messagebox

from pyimg.config import constants
from pyimg.menus.io_menu import ImageIO
from pyimg.menus.operation_interface import (
    UnaryImageOperation,
    UnaryWithParamsImageOperation,
)
from pyimg.models.image import ImageImpl, noise, operators
from pyimg.models.random_number.generator import (
    ExponentialGenerator,
    GaussianGenerator,
    Generator,
    RayleighGenerator,
)
from pyimg.modules.image_io import display_img, save_img


class NoiseOperation(UnaryWithParamsImageOperation):
    def __init__(self, image_io: ImageIO, button_text: str, func, params, bool_params):
        super(NoiseOperation, self).__init__(image_io, button_text, func, params)
        self.bool_params = bool_params
        self.params = self.params + self.bool_params

        for param in self.bool_params:
            self.add_radio_button_input(param)


def noise_image_wrapper(image: ImageImpl, generator: Generator, **kwargs):
    threshold = kwargs["threshold"]
    if 0.0 <= threshold <= 1.0:
        kwargs.pop("threshold", None)
        is_additive = bool(kwargs["is additive"])
        kwargs.pop("is additive", None)

        kwargs["size"] = image.array.size
        generator.kwargs = kwargs
        noise_img = noise.apply_noise(image, generator, is_additive, threshold)
        adjusted_img = operators.linear_adjustment(noise_img)
        img = adjusted_img.convert_to_pil()
        display_img(img)
        save_img(
            img,
            os.path.join(
                constants.SAVE_PATH,
                "result_img " + str(datetime.datetime.now()) + ".jpg",
            ),
        )

    else:
        messagebox.showerror(
            title="Error", message="Threshold should be between 0 and 1."
        )


def salt_peper_image_wrapper(image: ImageImpl, **kwargs):
    p0 = kwargs["p0"]
    p1 = kwargs.pop("p1", None)
    if 0.0 <= p0 <= 1.0 and (p1 is None or 0.0 <= p1 <= 1.0):
        salt_img = noise.salt_and_pepper_apply(image, p0, p1)
        adjusted_img = operators.linear_adjustment(salt_img)
        img = adjusted_img.convert_to_pil()
        display_img(img)
        save_img(
            img,
            os.path.join(
                constants.SAVE_PATH,
                "result_img " + str(datetime.datetime.now()) + ".jpg",
            ),
        )

    else:
        messagebox.showerror(
            title="Error", message="p0 and p1 should be between 0 and 1."
        )


class NoiseImageMenu:
    """
    This class is a simple wrapper around tkinter menu object.
    it's self display as a menu in the top bar.
    """

    def __init__(self, menubar, image_io):
        noise_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Noise", menu=noise_menu)

        noise_menu.add_command(
            label="Gaussian",
            command=NoiseOperation(
                image_io,
                "Apply Gaussian Noise",
                lambda image, **kwargs: noise_image_wrapper(
                    image, GaussianGenerator(), **kwargs
                ),
                params=["threshold", "mean", "std"],
                bool_params=[("is additive", "is multiplicative")],
            ).generate_interface,
        )
        noise_menu.add_command(
            label="Rayleigh",
            command=NoiseOperation(
                image_io,
                "Apply Rayleigh Noise",
                lambda image, **kwargs: noise_image_wrapper(
                    image, RayleighGenerator(), **kwargs
                ),
                params=["threshold", "xi"],
                bool_params=[("is additive", "is_multiplicative")],
            ).generate_interface,
        )
        noise_menu.add_command(
            label="Exponential",
            command=NoiseOperation(
                image_io,
                "Apply Exponential Noise",
                lambda image, **kwargs: noise_image_wrapper(
                    image, ExponentialGenerator(), **kwargs
                ),
                params=["threshold", "alpha"],
                bool_params=[("is additive", "is_multiplicative")],
            ).generate_interface,
        )
        noise_menu.add_command(
            label="Salt and Pepper",
            command=NoiseOperation(
                image_io,
                "Apply Salt and Pepper",
                lambda image, **kwargs: salt_peper_image_wrapper(image, **kwargs),
                params=["p0", "p1"],
                bool_params=[],
            ).generate_interface,
        )
