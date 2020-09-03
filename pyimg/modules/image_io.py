import logging
import os
from pathlib import Path

import numpy as np
import rawpy  # nt sure if we can use this library... but i don't see the point of not using it
from PIL import Image


def load_raw_image(path: Path) -> np.ndarray:
    """
    Given a system path to a raw file, load data
    :param
        path: system path to the .raw file
    :return: np.ndarray
    """

    raw = rawpy.imread(path)
    return raw.raw_image


def convert_array_to_img(img_array: np.ndarray) -> Image:
    """
    Given the matrix representation of an image convert to Image object
    :param img_array: numpy matrix representation of an image.
    :return: PIL.Image instance
    """
    return Image.fromarray(img_array)


def display_img(img: Image):
    """
    Given an Image instance, display result
    :param img: PIL.Image instance
    :return: None
    """
    img.show()


def save_img(img: np.ndarray, path: str, format="jpeg") -> bool:
    """
    Given a matrix image representation save the image to the specified format
    :param img: np.ndarray matrix representation of an image
    :param path: string path to the desired final location
    :param format: image format to use (default is jpg)
    :return: True if all went smoothly
    """
    # generate path
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass

    try:
        as_img = Image.fromarray(img)
        as_img.save(path, format)
        return True
    except Exception as e:
        logging.error(f"Problem while saving image \n {e}")
        return False


# TBD: deprecate this manual functions to load an image


def read_raw_image(path: Path):
    """
    Given a system path, returns a Image (see Pillow) instance.
    We define  that a raw image must have a info.txt file with
    the image metadata in the same path level as the '.raw' file.
    :param
        path: system path to the raw file
    :return:

    """

    last_slash_position = path.rfind("/")
    # build info.txt file path
    info_path = os.path.join(path[0:last_slash_position], "info.txt")
    image_map = read_lines(info_path)
    raw_image_info = []
    image_name = path[last_slash_position + 1 :].replace(".RAW", "")
    with open(path, "rb") as binary_file:
        # Read the whole file at once
        raw_image = binary_file.read()
    raw_image_info.append(raw_image)
    raw_image_info.append(image_map[image_name])
    return raw_image_info


def read_lines(filename: str):
    """
    Given the path to a raw image metadata returns the
    :param
        filename: system path to metadata
    :return:
        image map, loaded from metadata
    """

    file1 = open(filename, "r")
    lines = file1.readlines()
    images = {}
    count = 0
    for line in lines:
        count = count + 1
        if count > 2:
            image_info = get_image_info(line)
            images[image_info[0]] = [image_info[1], image_info[2]]
    return images


def get_image_info(line):
    info = line.replace("\n", "").replace(".RAW", "").split(" ")
    image_info = []
    count = 0
    for value in info:
        if len(value) > 0:
            image_info.append(value)
            count = count + 1
    return image_info
