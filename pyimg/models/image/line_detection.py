import re
import time

import numpy as np

from pyimg.menus.io_menu import ImageIO
from pyimg.models.image import ImageImpl
from pyimg.modules.image_io import display_img


def pixel_exchange(image: ImageImpl, top_left_vertex_x: int, top_left_vertex_y: int, bottom_right_vertex_x: int,
                   bottom_right_vertex_y: int, epsilon: float, max_iterations: int) -> ImageImpl:
    mask_array = np.ones((image.height, image.width)) * 3
    lin = {}
    lout = {}
    object_color = get_object_color(mask_array, image, top_left_vertex_x, top_left_vertex_y, bottom_right_vertex_x,
                                    bottom_right_vertex_y, lin, lout)
    mask_image, lin, lout = iterate_pixel_exchange(mask_array, image, lin, lout, max_iterations, object_color, epsilon)

    result = generate_image_with_border(mask_image, image)
    return result


def pixel_exchange_in_sequence(image: ImageImpl, image_name: str, top_left_vertex_x: int, top_left_vertex_y: int,
                               bottom_right_vertex_x: int, bottom_right_vertex_y: int, epsilon: float, max_iterations:
                               int, quantity: int) -> ImageImpl:
    last_slash_pos = image_name.rfind("/")
    absolute_path = image_name[0:last_slash_pos + 1]
    image_name = image_name[last_slash_pos + 1:]
    current_number = int(re.findall(r'\d+', image_name)[0])
    start = image_name.find("" + str(current_number))
    prefix = image_name[0:start]
    suffix_start = image_name.find(".")
    extension = image_name[suffix_start:]

    for current_image_index in range(current_number, current_number + quantity):
        if current_image_index != current_number:
            image_name = absolute_path + prefix + str(current_image_index) + extension
            image = load_image(image_name)

        new_image = pixel_exchange(image, top_left_vertex_x, top_left_vertex_y, bottom_right_vertex_x,
                                   bottom_right_vertex_y, epsilon, max_iterations)
        generated_image_name = "../../videos/generated/pixel_exchange_" + prefix + str(current_image_index) + extension
        display_img(new_image.convert_to_pil())

    return new_image


def load_image(filename: str) -> ImageImpl:
    image = ImageIO.load_image(filename)
    image_matrix = np.array(image)
    dims = len(image_matrix.shape)
    image_matrix = (
        np.expand_dims(image_matrix, axis=dims) if dims == 2 else image_matrix
    )
    return ImageImpl(image_matrix)


def get_object_color(mask_array: np.ndarray, image: ImageImpl, top_left_vertex_x: int, top_left_vertex_y: int,
                     bottom_right_vertex_x: int, bottom_right_vertex_y: int, lin: dict, lout: dict) -> np.ndarray:
    color_sum = np.zeros(image.channels)
    image_array = image.get_array()

    square_height = (bottom_right_vertex_y - top_left_vertex_y) + 1
    square_width = (bottom_right_vertex_x - top_left_vertex_x) + 1
    square_size = square_height * square_width
    for y in range(top_left_vertex_y, bottom_right_vertex_y + 1):
        for x in range(top_left_vertex_x, bottom_right_vertex_x + 1):
            mask_array[y, x] = -3
            for k in range(0, image.channels):
                color_sum[k] += image_array[y, x, k]

    for y in range(top_left_vertex_y, bottom_right_vertex_y + 1):
        mask_array[y, top_left_vertex_x - 1] = -1
        lin[(top_left_vertex_x - 1, y)] = -1
        mask_array[y, bottom_right_vertex_x + 1] = -1
        lin[(bottom_right_vertex_x + 1, y)] = -1
        mask_array[y, top_left_vertex_x - 2] = 1
        lout[(top_left_vertex_x - 2, y)] = 1
        mask_array[y, bottom_right_vertex_x + 2] = 1
        lout[(bottom_right_vertex_x + 2, y)] = 1

    for x in range(top_left_vertex_x, bottom_right_vertex_x + 1):
        mask_array[top_left_vertex_y - 1, x] = -1
        lin[(x, top_left_vertex_y - 1)] = -1
        mask_array[bottom_right_vertex_y + 1, x] = -1
        lin[(x, bottom_right_vertex_y + 1)] = -1
        mask_array[top_left_vertex_y - 2, x] = 1
        lout[(x, top_left_vertex_y - 2)] = 1
        mask_array[bottom_right_vertex_y + 2, x] = 1
        lout[(x, bottom_right_vertex_y + 2)] = 1

    mask_array[top_left_vertex_y - 1, top_left_vertex_x - 1] = 1
    lout[(top_left_vertex_x - 1, top_left_vertex_y - 1)] = 1
    mask_array[bottom_right_vertex_y + 1, top_left_vertex_x - 1] = 1
    lout[(top_left_vertex_x - 1, bottom_right_vertex_y + 1)] = 1
    mask_array[bottom_right_vertex_y + 1, bottom_right_vertex_x + 1] = 1
    lout[(bottom_right_vertex_x + 1, bottom_right_vertex_y + 1)] = 1
    mask_array[top_left_vertex_y - 1, bottom_right_vertex_x + 1] = 1
    lout[(bottom_right_vertex_x + 1, top_left_vertex_y - 1)] = 1

    return color_sum / square_size


def iterate_pixel_exchange(mask_array: np.ndarray, image: ImageImpl, lin: dict, lout: dict, max_iterations: int,
                           object_color: np.ndarray, epsilon: float):
    start_time = time.time()
    for i in range(0, max_iterations):
        new_lin = {}
        new_lout = {}
        iterate_over_lout(mask_array, image, object_color, epsilon, lout, new_lout, new_lin)
        iterate_over_lin(mask_array, image, lin, new_lin, new_lout)
        second_lin = {}
        second_lout = {}
        remove_extra_lin(mask_array, image, new_lin, second_lin, second_lout, object_color, epsilon)
        remove_extra_lout(mask_array, image, new_lout, second_lin, second_lout)
        lin = second_lin
        lout = second_lout
    print(time.time() - start_time)
    return mask_array, lin, lout


def iterate_over_lout(mask_array: np.ndarray, image: ImageImpl, object_color: np.ndarray, epsilon: float, lout: dict,
                      new_lout: dict, new_lin: dict):
    directions = [[-1, 0], [0, -1], [1, 0], [0, 1]]
    for pixel in lout:
        current_x = pixel[0]
        current_y = pixel[1]
        if has_same_color_as_object(image, current_y, current_x, object_color, epsilon):
            mask_array[current_y, current_x] = -1
            new_lin[(current_x, current_y)] = -1
            for i in range(0, 4):
                x_increment = directions[i][0]
                y_increment = directions[i][1]
                if (
                    0 <= x_increment + current_x < image.width and 0 <= y_increment + current_y < image.height and
                    mask_array[current_y + y_increment, current_x + x_increment] == 3):
                    mask_array[current_y + y_increment, current_x + x_increment] = 1
                    new_lout[(current_x + x_increment, current_y + y_increment)] = 1
        else:
            new_lout[(current_x, current_y)] = 1


def has_same_color_as_object(image: ImageImpl, y: int, x: int, object_color: np.ndarray, epsilon: float):
    if image.channels == 1:
        return abs(image.array[y, x, 0] - object_color) <= epsilon
    else:
        current_color = np.zeros(3)
        current_color[0] = image.array[y, x, 0]
        current_color[1] = image.array[y, x, 1]
        current_color[2] = image.array[y, x, 2]
        value = np.linalg.norm(current_color - object_color) / (np.sqrt(3) * 256)
        return value <= epsilon


def iterate_over_lin(mask_array: np.ndarray, image: ImageImpl, lin: dict, new_lin: dict, new_lout: dict):
    directions = [[-1, 0], [0, -1], [1, 0], [0, 1]]
    for pixel in lin:
        blue_count = 0
        current_x = pixel[0]
        current_y = pixel[1]
        for i in range(0, 4):
            x_increment = directions[i][0]
            y_increment = directions[i][1]
            new_x = current_x + x_increment
            new_y = current_y + y_increment
            if 0 <= new_x < image.width and 0 <= new_y < image.height and (new_x, new_y) in new_lout:
                blue_count += 1
        if blue_count == 0:
            mask_array[current_y, current_x] = -3
        else:
            new_lin[(current_x, current_y)] = -1


def remove_extra_lin(mask_array: np.ndarray, image: ImageImpl, lin: dict, second_lin: dict, second_lout: dict,
                     object_color: np.ndarray, epsilon: float):
    directions = [[-1, 0], [0, -1], [1, 0], [0, 1]]
    for pixel in lin:
        current_x = pixel[0]
        current_y = pixel[1]
        if not has_same_color_as_object(image, current_y, current_x, object_color, epsilon):
            second_lout[(current_x, current_y)] = 1
            mask_array[current_y, current_x] = 1
            for i in range(0, 4):
                new_x = current_x + directions[i][1]
                new_y = current_y + directions[i][0]
                if 0 <= new_x < image.width and 0 <= new_y < image.height and mask_array[new_y, new_x] == -3:
                    mask_array[new_y, new_x] = -1
                    second_lin[(new_x, new_y)] = -1
        else:
            second_lin[(current_x, current_y)] = -1


def remove_extra_lout(mask_array: np.ndarray, image: ImageImpl, lout: dict, second_lin: dict, second_lout: dict):
    directions = [[-1, 0], [0, -1], [1, 0], [0, 1]]
    for pixel in lout:
        current_x = pixel[0]
        current_y = pixel[1]
        lin_count = 0
        for i in range(0, 4):
            new_x = current_x + directions[i][1]
            new_y = current_y + directions[i][0]
            if 0 <= new_x < image.width and 0 <= new_y < image.height and (new_x, new_y) in second_lin:
                lin_count += 1
        if lin_count == 0:
            mask_array[current_y, current_x] = 3
        else:
            second_lout[(current_x, current_y)] = 1


def generate_image_with_border(mask_array: np.ndarray, image: ImageImpl) -> ImageImpl:
    border_image = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    image_array = image.get_array()

    for y in range(0, image.height):
        for x in range(0, image.width):
            if mask_array[y, x] == -1 or mask_array[y, x] == 1:
                border_image[y, x, 0] = np.uint8(0)
                border_image[y, x, 1] = np.uint8(255)
                border_image[y, x, 2] = np.uint8(0)
            else:
                border_image[y, x, 0] = image_array[y, x, 0]
                border_image[y, x, 1] = image_array[y, x, 1]
                border_image[y, x, 2] = image_array[y, x, 2]

    return ImageImpl.from_array(border_image)

