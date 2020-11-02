import math
import re
import time

import numpy as np

from pyimg.config import constants
from pyimg.menus.io_menu import ImageIO
from pyimg.models.image import ImageImpl, border_detection
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

        display_img(new_image.convert_to_pil())

    return new_image


def hough_line(edges: ImageImpl):
    # Theta 0 - 180 degree
    # Calculate 'cos' and 'sin' value ahead to improve running time
    thetas = np.arange(0, 180, 1)
    cos = np.cos(np.deg2rad(thetas))
    sin = np.sin(np.deg2rad(thetas))

    # Generate a accumulator matrix to store the values
    rho_range = round(math.sqrt(edges.height**2 + edges.width**2))
    accumulator = np.zeros((2 * rho_range, len(thetas)), dtype=np.uint8)

    # Threshold to get edges pixel location (x,y)
    edge_pixels = np.where(edges.get_array()[..., 0] == 255)
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

    # Calculate rho value for each edge location (x,y) with all the theta range
    for p in range(len(coordinates)):
        for theta in range(len(thetas)):
            rho = int(round(coordinates[p][1] * cos[theta] + coordinates[p][0] * sin[theta]))
            accumulator[rho, theta] += 2 # Suppose add 1 only, Just want to get clear result

    return accumulator, rho_range, thetas


def hough_line_detector(image: ImageImpl, epsilon: float, threshold: int):

    border_image = border_detection.canny_detection(image, 3, 10, 10)

    """
    delta_rho = 1
    min_rho = - np.sqrt(2) * max(image.height, image.width)
    accumulator, rho_range, thetas = hough_line(border_image)
    """
    max_size = max(image.height, image.width)
    max_rho = np.sqrt(2) * max_size
    min_rho = - np.sqrt(2) * max_size
    delta_rho = 1
    rows = int((max_rho - min_rho) / delta_rho) + 1
    delta_theta = (15 / 180) * np.pi
    cols = int(((np.pi / 2) - (- np.pi / 2)) / delta_theta)
    cols = 6
    thetas = [0, np.pi / 8, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
    accumulator = np.zeros((rows, cols))

    edge_pixels = np.where(border_image.get_array()[..., 0] == constants.MAX_PIXEL_VALUE)
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

    for p in range(len(coordinates)):
        for rho in range(0, rows):
            for theta in range(0, cols):
                current_rho = min_rho + rho * delta_rho
                # current_theta = (-np.pi / 2) + (theta * delta_theta)
                current_theta = thetas[theta]
                value = abs(current_rho - coordinates[p][1] * np.cos(current_theta)
                            - coordinates[p][0] * np.sin(current_theta))
                if value <= epsilon:
                    accumulator[rho, theta] += 1

    result = image.to_rgb()
    for rho in range(0, rows):
        for theta in range(0, cols):
            if accumulator[rho, theta] >= threshold:
                current_rho = min_rho + rho * delta_rho
                # current_theta = (-np.pi / 2) + (theta * delta_theta)
                current_theta = thetas[theta]
                result = draw_lines(result, current_rho, current_theta)
    return result


def draw_lines(image: ImageImpl, rho: float, theta: float):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    if x1 == x2:
        for y in range(0, image.width):
            image.array[y, int(x0), 0] = constants.MAX_PIXEL_VALUE
            image.array[y, int(x0), 1] = 0
            image.array[y, int(x0), 2] = 0

    else:
        slope = (y2 - y1) / (x2 - x1)
        origin_ordenate = y0 - slope * x0
        for x in range(0, image.width):
            y = int(slope * x + origin_ordenate)
            # y = int(- ((np.cos(theta) / np.sin(theta)) * x) + rho / np.sin(theta))
            if 0 <= y < image.height:
                image.array[y, x, 0] = constants.MAX_PIXEL_VALUE
                image.array[y, x, 1] = 0
                image.array[y, x, 2] = 0
    return image


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

