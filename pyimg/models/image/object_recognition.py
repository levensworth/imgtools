from typing import List

import cv2
import numpy as np

from pyimg.models.image import ImageImpl


def sift_method(a_img: ImageImpl) -> (ImageImpl, List, List):
    gray_image = a_img.to_gray()

    sift = cv2.xfeatures2d.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(gray_image.get_array(), None)

    return gray_image, key_points, descriptors


def compare_images_sith(img1: ImageImpl, img2: ImageImpl, threshold, acceptance
                        ) -> (ImageImpl, bool, int, int, int):
    gray1, key_points1, descriptors1 = sift_method(img1)
    gray2, key_points2, descriptors2 = sift_method(img2)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    matches_qty = (np.array([x.distance for x in matches]) <= threshold).sum()

    matching_image = cv2.drawMatches(img1.get_array(), key_points1, img2.get_array(), key_points2, matches[:matches_qty],
                                     img2.get_array(), flags=2)

    min_dimension = min(len(descriptors1), len(descriptors2))
    matching_percentage = matches_qty / min_dimension

    return (ImageImpl.from_array(matching_image), matching_percentage >= acceptance, len(descriptors1),
            len(descriptors2), matches_qty)
