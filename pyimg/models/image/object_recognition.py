import math
from abc import ABC, abstractmethod
from typing import List, Union

import cv2
import numpy as np

from pyimg.models.image import ImageImpl


def sift_method(a_img: ImageImpl) -> (ImageImpl, List, List):
    gray_image = a_img.to_gray()

    sift = cv2.xfeatures2d.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(gray_image.get_array(), None)

    return gray_image, key_points, descriptors


def compare_images_sift(
    img1: ImageImpl, img2: ImageImpl, threshold: float, acceptance: float,
    similarity: Union[int, str, None], validate_second_min: bool, validate_second_threshold: float
) -> (ImageImpl, bool, int, int, int, float, float):
    gray1, key_points1, descriptors1 = sift_method(img1)
    gray2, key_points2, descriptors2 = sift_method(img2)

    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bf = MatcherBuilder.create(similarity, validate_second_min, validate_second_threshold)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.get_distance())
    matches_dist = np.array([x.distance for x in matches if x.valid])

    if len(matches_dist) != 0:
        if matches_dist.max() == matches_dist.min():
            matches_norm = matches_dist - matches_dist.min()
        elif matches_dist.max() <= 1:
            matches_norm = matches_dist
        else:
            matches_norm = (matches_dist - matches_dist.min()) / (matches_dist.max() - matches_dist.min())
        matches_qty = (matches_norm < threshold).sum()
    else:
        matches_norm = np.array([])
        matches_qty = 0

    matching_image = cv2.drawMatches(
        img1.get_array(),
        key_points1,
        img2.get_array(),
        key_points2,
        Match.Adapter.adapt_all(matches[:matches_qty]),
        img2.get_array(),
        flags=2,
    )

    min_dimension = min(len(descriptors1), len(descriptors2))
    matching_percentage = matches_qty / min_dimension

    return (
        ImageImpl.from_array(matching_image),
        matching_percentage >= acceptance,
        len(descriptors1),
        len(descriptors2),
        matches_qty,
        matches_norm.mean(),
        matches_norm.std(),
    )


class Match:
    """
    Class to represent a match between to descriptors.
    """

    def __init__(self, img_idx: int, d1_idx: int, d2_idx: int, distance: float, valid: bool = True):
        self.img_idx = img_idx
        self.d1_idx = d1_idx
        self.d2_idx = d2_idx
        self.distance = distance
        self.valid = valid

    def get_distance(self):
        return self.distance if self.valid else math.inf

    class Adapter:
        """
        Adapter to the DMatch class in the cv2 library.
        """

        @staticmethod
        def adapt(match) -> cv2.DMatch:
            return cv2.DMatch(
                match.d1_idx, match.d2_idx, match.img_idx, match.get_distance()
            )

        @staticmethod
        def adapt_all(matches: List) -> List[cv2.DMatch]:
            return [Match.Adapter.adapt(x) for x in matches]


class DistanceComputer(ABC):
    """
    @brief Compute the distance between two vectors with dim=3, the minimum distance is 0.
    """

    @abstractmethod
    def compute_distance(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        pass


class MinkowskiDistance(DistanceComputer):

    def __init__(self, p: Union[int, str, None] = 2, check_second: bool = False, second_threshold: float = 0):
        self.p = p
        self.check_second = check_second
        self.second_threshold = second_threshold

    def compute_distance(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return np.linalg.norm(v1 - v2, axis=2, ord=self.p)


class CosineDistance(DistanceComputer):

    def compute_distance(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        cosine_similarity = np.einsum('ijk,ijk->ij', v1, v2) / (np.linalg.norm(v1, axis=2) * np.linalg.norm(v2, axis=2))
        return np.ones_like(cosine_similarity) - cosine_similarity


class MatchValidator(ABC):
    """
    @brief Checks if a match is a valid match or not, it can use the list of distances to all the descriptors.
    """

    @abstractmethod
    def validate(self, matches_2d: np.ndarray) -> List[bool]:
        pass


class SecondMinValidator(MatchValidator):

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def validate(self, matches_2d: np.ndarray) -> List[bool]:
        matches_dist_min, matches_dist_min2 = np.partition(matches_2d, 1, axis=0)[0:2]
        return matches_dist_min / matches_dist_min2 < self.threshold


class Matcher(ABC):

    def __init__(self, distance_func: DistanceComputer, match_validator: MatchValidator = None):
        self.distance_func = distance_func
        self.match_validator = match_validator

    def match(self, img1_descriptors: np.ndarray, img2_descriptors: np.ndarray) -> List[Match]:
        """
        @brief Finds the best match for each descriptor in img1_descriptors.
        """
        d1_expanded, d2_expanded = Matcher.expand_descriptors(img1_descriptors, img2_descriptors)

        matches_2d = self.distance_func.compute_distance(d1_expanded, d2_expanded)
        matches_idx = matches_2d.argmin(axis=0)
        matches_dist = np.min(matches_2d, axis=0)

        valid_matches = self.valid_matches(matches_2d)

        return [
                Match(0, i, idx, dist, valid)
                for i, (idx, dist, valid) in enumerate(zip(matches_idx, matches_dist, valid_matches))
        ]

    def valid_matches(self, matches_2d: np.ndarray) -> List[bool]:
        if self.match_validator is not None:
            return self.match_validator.validate(matches_2d)
        return np.ones_like(matches_2d[0])

    @staticmethod
    def expand_descriptors(img1_descriptors: np.ndarray, img2_descriptors: np.ndarray) -> (np.ndarray, np.ndarray):
        img1_descriptors_expand = np.array([img1_descriptors] * img2_descriptors.shape[0])
        img2_descriptors_expand = np.array([img2_descriptors] * img1_descriptors.shape[0]).transpose(1, 0, 2)
        return img1_descriptors_expand, img2_descriptors_expand


class MatcherBuilder:
    MANHATTAN = 1
    EUCLIDEAN = 2
    CHEBYSHEV_PARAM = -1
    CHEBYSHEV = np.inf
    COSINE = 'cos'

    @staticmethod
    def create(p: Union[int, str, None] = 2, validate_second_min: bool = False, validate_second_threshold: float = 0.5
               ) -> Matcher:
        match_validator = None
        if p == MatcherBuilder.COSINE:
            similarity_func = CosineDistance()
        else:
            similarity_func = MinkowskiDistance(float(p) if p is not MatcherBuilder.CHEBYSHEV_PARAM
                                                else MatcherBuilder.CHEBYSHEV)
        if validate_second_min:
            match_validator = SecondMinValidator(validate_second_threshold)
        return Matcher(similarity_func, match_validator)


"""

Primera version, con un for:

for i, d1 in enumerate(query):
    match = np.array([np.linalg.norm(d1 - d2) for d2 in train])
    min_idx = match.argmin()
    matches.append(Match(0, i, min_idx, match[min_idx]))


=============================

Segunda version usando operaciones para arrays:

# query (n_descriptors1, size:128)
# train (n_descriptors2, size:128)

# query_expand - train_expand
# query_expand = np.array([query] * train.shape[0])
# query_expand (n_descriptors2, n_descriptors1, size:128)

   
query_expand:  
        0        1               nd1
0    desc1_1  desc1_2  .  .  .  (128)
1    desc1_1  desc1_2
2    desc1_1  desc1_2
3    desc1_1  desc1_2
        .        .
        .        .
        .        .
      (128)    (128)
nd2


# train_expand = np.array([train] * query.shape[0]).transpose(1,0,2)
# train_expand (n_descriptors1, n_descriptors2, size:128)
  
train_expand:  
        0        1               nd1
0    desc2_1  desc2_1  .  .  .  (128)
1    desc2_2  desc2_2
2    desc2_3  desc2_3
3    desc2_4  desc2_4
        .        .
        .        .
        .        .
      (128)    (128)
nd2

Hago la diferencia todos con todos
==> query_expand - train_expand :
              0                    1                     nd1
0    (desc1_1 - desc2_1)  (desc1_2 - desc2_1)  .  .  .  (128)
1    (desc1_1 - desc2_2)  (desc1_2 - desc2_2)
2    (desc1_1 - desc2_3)  (desc1_2 - desc2_3)
3    (desc1_1 - desc2_4)  (desc1_2 - desc2_4)
        .        .
        .        .
        .        .
      (128)    (128)
nd2

norm(query_expand - train_expand, axis=2) es de shape (n_descriptors1, n_descriptors2)
"""
